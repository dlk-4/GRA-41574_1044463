import numpy as np
import pandas as pd
import seaborn as sns
import yfinance as yf
from typing import Optional
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_curve, roc_auc_score


class PairsTradingPipeline:
    """
    PairsTradingPipeline
    --------------------------
    End-to-end pipeline for pairs trading on any two assets. This class
    downloads hourly data, optionally resamples crypto to U.S. equity
    trading hours, builds the spread + rolling z-score, constructs an
    ML dataset, optimizes Z / probability thresholds, trains a model,
    and evaluates Z-only vs ML-filtered strategies.

    Main public methods
    -------------------
    download_data
        :Fetch hourly close prices from Yahoo Finance.

    resample_crypto
        :Align crypto and equity to a common intraday session.

    build_pair_dataset
        :Align two non-crypto assets to a joint hourly dataset.

    build_spread_and_zscore
        :Build spread, z-score and simple threshold-based signals.

    ml_dataset
        :Construct ML features and mean-reversion labels.

    optimize_thresholds
        :Grid-search optimal Z-score boundaries on train data.

    train_model
        :Train LR or HGB, tune hyperparameters and probability cutoffs.

    evaluate_model
        :Compare Z-only vs ML-filtered strategies on the test set.

    plot_prices
        :Plot price series for one or both assets.

    plot_spread_zscore_signals
        :Plot spread, z-score and trading signals.
    """

    def __init__(
        self,
        symbol_x: str,
        symbol_y: str,
        start: str,
        end: str,
        crypto: bool = False,
        name_x: Optional[str] = None,
        name_y: Optional[str] = None,
        tz: str = "America/New_York",
    ) -> None:

        self.symbol_x = symbol_x
        self.symbol_y = symbol_y
        self.start = start
        self.end = end
        self.crypto = crypto
        self.tz = tz

        self.name_x = name_x or symbol_x
        self.name_y = name_y or symbol_y

        self.data_raw_x = None
        self.data_raw_y = None
        self.window = None
        self.data = None

        self.window = None
        self.upper_threshold = None
        self.lower_threshold = None

        self.boundary_results = None
        self.boundary_best = None

        self.p_long_cutoff = None
        self.p_short_cutoff = None

        self.prob_results = None
        self.prob_best = None

    def download_data(self) -> None:
        """
        Download hourly close data for both assets.
        """
        self.data_raw_x = yf.download(
            self.symbol_x, start=self.start, end=self.end,
            interval="1h", progress=False
        )[["Close"]].copy()
        self.data_raw_x.columns = ["Close"]

        self.data_raw_y = yf.download(
            self.symbol_y, start=self.start, end=self.end,
            interval="1h", progress=False
        )[["Close"]].copy()
        self.data_raw_y.columns = ["Close"]

    def resample_crypto(self) -> None:
        """
        resample_crypto
        --------------------------
        New York session resampling for crypto + stock pairs.
        User must set crypto=True in constructor.
        (Reproduces Figure 1)
        """
        if not self.crypto:
            raise ValueError("resample_crypto() called but crypto=False in constructor.")

        if self.data_raw_x is None or self.data_raw_y is None:
            raise ValueError("Call download_data() before resample_crypto().")

        if self.symbol_x.endswith("-USD"):
            crypto_df = self.data_raw_x.rename(columns={"Close": self.name_x})
            other_df = self.data_raw_y.rename(columns={"Close": self.name_y})
            crypto_name = self.name_x
            other_name = self.name_y
        else:
            crypto_df = self.data_raw_y.rename(columns={"Close": self.name_y})
            other_df = self.data_raw_x.rename(columns={"Close": self.name_x})
            crypto_name = self.name_y
            other_name = self.name_x

        crypto_df.index = crypto_df.index.tz_convert(self.tz)
        crypto_df = crypto_df[crypto_df.index.dayofweek < 5]
        crypto_df = crypto_df.between_time("09:30", "16:00")
        crypto_df = crypto_df.dropna()

        other_df.index = other_df.index.tz_convert(self.tz)
        other_df.index = other_df.index + pd.Timedelta(minutes=30)
        other_df = other_df[other_df.index.dayofweek < 5]
        other_df = other_df.between_time("10:00", "16:00")
        other_df = other_df.dropna()

        merged = crypto_df.join(other_df, how="inner").dropna()

        clean = merged.copy()
        clean.reset_index(inplace=True)
        clean.rename(columns={"Datetime": "time"}, inplace=True)
        clean = clean[["time", crypto_name, other_name]]

        data = clean.copy()
        data = data.set_index(pd.to_datetime(data["time"]))
        data.drop(columns=["time"], inplace=True)

        self.data = data

    def build_pair_dataset(self) -> None:
        """
        For crypto=False: simple alignment of raw hourly closes.
        """
        if self.crypto:
            raise ValueError("build_pair_dataset() called but crypto=True in constructor. Use resample_crypto().")

        df = pd.concat(
            [
                self.data_raw_x.rename(columns={"Close": self.name_x}),
                self.data_raw_y.rename(columns={"Close": self.name_y}),
            ],
            axis=1
        ).dropna()

        self.data = df

    def build_spread_and_zscore(
        self,
        window: int = 60,
        upper: float = 1.0,
        lower: float = -1.0,
    ) -> None:
        """
        build_spread_and_zscore
        --------------------------
        Construct the spread, rolling z-score, and trading signals.

        Spread_t = price_x_t - price_y_t
        Z_t      = (Spread_t - rolling_mean_t) / rolling_std_t

        Signals (position in the spread):
            Z_t > upper  →  -1  (short spread)
            Z_t < lower  →  +1  (long spread)
            lower ≤ Z_t ≤ upper →  0  (flat)

        Parameters
        ----------
        window : int
            :Rolling window length (in bars) used for mean and standard deviation.

        upper : float
            :Upper z-score threshold for entering short positions.

        lower : float
            :Lower z-score threshold for entering long positions.

        Side effects
        ------------
        Sets
            self.spread : pd.Series
            self.spread_var : float
            self.rolling_mean : pd.Series
            self.rolling_std : pd.Series
            self.zscore : pd.Series
            self.signals : pd.Series
            self.upper_threshold : float
            self.lower_threshold : float
        """
        if self.data is None:
            raise ValueError(
                "Data not found. Build the dataset before calling build_spread_and_zscore()."
            )
        self.window = window

        spread = self.data[self.name_x] - self.data[self.name_y]
        spread_var = float(np.var(spread.dropna()))

        rolling_mean = spread.rolling(window=self.window).mean()
        rolling_std = spread.rolling(window=self.window).std()

        zscore = (spread - rolling_mean) / rolling_std

        signals = pd.Series(index=zscore.index, dtype="float64")
        signals[zscore > upper] = -1.0
        signals[zscore < lower] = 1.0
        signals[(zscore >= lower) & (zscore <= upper)] = 0.0
        signals = signals.dropna()

        self.spread = spread
        self.spread_var = spread_var
        self.rolling_mean = rolling_mean
        self.rolling_std = rolling_std
        self.zscore = zscore
        self.signals = signals
        self.upper_threshold = upper
        self.lower_threshold = lower

    def ml_dataset(self) -> None:
        """
        build_ml_dataset
        --------------------------
        Construct ML features and labels from price data and z-score logic.

        Features:
            z, abs_z, delta_z, spread_ret, vol_spread, corr, beta

        Label:
            y_t = 1 if |z_{t+1}| < |z_t| (mean reversion next step),
                = 0 otherwise.

        Side effects
        ------------
        Sets self.ml_df, self.X, self.y.
        """
        if self.data is None:
            raise ValueError("Data not found. Build dataset before build_ml_dataset().")
        if getattr(self, "window", None) is None:
            raise ValueError("self.window is None. Call build_spread_and_zscore() first.")

        w = self.window

        prices = self.data.copy()
        col_x, col_y = self.name_x, self.name_y

        ret_x = prices[col_x].pct_change()
        ret_y = prices[col_y].pct_change()

        spread = prices[col_x] - prices[col_y]

        rolling_mean = spread.rolling(window=w).mean()
        rolling_std = spread.rolling(window=w).std()
        z = (spread - rolling_mean) / rolling_std

        spread_ret = ret_x - ret_y
        delta_z = z.diff()
        vol_spread = spread.rolling(window=w).std()
        corr = ret_x.rolling(window=w).corr(ret_y)
        cov = ret_x.rolling(window=w).cov(ret_y)
        beta = cov / ret_y.rolling(window=w).var()

        features = pd.DataFrame(
            {
                "z": z,
                "abs_z": z.abs(),
                "delta_z": delta_z,
                "spread_ret": spread_ret,
                "vol_spread": vol_spread,
                "corr": corr,
                "beta": beta,
            },
            index=prices.index,
        )

        features["y"] = (features["abs_z"].shift(-1) < features["abs_z"]).astype(int)

        ml_df = features.dropna().copy()

        self.ml_df = ml_df
        self.X = ml_df.drop(columns=["y"])
        self.y = ml_df["y"]

    def optimize_thresholds(
        self,
        upper_grid,
        lower_grid,
        train_frac: float = 0.7) -> pd.DataFrame:
        """
        optimize_thresholds
        --------------------------
        Grid-search upper/lower Z-score thresholds using self.window.
        Maximizes Sharpe ratio of a simple mean-reversion strategy
        on the TRAIN sample.

        Side effects
        ------------
        Sets self.upper_threshold, self.lower_threshold,
        self.boundary_results, self.boundary_best.
        """
        if self.data is None:
            raise ValueError("No data found. Build dataset first.")
        if self.window is None:
            raise ValueError("self.window is None. Call build_spread_and_zscore() first.")

        w = self.window
        ANNUALIZATION = 252 * 6.0

        prices = self.data.copy()
        col_x, col_y = self.name_x, self.name_y

        spread = prices[col_x] - prices[col_y]
        ret_x = prices[col_x].pct_change()
        ret_y = prices[col_y].pct_change()
        spread_ret = (ret_x - ret_y).dropna()

        rolling_mean = spread.rolling(window=w).mean()
        rolling_std = spread.rolling(window=w).std()
        z_all = (spread - rolling_mean) / rolling_std

        idx = sorted(z_all.index.intersection(spread_ret.index))
        z_all = z_all.loc[idx]
        r_all = spread_ret.loc[idx]

        split_idx = int(len(z_all) * train_frac)
        split_date = z_all.index[split_idx]

        z_tr = z_all.loc[z_all.index < split_date]
        r_tr = r_all.loc[r_all.index < split_date]

        all_results = []
        best = {"score": -np.inf}

        for u in upper_grid:
            for l in lower_grid:
                if l >= u:
                    continue

                sig = pd.Series(0.0, index=z_tr.index)
                sig[z_tr > u] = -1.0
                sig[z_tr < l] = 1.0

                strat_ret = sig.shift(1).fillna(0.0) * r_tr
                equity = (1.0 + strat_ret).cumprod()

                vol = strat_ret.std()
                sharpe = (strat_ret.mean() / vol * np.sqrt(ANNUALIZATION)) if vol > 0 else 0.0

                max_dd = float((equity / equity.cummax() - 1.0).min())
                final_cum = float(equity.iloc[-1])

                all_results.append((u, l, final_cum, sharpe, max_dd))

                if sharpe > best["score"]:
                    best = {
                        "upper": float(u),
                        "lower": float(l),
                        "score": float(sharpe),
                        "sharpe": float(sharpe),
                        "cum": final_cum,
                        "max_dd": max_dd,
                    }

        results_df = pd.DataFrame(
            all_results,
            columns=["upper", "lower", "cum", "sharpe", "max_dd"]
        ).sort_values("sharpe", ascending=False)

        self.upper_threshold = best["upper"]
        self.lower_threshold = best["lower"]
        self.boundary_results = results_df
        self.boundary_best = best

        return results_df

    def train_model(self, model_type: str = "lr") -> None:
        """
        train_model
        --------------------------
        Train a classifier to predict mean-reversion probability.

        model_type:
            "lr"  - Logistic Regression
            "hgb" - HistGradientBoostingClassifier

        Uses time-series CV with gap=self.window and optimizes ROC-AUC.
        Also searches probability cutoffs on TRAIN for trading.

        Side effects
        ------------
        Sets self.model, self.model_type, self.X_train, self.X_test,
        self.y_train, self.y_test, self.ml_results,
        self.p_long_cutoff, self.p_short_cutoff, self.prob_results,
        self.prob_best.
        (Reproduces Figure 7,8)
        """
        if getattr(self, "ml_df", None) is None:
            raise ValueError("ML dataset not found. Call build_ml_dataset() first.")
        if self.window is None:
            raise ValueError("self.window is None. Call build_spread_and_zscore() first.")

        X = self.ml_df.drop(columns=["y"])
        y = self.ml_df["y"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, shuffle=False, random_state=42
        )

        self.X_train, self.X_test = X_train, X_test
        self.y_train, self.y_test = y_train, y_test

        lookback = self.window
        tscv = TimeSeriesSplit(n_splits=5, gap=lookback)

        if model_type == "lr":
            base_model = LogisticRegression()
            param_grid = {
                "C": [0.001, 0.01, 0.1, 1, 10, 100],
                "max_iter": [100, 200, 300, 400, 500],
            }

        elif model_type == "hgb":
            base_model = HistGradientBoostingClassifier(
                early_stopping=False,
                random_state=42,
            )
            param_grid = {
                "learning_rate": [0.03, 0.05, 0.07],
                "max_depth": [4, 6, 8],
                "max_leaf_nodes": [None, 31, 63],
                "min_samples_leaf": [20, 50, 100],
                "l2_regularization": [0.0, 0.5, 1.0],
            }
        else:
            raise ValueError("model_type must be 'lr' or 'hgb'.")

        grid = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            scoring="roc_auc",
            cv=tscv,
            n_jobs=-1,
            refit=True,
        )
        grid.fit(X_train, y_train)

        best_params = grid.best_params_
        best_cv_score = grid.best_score_

        if model_type == "lr":
            model = grid.best_estimator_
        else:
            model = HistGradientBoostingClassifier(
                **best_params,
                early_stopping=True,
                random_state=42,
            )
        model.fit(X_train, y_train)

        y_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc_val = roc_auc_score(y_test, y_proba)

        plt.figure(figsize=(7, 5))
        label = f"{model_type.upper()} ROC (AUC = {auc_val:.2f})"
        plt.plot(fpr, tpr, label=label)
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve — Probability of Mean Reversion")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        if model_type == "lr":
            importance = pd.Series(model.coef_[0], index=X_train.columns)
            importance = importance.sort_values()

            plt.figure(figsize=(8, 5))
            importance.plot(kind="barh")
            plt.title("Feature Importance — Logistic Regression")
            plt.xlabel("Coefficient")
            plt.tight_layout()
            plt.show()

        else:
            perm = permutation_importance(
                model,
                X_test,
                y_test,
                n_repeats=10,
                random_state=42,
                n_jobs=-1,
                scoring="roc_auc",
            )
            imp = pd.Series(perm.importances_mean, index=X_train.columns).sort_values()

            plt.figure(figsize=(8, 5))
            imp.plot(kind="barh")
            plt.title("HistGradientBoosting — Permutation Importance")
            plt.xlabel("Importance (mean decrease in ROC-AUC)")
            plt.tight_layout()
            plt.show()

        if self.upper_threshold is None or self.lower_threshold is None:
            raise ValueError(
                "Z-score thresholds not set. Run optimize_thresholds() before train_model "
                "to define self.upper_threshold and self.lower_threshold."
            )

        upper = self.upper_threshold
        lower = self.lower_threshold

        p_train = pd.Series(
            model.predict_proba(X_train)[:, 1],
            index=X_train.index,
        )

        z_train = X_train["z"].copy()

        ret_train = (
            self.data[self.name_x].pct_change() -
            self.data[self.name_y].pct_change()
        ).reindex(p_train.index)

        valid_idx = (
            p_train.dropna().index
            .intersection(z_train.dropna().index)
            .intersection(ret_train.dropna().index)
        )

        p_train = p_train.loc[valid_idx]
        z_train = z_train.loc[valid_idx]
        ret_train = ret_train.loc[valid_idx]

        ANNUALIZATION = 252 * 6.0

        def backtest_with_prob(pL, pS):
            """
            backtest_with_prob
            --------------------------
            Backtest TRAIN strategy using Z thresholds (lower, upper)
            and probability cutoffs (pL, pS) for long/short entries.
            """
            sig = np.zeros(len(p_train), dtype=float)
            zv = z_train.values
            pv = p_train.values

            sig[(zv < lower) & (pv > pL)] = +1.0
            sig[(zv > upper) & (pv > pS)] = -1.0

            sig = pd.Series(sig, index=p_train.index)

            strat_ret = sig.shift(1) * ret_train
            strat_ret = strat_ret.fillna(0.0)

            equity = (1.0 + strat_ret).cumprod()

            vol = strat_ret.std()
            sharpe = (
                (strat_ret.mean() / vol) * np.sqrt(ANNUALIZATION)
                if vol > 0 else 0.0
            )
            dd = float((equity / equity.cummax() - 1.0).min())
            cum = float(equity.iloc[-1])
            trades = int((sig.diff().abs() > 0).sum())

            return {
                "p_long": float(pL),
                "p_short": float(pS),
                "cum": cum,
                "sharpe": sharpe,
                "max_dd": dd,
                "trades": trades,
            }

        grid_long = np.round(np.arange(0.50, 0.81, 0.01), 2)
        grid_short = np.round(np.arange(0.50, 0.81, 0.01), 2)

        results = []
        best_p = {"score": -np.inf}

        for pL in grid_long:
            for pS in grid_short:
                res = backtest_with_prob(pL, pS)
                results.append(res)

                if res["sharpe"] > best_p["score"]:
                    best_p = {
                        "p_long": res["p_long"],
                        "p_short": res["p_short"],
                        "score": res["sharpe"],
                        "sharpe": res["sharpe"],
                        "cum": res["cum"],
                        "max_dd": res["max_dd"],
                        "trades": res["trades"],
                    }

        res_df = (
            pd.DataFrame(results)
            .sort_values(["sharpe", "cum"], ascending=[False, False])
            .reset_index(drop=True)
        )

        self.p_long_cutoff = best_p["p_long"]
        self.p_short_cutoff = best_p["p_short"]
        self.prob_results = res_df
        self.prob_best = best_p

        self.model = model
        self.model_type = model_type
        self.ml_results = {
            "best_params": best_params,
            "cv_auc": float(best_cv_score),
            "test_auc": float(auc_val),
        }

    def evaluate_model(self) -> None:
        """
        evaluate_model
        --------------------------
        Evaluate Z-only vs ML-filtered strategies on the TEST sample.

        Uses self.X_test, self.y_test, self.model, optimized Z thresholds
        and probability cutoffs, and the underlying price data.

        Side effects
        ------------
        Plots equity curves and stores metrics in self.eval_metrics.
        (Reproduces Figure 9,10,11,12)
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train_model() first.")
        if self.X_test is None or self.y_test is None:
            raise ValueError("Train/test split not found. Run train_model() first.")
        if self.upper_threshold is None or self.lower_threshold is None:
            raise ValueError("Z-score thresholds not set. Run optimize_thresholds() first.")
        if self.p_long_cutoff is None or self.p_short_cutoff is None:
            raise ValueError(
                "Probability thresholds not set. Ensure train_model() ran the "
                "probability grid search and set p_long_cutoff / p_short_cutoff."
            )

        ANNUALIZATION = 252 * 6.0

        def compute_metrics(equity, ret_series):
            cum = float(equity.iloc[-1])
            vol = ret_series.std()
            sharpe = (ret_series.mean() / vol) * np.sqrt(ANNUALIZATION) if vol > 0 else 0.0
            max_dd = (equity / equity.cummax() - 1.0).min()
            return cum, sharpe, float(max_dd)

        def plot_equity(curves, title="Cumulative Returns — Comparison"):
            plt.figure(figsize=(12, 6))
            for label, series in curves.items():
                series.plot(label=label)
            plt.title(title)
            plt.xlabel("Date")
            plt.ylabel("Cumulative Return")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.show()

        X_test = self.X_test
        test_idx = X_test.index

        z_test = X_test["z"].copy()

        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        p_test = pd.Series(y_pred_proba, index=test_idx, name="p")

        spread_ret_test = (
            self.data[self.name_x].pct_change()
            - self.data[self.name_y].pct_change()
        ).reindex(test_idx)

        upper = self.upper_threshold
        lower = self.lower_threshold
        p_long_opt = self.p_long_cutoff
        p_short_opt = self.p_short_cutoff

        sig_z = np.zeros(len(z_test), dtype=float)
        sig_z[z_test > upper] = -1.0
        sig_z[z_test < lower] = 1.0
        sig_z = pd.Series(sig_z, index=z_test.index)

        ret_z = (sig_z.shift(1) * spread_ret_test).fillna(0.0)
        eq_z = (1.0 + ret_z).cumprod()

        sig_ml = np.zeros(len(z_test), dtype=float)
        sig_ml[(z_test < lower) & (p_test > p_long_opt)] = 1.0
        sig_ml[(z_test > upper) & (p_test > p_short_opt)] = -1.0
        sig_ml = pd.Series(sig_ml, index=z_test.index)

        ret_ml = (sig_ml.shift(1) * spread_ret_test).fillna(0.0)
        eq_ml = (1.0 + ret_ml).cumprod()

        curves = {
            "Z-only": eq_z,
            "ML filtered": eq_ml,
        }
        title = f"Cumulative Returns — Z-only vs ML ({self.name_x} / {self.name_y})"
        plot_equity(curves, title=title)

        m_z = compute_metrics(eq_z, ret_z)
        m_ml = compute_metrics(eq_ml, ret_ml)

        def pretty(name, m):
            print(f"{name:<18}  cum={m[0]:.4f}  sharpe={m[1]:.4f}  maxDD={m[2]:.4%}")

        print(f"\n✅ TEST metrics ({self.model_type.upper()} — {self.name_x}/{self.name_y})")
        pretty("Z-only", m_z)
        pretty("ML filtered", m_ml)

        trades_z = int((sig_z.diff().abs() > 0).sum())
        trades_ml = int((sig_ml.diff().abs() > 0).sum())
        print(f"\nTrades:  Z-only={trades_z},  ML filtered={trades_ml}")

        self.eval_curves = curves
        self.eval_metrics = {
            "Z-only": {
                "cum": m_z[0],
                "sharpe": m_z[1],
                "max_dd": m_z[2],
                "trades": trades_z,
            },
            "ML filtered": {
                "cum": m_ml[0],
                "sharpe": m_ml[1],
                "max_dd": m_ml[2],
                "trades": trades_ml,
            },
        }

    def plot_prices(self, assets=None):
        """
        plot_prices
        --------------------------
        Plot raw or processed price series for selected assets.

        Parameters
        ----------
        assets : list or None
            :If None, plot both name_x and name_y. Otherwise, plot the
            specified subset of columns in self.data.
        """
        if self.data is None:
            raise ValueError("No data available. Run download_data() and resample/build first.")

        if assets is None:
            assets = [self.name_x, self.name_y]

        missing = [a for a in assets if a not in self.data.columns]
        if missing:
            raise ValueError(
                f"Columns not found in data: {missing}."
                f"Available columns: {list(self.data.columns)}"
            )

        self.data[self.name_x].plot(figsize=(14,6), title=self.name_x)

        plt.title(f"Price Series — {self.name_x}")
        plt.xlabel("Time")
        plt.ylabel("Price")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

    def plot_spread_zscore_signals(self) -> None:
        """
        plot_spread_zscore_signals
        --------------------------
        Plot price spread, z-score with thresholds, and trading signals
        on aligned subplots.

        Requires build_spread_and_zscore() to be called beforehand.
        (Reproduces Figure 4,5,6)
        """
        if getattr(self, "spread", None) is None or getattr(self, "zscore", None) is None:
            raise ValueError("Spread and z-score not found. Call build_spread_and_zscore() first.")
        if getattr(self, "signals", None) is None:
            raise ValueError("Signals not found. build_spread_and_zscore() should also create signals.")

        spread = self.spread
        z = self.zscore
        sig = self.signals

        upper = self.upper_threshold if getattr(self, "upper_threshold", None) is not None else 1.0
        lower = self.lower_threshold if getattr(self, "lower_threshold", None) is not None else -1.0

        plt.figure(figsize=(14, 4))
        plt.plot(spread.index, spread, label=f"{self.name_x} - {self.name_y}")
        plt.axhline(spread.mean(), color="black", linestyle="--", label="Mean")
        plt.ylabel("Spread")
        plt.title(f"Price Spread — {self.name_x} vs {self.name_y}")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(14, 4))
        plt.plot(z.index, z, label="Z-score")
        plt.axhline(0.0, color="black", linestyle="--", label="Mean")
        plt.axhline(upper, color="red", linestyle="--", label=f"Upper ({upper:.2f})")
        plt.axhline(lower, color="green", linestyle="--", label=f"Lower ({lower:.2f})")
        plt.ylabel("Z-score")
        plt.title("Z-score of Spread")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(14, 4))
        plt.plot(sig.index, sig, label="Signal", linewidth=1)
        plt.axhline(0.0, color="black", linestyle="--", label="Neutral")
        plt.axhline(1.0, color="green", linestyle="--", label="Long")
        plt.axhline(-1.0, color="red", linestyle="--", label="Short")
        plt.ylabel("Signal")
        plt.xlabel("Date")
        plt.title("Trading Signal (based on Z-score)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()
    
    def covariance(self):
        """
        Generate covariance matrix with 5 different assets.
        (Reproduces Figure 3)
        """
        assets = {
            'IBIT': "Bitcoin ETF",
            'MSTR': 'MicroStrategy',
            '^GSPC': 'S&P 500',
            'IVV': 'S&P 500 ETF'
        }

        data = yf.download(list(assets.keys()), start="2024-01-11", end="2025-10-28", progress=False)['Close']
        data.rename(columns=assets, inplace=True)

        btc = yf.download(
            'BTC-USD',
            interval='1h',
            start='2024-01-11',
            end='2025-10-28',
            progress=False
        )['Close']

        # Convert to New York time (handles daylight saving)
        btc_ny = btc.tz_convert('America/New_York')

        # Keep only 4 PM ET closes
        btc_4pm = btc_ny[btc_ny.index.hour == 16]

        # Make it timezone-naive and align to business days
        btc_4pm = btc_4pm.tz_localize(None)
        btc_4pm.index = btc_4pm.index.normalize()
        btc_4pm = btc_4pm.asfreq('B')
        btc_4pm.name = 'Bitcoin'

        assets = {
            'BTC-USD': "Bitcoin",
            'IBIT': "Bitcoin ETF",
            'MSTR': 'MicroStrategy',
            '^GSPC': 'S&P 500',
            'IVV': 'S&P 500 ETF'
        }

        data = data.join(btc_4pm, how='inner')

        # Clean and name index
        data.index.name = 'Date'

        data.rename(columns=assets, inplace=True)

        returns = np.log(data / data.shift(1))

        corr_matrix = returns.corr()

        plt.figure(figsize=(8,6))
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
        plt.title("Correlation Matrix of Daily Log Returns")
        plt.show()
            