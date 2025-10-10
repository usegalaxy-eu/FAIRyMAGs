
# Cell 1 — imports & config
from __future__ import annotations

import math
import json
import pathlib
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, DotProduct, ConstantKernel
from sklearn.model_selection import KFold, GroupKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.metrics import mean_squared_error, mean_absolute_error

from scipy.stats import norm

TARGET_COL = "peak_mem_in_gbs"   # change to match your dataset
JOB_ID_COL = "primary_accession"                # optional: if you have job IDs
GROUP_COL = None                 # optional: grouping for CV (e.g., user/project)

def load_your_data(stats_file,y_file) -> pd.DataFrame:
    df = pd.read_csv(stats_file, index_col=0)
    y_df = pd.read_csv(y_file, sep=",")
    y_dict = dict(zip(y_df['srr_id'], y_df[TARGET_COL]))
    df[TARGET_COL] = df.index.map(y_dict)

    # set nan values to 0
    df = df.fillna(0)
    
    return df

def infer_feature_types(df: pd.DataFrame, target_col: str) -> Tuple[List[str], List[str]]:
    """Infer numeric vs categorical columns (simple heuristic)."""
    feats = [c for c in df.columns if c != target_col]
    num_cols, cat_cols = [], []
    for c in feats:
        if pd.api.types.is_numeric_dtype(df[c]):
            num_cols.append(c)
        else:
            cat_cols.append(c)
    return num_cols, cat_cols


@dataclass
class TargetTransform:
    log_target: bool = True
    clip_min: Optional[float] = 1e-6  # avoid log(0)
    
    def transform(self, y: np.ndarray) -> np.ndarray:
        y = np.asarray(y, dtype=float)
        if self.log_target:
            y = np.maximum(y, self.clip_min if self.clip_min is not None else 0.0)
            return np.log(y)
        return y
    
    def inverse(self, y_trans: np.ndarray) -> np.ndarray:
        if self.log_target:
            return np.exp(y_trans)
        return y_trans


def build_preprocessor(df: pd.DataFrame) -> Tuple[ColumnTransformer, List[str], List[str]]:
    num_cols, cat_cols = infer_feature_types(df, TARGET_COL)
    numeric = Pipeline(steps=[
        ("scaler", StandardScaler(with_mean=True, with_std=True))
    ])
    categorical = Pipeline(steps=[
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    pre = ColumnTransformer(
        transformers=[
            ("num", numeric, num_cols),
            ("cat", categorical, cat_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False
    )
    return pre, num_cols, cat_cols


def make_kernel(n_features: int) -> Any:
    """
    ARD RBF + linear, scaled by a Constant, plus noise.
    Length_scale bounds roughly [1e-2, 1e2] after StandardScaler.
    """
    rbf = RBF(length_scale=np.ones(n_features), length_scale_bounds=(1e-2, 1e2))
    linear = DotProduct(sigma_0=1.0, sigma_0_bounds=(1e-3, 1e3))
    core = rbf + linear
    kernel = ConstantKernel(constant_value=1.0, constant_value_bounds=(1e-3, 1e3)) * core
    noise = WhiteKernel(noise_level=1e-2, noise_level_bounds=(1e-6, 1e1))
    return kernel + noise




class GPRegressorWrapper(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        log_target: bool = True,
        alpha: float = 1e-10,              # jitter for GPR stability (not noise)
        n_restarts_optimizer: int = 2,     # bump up if kernel opt gets stuck (trade-off: slower)
        normalize_y: bool = False,         # we handle target transform ourselves
        random_state: Optional[int] = 42,
    ):
        self.log_target = log_target
        self.alpha = alpha
        self.n_restarts_optimizer = n_restarts_optimizer
        self.normalize_y = normalize_y
        self.random_state = random_state
        
        self.tt = TargetTransform(log_target=log_target)
        self.pipeline_: Optional[Pipeline] = None
        self.feature_names_out_: Optional[List[str]] = None

    def _make_pipeline(self, X_df: pd.DataFrame) -> Pipeline:
        pre, num_cols, cat_cols = build_preprocessor(pd.concat([X_df, pd.Series(dtype=float, name=TARGET_COL)], axis=1))
        # We need n_features AFTER preprocessing to build ARD kernel
        # Make a temporary fit to get transformed feature count.
        pre_fit = clone(pre).fit(X_df)
        X_tmp = pre_fit.transform(X_df.iloc[:1])
        n_features = X_tmp.shape[1]
        kernel = make_kernel(n_features)
        gpr = GaussianProcessRegressor(
            kernel=kernel,
            alpha=self.alpha,
            n_restarts_optimizer=self.n_restarts_optimizer,
            normalize_y=self.normalize_y,
            random_state=self.random_state,
        )
        pipe = Pipeline(steps=[("pre", pre_fit), ("gpr", gpr)])
        # store feature names for introspection
        try:
            self.feature_names_out_ = pre_fit.get_feature_names_out().tolist()
        except Exception:
            self.feature_names_out_ = None
        return pipe

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        y_t = self.tt.transform(y)
        self.pipeline_ = self._make_pipeline(X)
        self.pipeline_.fit(X, y_t)
        return self

    def predict_distribution(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns predictive mean and std in ORIGINAL target space.
        Uses delta-method for log-target to transform mean/std.
        """
        assert self.pipeline_ is not None, "Model not fitted."
        # Get predictions in transformed space
        mean_t, std_t = self.pipeline_["gpr"].predict(self.pipeline_["pre"].transform(X), return_std=True)
        if self.log_target:
            # If Y = exp(Y_t), then E[Y] ≈ exp(mu + 0.5*sigma^2); Var[Y] ≈ (exp(sigma^2)-1)*exp(2mu+sigma^2)
            exp_term = np.exp(std_t**2)
            mean = np.exp(mean_t + 0.5 * std_t**2)
            var = (exp_term - 1.0) * np.exp(2.0 * mean_t + std_t**2)
            std = np.sqrt(var)
            return mean, std
        else:
            return mean_t, std_t

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        mean, _ = self.predict_distribution(X)
        return mean

    def predict_quantile(self, X: pd.DataFrame, q: float = 0.95) -> np.ndarray:
        """
        Gaussian assumption in transformed space; map quantile to original space.
        For log-target, Quantile(Y) = exp( mu_t + z_q * sigma_t ).
        For linear target, Quantile(Y) = mu + z_q * sigma.
        """
        assert 0.0 < q < 1.0
        z = z = norm.ppf(q)  # inverse CDF of standard normal
        pre = self.pipeline_["pre"]
        gpr = self.pipeline_["gpr"]
        mu_t, sigma_t = gpr.predict(pre.transform(X), return_std=True)
        if self.log_target:
            return np.exp(mu_t + z * sigma_t)
        else:
            return mu_t + z * sigma_t

    def save(self, path: str | pathlib.Path):
        assert self.pipeline_ is not None
        blob = {
            "pipeline": self.pipeline_,
            "log_target": self.log_target,
        }
        joblib.dump(blob, path)

    @classmethod
    def load(cls, path: str | pathlib.Path) -> "GPRegressorWrapper":
        blob = joblib.load(path)
        model = cls(log_target=blob["log_target"])
        model.pipeline_ = blob["pipeline"]
        return model



@dataclass
class CVConfig:
    n_splits: int = 5
    shuffle: bool = True
    random_state: int = 42
    quantile_for_safety: float = 0.95

def coverage_at_quantile(y_true: np.ndarray, y_q: np.ndarray) -> float:
    """Fraction of times y_true <= y_q."""
    return float(np.mean(y_true <= y_q))

def run_cv(
    df: pd.DataFrame,
    log_target: bool = True,
    group_col: Optional[str] = GROUP_COL,
    cfg: CVConfig = CVConfig()
) -> Dict[str, Any]:
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL].to_numpy(dtype=float)
    groups = df[group_col].to_numpy() if group_col is not None else None

    splitter = (GroupKFold(n_splits=cfg.n_splits) if groups is not None
                else KFold(n_splits=cfg.n_splits, shuffle=cfg.shuffle, random_state=cfg.random_state))

    metrics = []
    for fold, (tr, va) in enumerate(splitter.split(X, y, groups=groups), 1):
        X_tr, X_va = X.iloc[tr], X.iloc[va]
        y_tr, y_va = y[tr], y[va]
        model = GPRegressorWrapper(log_target=log_target, n_restarts_optimizer=2)
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_va)
        y_q = model.predict_quantile(X_va, q=cfg.quantile_for_safety)

        fold_metrics = {
            "fold": fold,
            "MAE": mean_absolute_error(y_va, y_pred),
            "RMSE": math.sqrt(mean_squared_error(y_va, y_pred)),
            "Coverage@q": coverage_at_quantile(y_va, y_q),
        }
        metrics.append(fold_metrics)
        print(f"[Fold {fold}] MAE={fold_metrics['MAE']:.3f}  RMSE={fold_metrics['RMSE']:.3f}  "
              f"Coverage@{cfg.quantile_for_safety:.2f}={fold_metrics['Coverage@q']:.3f}")

    df_metrics = pd.DataFrame(metrics)
    print("\nCV summary:\n", df_metrics.describe().T)
    return {"per_fold": df_metrics, "summary": df_metrics.describe().T.to_dict()}



def train_and_save(df: pd.DataFrame, out_path: Optional[str] = "gp_memory_model.joblib", log_target: bool = True) -> GPRegressorWrapper:
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL].to_numpy(dtype=float)
    model = GPRegressorWrapper(log_target=log_target, n_restarts_optimizer=5)
    model.fit(X, y)
    if out_path:
        model.save(out_path)
        print(f"Saved model to: {out_path}")
    return model


def predict_safe_allocation(
    model: GPRegressorWrapper,
    X_new: pd.DataFrame,
    quantile: float = 0.95,
    min_mem_gb: Optional[float] = None,
    max_mem_gb: Optional[float] = None,
) -> pd.Series:
    q_pred = model.predict_quantile(X_new, q=quantile)
    if min_mem_gb is not None:
        q_pred = np.maximum(q_pred, min_mem_gb)
    if max_mem_gb is not None:
        q_pred = np.minimum(q_pred, max_mem_gb)
    return pd.Series(q_pred, index=X_new.index, name=f"safe_mem_q{int(quantile*100)}_GB")


def learn_safety_factor(
    model: GPRegressorWrapper,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    quantile: float = 0.95,
) -> float:
    """
    Learn multiplicative factor 'k' such that P(y_true <= k * q_pred) ≈ quantile.
    Useful if predictive intervals are under/over-confident in practice.
    """
    q_pred = model.predict_quantile(X_val, q=quantile)
    # grid search on simple scale factors
    candidates = np.linspace(0.8, 1.3, 51)
    best_k, best_err = 1.0, 1e9
    for k in candidates:
        cov = coverage_at_quantile(y_val, k * q_pred)
        err = abs(cov - quantile)
        if err < best_err:
            best_err, best_k = err, k
    print(f"Calibrated safety factor k={best_k:.3f} (coverage error={best_err:.3f})")
    return float(best_k)
