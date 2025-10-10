"""
Evaluator for HPC memory predictors.

- Computes importance weights to debias a labeled subset toward a population distribution.
- Evaluates per-job metrics (waste, wall time, failure flag) for one or more predictors.
- Runs weighted Monte Carlo to get means + 95% CIs (per 1000 jobs).
- Optional plotting helpers (matplotlib, one chart per figure, no explicit colors).
- Improved plotting readability for long predictor names, long-tailed metrics, and legends.

Core deps: numpy, pandas (matplotlib is optional unless you call plotting methods).
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict, Tuple, Optional, List
import numpy as np
import pandas as pd

# ----------------------------
# Config + Result dataclasses
# ----------------------------
@dataclass
class WeightingConfig:
    nbins: int = 60
    log_scale: bool = True
    eps: float = 1e-12
    trim_extreme_bins: bool = True

@dataclass
class MCResult:
    predictor: str
    waste_per_1000_mean: float
    waste_per_1000_lo: float
    waste_per_1000_hi: float
    wall_per_1000_mean: Optional[float]
    wall_per_1000_lo: Optional[float]
    wall_per_1000_hi: Optional[float]
    failure_rate_mean: float
    failure_rate_lo: float
    failure_rate_hi: float

# ----------------------------
# Helper functions (pure)
# ----------------------------
def _make_bins(x: np.ndarray, nbins=60, log=True) -> np.ndarray:
    x = np.asarray(x, float)
    x = x[np.isfinite(x) & (x > 0)]
    if x.size == 0:
        raise ValueError("All values are non-positive or non-finite; cannot make bins.")
    xmin, xmax = float(np.min(x)), float(np.max(x))
    if log:
        return np.logspace(np.log10(xmin) * 0.999, np.log10(xmax) * 1.001, nbins + 1)
    return np.linspace(xmin * 0.999, xmax * 1.001, nbins + 1)

def _hist_density(x: np.ndarray, bins: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    h, e = np.histogram(x, bins=bins, density=True)
    return h.astype(float), e

def _assign_bins(x: np.ndarray, edges: np.ndarray) -> np.ndarray:
    idx = np.digitize(x, edges) - 1
    idx[(x < edges[0]) | (x >= edges[-1])] = -1
    return idx

def _weighted_monte_carlo(values: np.ndarray, weights: np.ndarray,
                          n_jobs: int = 1000, n_iter: int = 5000, seed: int = 0):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(values))
    draws = np.empty(n_iter, dtype=float)
    for i in range(n_iter):
        sidx = rng.choice(idx, size=n_jobs, replace=True, p=weights)
        draws[i] = float(np.mean(values[sidx]))
    mean = float(np.mean(draws))
    lo, hi = [float(v) for v in np.percentile(draws, [2.5, 97.5])]
    return mean, lo, hi, draws

# ----------------------------
# Plot helpers
# ----------------------------
def _wrap_labels(labels, width: int = 18):
    """Word-wrap tick labels to multiple lines."""
    import textwrap
    wrapped = []
    for lab in labels:
        s = str(lab) if lab is not None else ""
        wrapped.append("\n".join(textwrap.wrap(s, width=width, break_long_words=True)))
    return wrapped

def _auto_figsize(n_items: int, orientation: str = "vertical"):
    """
    Returns a (w, h) tuple scaled by item count.
    Vertical: increase width a bit; Horizontal: grow height with items.
    """
    if orientation == "horizontal":
        # more items => taller figure
        h = min(12, 3.5 + 0.32 * max(0, n_items - 3))
        w = 8
    else:
        # more items => wider figure
        w = min(14, 6 + 0.5 * max(0, n_items - 4))
        h = 5
    return (w, h)

def _fmt_thousands():
    from matplotlib.ticker import FuncFormatter
    return FuncFormatter(lambda x, pos: f"{x:,.0f}")

def _legend_space(n_items: int, max_label_len: int) -> Tuple[float, List[float]]:
    """
    Decide how much extra width to allocate and what right margin to reserve when placing the legend outside.
    Returns (extra_width, rect) where rect is [left, bottom, right, top] for tight_layout.
    """
    # Heuristic: more items and longer labels => need more width.
    extra = min(12.0, 2.2 + 0.18 * n_items + 0.06 * max(0, max_label_len - 12))
    # Reserve ~20–25% of the canvas for the legend column.
    rect = [0, 0, 0.78, 1]  # right=0.78 leaves 22% for legend
    return extra, rect

# ----------------------------
# Default metric proxies (fixed signatures)
# (swap these for production if you have better ones)
# ----------------------------
def default_total_waste(true_mem: np.ndarray,
                        base_time: Optional[np.ndarray],
                        pred_mem: np.ndarray) -> np.ndarray:
    """Over-allocation only (no penalty for under)."""
    return np.maximum(pred_mem - true_mem, 0.0)

def default_total_wall_time(true_mem: np.ndarray,
                            base_time: Optional[np.ndarray],
                            pred_mem: np.ndarray) -> np.ndarray:
    """
    If under-allocated (pred < true), assume one failure + rerun (100% time penalty).
    Requires base_time to be provided by the evaluator.
    """
    if base_time is None:
        raise ValueError("base_time is required for wall time computation.")
    under = pred_mem < true_mem
    return base_time * (1.0 + under.astype(float))

# ----------------------------
# Main class
# ----------------------------
class HPCMemoryEvaluator:
    """
    Usage:
        ev = HPCMemoryEvaluator(sample_true, pop_true, predictors, base_wall_time=..., ...)
        ev.compute_importance_weights()
        summary_df, draws = ev.evaluate(n_jobs=1000, n_iter=4000, seed=42)
        ev.save_summary_csv("summary.csv")
    """

    def __init__(
        self,
        sample_peak_true: np.ndarray,
        pop_peak_true: np.ndarray,
        predictors: Dict[str, np.ndarray],
        total_waste_fn: Callable[
            [np.ndarray, Optional[np.ndarray], np.ndarray], np.ndarray
        ],
        total_wall_time_fn: Optional[
            Callable[[np.ndarray, Optional[np.ndarray], np.ndarray], np.ndarray]
        ],
        base_wall_time: Optional[np.ndarray] = None,
        weighting_cfg: WeightingConfig = WeightingConfig(),
    ):
        # Validate/standardize inputs
        self.sample_true = np.asarray(sample_peak_true, float)
        self.pop_true = np.asarray(pop_peak_true, float)
        self.predictors = {k: np.asarray(v, float) for k, v in predictors.items()}
        if not all(len(v) == len(self.sample_true) for v in self.predictors.values()):
            raise ValueError("All predictor arrays must match the labeled sample length.")
        self.N = len(self.sample_true)

        # Optional wall time
        self.base_wall_time = None if base_wall_time is None else np.asarray(base_wall_time, float)
        if self.base_wall_time is not None and len(self.base_wall_time) != self.N:
            raise ValueError("base_wall_time length must match the labeled sample length.")

        self.weighting_cfg = weighting_cfg
        self.total_waste_fn = total_waste_fn
        self.total_wall_time_fn = total_wall_time_fn if self.base_wall_time is not None else None

        # Filled later
        self.weights: Optional[np.ndarray] = None
        self.weight_info: Optional[Dict] = None
        self.per_job_metrics: Optional[Dict[str, Dict[str, np.ndarray]]] = None
        self.summary_df: Optional[pd.DataFrame] = None
        self.draws_store: Optional[Dict[str, Dict[str, np.ndarray]]] = None

    # ---- weighting ----
    def compute_importance_weights(self) -> Tuple[np.ndarray, Dict]:
        cfg = self.weighting_cfg
        bins = _make_bins(np.concatenate([self.pop_true, self.sample_true]), nbins=cfg.nbins, log=cfg.log_scale)
        p_true, edges = _hist_density(self.pop_true, bins)
        p_samp, _ = _hist_density(self.sample_true, bins)
        ratio = (p_true + cfg.eps) / (p_samp + cfg.eps)

        if cfg.trim_extreme_bins:
            ratio[p_samp < cfg.eps * 10] = np.nan

        bidx = _assign_bins(self.sample_true, edges)
        w = np.full(self.N, np.nan, dtype=float)
        in_range = bidx >= 0
        w[in_range] = ratio[bidx[in_range]]
        valid_bins = np.where(np.isfinite(ratio))[0]
        if valid_bins.size == 0:
            w[:] = 1.0 / self.N
        else:
            for i in np.where(~np.isfinite(w))[0]:
                b = max(bidx[i], 0)
                nearest = valid_bins[np.argmin(np.abs(valid_bins - b))]
                w[i] = ratio[nearest]
        w = w / np.sum(w)

        self.weights = w
        self.weight_info = {"bins": bins, "p_true": p_true, "p_samp": p_samp, "edges": edges, "ratio": ratio}
        return w, self.weight_info

    # ---- metrics ----
    def _build_per_job_metrics(self) -> Dict[str, Dict[str, np.ndarray]]:
        if self.weights is None:
            raise RuntimeError("Call compute_importance_weights() first.")
        metrics: Dict[str, Dict[str, np.ndarray]] = {}
        for name, pred in self.predictors.items():
            waste = self.total_waste_fn(self.sample_true, self.base_wall_time, pred)
            fail = (self.sample_true > pred).astype(float)
            entry = {"waste": waste, "fail": fail}
            if self.total_wall_time_fn is not None:
                entry["wall"] = self.total_wall_time_fn(self.sample_true, self.base_wall_time, pred)  # type: ignore[arg-type]
            metrics[name] = entry
        self.per_job_metrics = metrics
        return metrics

    # ---- evaluation (weighted MC) ----
    def evaluate(self, n_jobs: int = 1000, n_iter: int = 4000, seed: int = 100
                 ) -> Tuple[pd.DataFrame, Dict[str, Dict[str, np.ndarray]]]:
        if self.weights is None:
            self.compute_importance_weights()
        if self.per_job_metrics is None:
            self._build_per_job_metrics()

        summary_rows: List[Dict] = []
        draws_store: Dict[str, Dict[str, np.ndarray]] = {}
        scale = 1000.0

        for i, (name, m) in enumerate(self.per_job_metrics.items()):
            w_mean, w_lo, w_hi, w_draws = _weighted_monte_carlo(m["waste"], self.weights, n_jobs, n_iter, seed=seed + 17 * i + 1)
            f_mean, f_lo, f_hi, f_draws = _weighted_monte_carlo(m["fail"], self.weights, n_jobs, n_iter, seed=seed + 17 * i + 3)

            wall_mean = wall_lo = wall_hi = None
            wall_draws = None
            if "wall" in m:
                t_mean, t_lo, t_hi, t_draws = _weighted_monte_carlo(m["wall"], self.weights, n_jobs, n_iter, seed=seed + 17 * i + 2)
                wall_mean, wall_lo, wall_hi = t_mean * scale, t_lo * scale, t_hi * scale
                wall_draws = t_draws * scale

            res = MCResult(
                predictor=name,
                waste_per_1000_mean=w_mean * scale, waste_per_1000_lo=w_lo * scale, waste_per_1000_hi=w_hi * scale,
                wall_per_1000_mean=wall_mean, wall_per_1000_lo=wall_lo, wall_per_1000_hi=wall_hi,
                failure_rate_mean=f_mean, failure_rate_lo=f_lo, failure_rate_hi=f_hi,
            )
            summary_rows.append(res.__dict__)

            draws_store[name] = {
                "waste": w_draws * scale,
                "fail": f_draws,
                **({"wall": wall_draws} if wall_draws is not None else {}),
            }

        self.summary_df = pd.DataFrame(summary_rows)
        self.draws_store = draws_store
        return self.summary_df, self.draws_store

    # ---- persistence ----
    def save_summary_csv(self, path: str) -> None:
        if self.summary_df is None:
            raise RuntimeError("Nothing to save. Run evaluate() first.")
        self.summary_df.to_csv(path, index=False)

    # ---- optional visuals (matplotlib only) ----
    def plot_errorbars(self, metric: str = "waste") -> None:
        """
        metric ∈ {"waste", "wall"}; shows mean +/- 95% CI per predictor (per 1000 for waste/wall).
        Uses horizontal error bars when labels are long or there are many predictors.
        """
        if self.summary_df is None:
            raise RuntimeError("Run evaluate() first.")
        import numpy as np
        import matplotlib.pyplot as plt

        if metric == "waste":
            y = "waste_per_1000_mean"; lo = "waste_per_1000_lo"; hi = "waste_per_1000_hi"
            axis_label = "Total waste per 1000 jobs (GB*hours))"
            title = "Predictor comparison — Total waste (95% CI)"
        elif metric == "wall":
            y = "wall_per_1000_mean"; lo = "wall_per_1000_lo"; hi = "wall_per_1000_hi"
            axis_label = "Total wall time per 1000 jobs (Hours)"
            title = "Predictor comparison — Wall time (95% CI)"
        else:
            raise ValueError("metric must be 'waste' or 'wall'.")

        df = self.summary_df.dropna(subset=[y]).copy()
        labels = df["predictor"].astype(str).tolist()

        # Heuristic: if any label is long or there are many predictors, go horizontal
        long_label = any(len(s) > 18 for s in labels)
        many = len(labels) > 6
        horizontal = long_label or many

        # Prepare data
        means = df[y].values
        los = means - df[lo].values
        his = df[hi].values - means

        fig_w, fig_h = _auto_figsize(len(labels), "horizontal" if horizontal else "vertical")
        plt.figure(figsize=(fig_w, fig_h))
        ax = plt.gca()

        if horizontal:
            pos = np.arange(len(labels))
            ax.errorbar(means, pos, xerr=[los, his], fmt="o", capsize=6)
            ax.set_xlabel(axis_label)
            ax.set_yticks(pos)
            ax.set_yticklabels(_wrap_labels(labels, width=22))
            ax.xaxis.set_major_formatter(_fmt_thousands())
        else:
            pos = np.arange(len(labels))
            ax.errorbar(pos, means, yerr=[los, his], fmt="o", capsize=6)
            ax.set_xticks(pos)
            ax.set_xticklabels(_wrap_labels(labels, width=18), rotation=0)
            ax.set_ylabel(axis_label)
            ax.yaxis.set_major_formatter(_fmt_thousands())

        ax.set_title(title)
        ax.grid(True, linestyle="--", alpha=0.4)
        plt.tight_layout()
        plt.show()

    def plot_weighted_ccdf(
        self,
        metric: str = "waste",
        clip_min_quantile: float = 0.0,
        log_y: bool = True,
        legend: str = "outside",
    ) -> None:
        """
        Plot the (weighted) complementary CDF (1 - CDF), which makes long tails
        much easier to compare. By default, uses log-scale on Y to highlight the tail.

        - clip_min_quantile: left-trims the very small values to focus on the right tail.
        - legend: "outside" | "inside" | "none"
        """
        if self.per_job_metrics is None or self.weights is None:
            raise RuntimeError("Run compute_importance_weights() and evaluate() first.")
        import numpy as np
        import matplotlib.pyplot as plt

        names = [n for n, m in self.per_job_metrics.items() if metric in m]
        if not names:
            raise ValueError(f"No predictors contain metric '{metric}'.")

        # Layout & legend
        base_w, base_h = _auto_figsize(len(names), "vertical")
        fig_w, fig_h = base_w, base_h
        rect = None
        if legend == "outside":
            extra_w, rect = _legend_space(len(names), max(len(s) for s in names))
            fig_w += extra_w

        plt.figure(figsize=(fig_w, fig_h))
        ax = plt.gca()

        # Build a shared x-grid from pooled values (weighted quantiles)
        pooled = []
        pooled_w = []
        for name in names:
            v = np.asarray(self.per_job_metrics[name][metric], float)
            m = np.isfinite(v)
            pooled.append(v[m])
            pooled_w.append(self.weights[m])
        pooled = np.concatenate(pooled)
        pooled_w = np.concatenate(pooled_w)

        # Optionally left-trim to focus on the right tail
        order = np.argsort(pooled)
        v_sorted = pooled[order]
        w_sorted = pooled_w[order]
        w_cum = np.cumsum(w_sorted)
        w_cum /= w_cum[-1]
        xmin = np.interp(clip_min_quantile, w_cum, v_sorted)
        xgrid = np.linspace(xmin, v_sorted[-1], 300)

        for name in names:
            v = np.asarray(self.per_job_metrics[name][metric], float)
            m = np.isfinite(v)
            v = v[m]
            w = self.weights[m]
            # Weighted ECDF
            o = np.argsort(v)
            vs = v[o]
            ws = w[o]
            wc = np.cumsum(ws)
            wc /= wc[-1]
            # For each xgrid, compute CCDF = 1 - CDF(x)
            cdf = np.interp(xgrid, vs, wc, left=0.0, right=1.0)
            ccdf = 1.0 - cdf
            ax.plot(xgrid, ccdf, label=name)

        ax.set_xlabel(f"Per-job {metric}")
        ax.set_ylabel("1 - CDF")
        ax.set_title("Complementary CDF (weighted) — tail comparison")
        if log_y:
            ax.set_yscale("log")
        ax.grid(True, linestyle="--", alpha=0.3)

        if legend == "outside":
            ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0.)
            if rect is not None:
                plt.tight_layout(rect=rect)
            else:
                plt.tight_layout()
        elif legend == "inside":
            ax.legend()
            plt.tight_layout()
        else:
            plt.tight_layout()
        plt.show()

    def plot_mc_distribution(
        self,
        metric: str = "waste",
        bins: int = 40,
        clip_quantile: Tuple[float, float] = (0.0, 0.995),
        legend: str = "outside",  # "outside" | "inside" | "none"
    ) -> None:
        """
        Overlay MC sampling distributions across predictors (per 1000 for waste/wall).
        - Uses quantile clipping to avoid very long tails squeezing the view.
        - Legend placed outside by default, but the figure expands to keep the plot wide.

        metric ∈ {"waste", "wall", "fail"}.
        """
        if self.draws_store is None:
            raise RuntimeError("Run evaluate() first.")
        import numpy as np
        import matplotlib.pyplot as plt

        names = [n for n, d in self.draws_store.items() if metric in d]
        if not names:
            raise ValueError(f"No predictors contain metric '{metric}'.")

        # Build pooled draws to get global clipping range
        pooled = []
        for name in names:
            v = np.asarray(self.draws_store[name][metric], float)
            pooled.append(v[np.isfinite(v)])
        pooled = np.concatenate(pooled)
        if pooled.size == 0:
            raise ValueError(f"No finite draws for metric '{metric}'.")

        lo_q, hi_q = clip_quantile
        lo = np.quantile(pooled, lo_q)
        hi = np.quantile(pooled, hi_q)
        clipped_pct = 100.0 * (1.0 - (hi_q - lo_q))

        # Layout & legend
        base_w, base_h = _auto_figsize(len(names), "vertical")
        fig_w, fig_h = base_w, base_h
        rect = None
        if legend == "outside":
            extra_w, rect = _legend_space(len(names), max(len(s) for s in names))
            fig_w += extra_w

        plt.figure(figsize=(fig_w, fig_h))
        ax = plt.gca()

        edges = np.linspace(lo, hi, bins + 1)
        for name in names:
            v = np.asarray(self.draws_store[name][metric], float)
            v = v[np.isfinite(v)]
            counts, edges_i = np.histogram(np.clip(v, lo, hi), bins=edges, density=True)
            centers = 0.5 * (edges_i[1:] + edges_i[:-1])
            ax.plot(centers, counts, label=name)

        pretty = {"waste": "Total waste per 1000 jobs (GB*hours))", "wall": "Total wall time per 1000 jobs (min)", "fail": "Failure rate"}
        ax.set_xlabel(pretty.get(metric, metric))
        ax.set_ylabel("Density")
        ax.set_title(
            f"Monte Carlo distribution — {pretty.get(metric, metric)}\n"
        )
        ax.grid(True, linestyle="--", alpha=0.3)

        if legend == "outside":
            ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0.)
            if rect is not None:
                plt.tight_layout(rect=rect)
            else:
                plt.tight_layout()
        elif legend == "inside":
            ax.legend()
            plt.tight_layout()
        else:
            plt.tight_layout()
        plt.show()

    def plot_tradeoff(self) -> None:
        """Pareto-ish scatter: failure rate vs waste (mean estimates) with wrapped labels and light jitter to reduce overlaps."""
        if self.summary_df is None:
            raise RuntimeError("Run evaluate() first.")
        import numpy as np
        import matplotlib.pyplot as plt

        df = self.summary_df.copy()
        x = df["wall_per_1000_mean"].values
        y = df["waste_per_1000_mean"].values
        labels = df["predictor"].astype(str).tolist()

        fig_w, fig_h = _auto_figsize(len(labels), "vertical")
        plt.figure(figsize=(fig_w, fig_h))
        ax = plt.gca()
        ax.scatter(x, y)

        # Light, deterministic jitter to reduce text overlap (no randomness across runs)
        def _jitter(val, idx, scale):
            return val + scale * (((hash(labels[idx]) % 997) / 997.0) - 0.5)

        for i, lab in enumerate(labels):
            ax.annotate(
                _wrap_labels([lab], width=22)[0],
                (x[i], y[i]),
                xytext=(_jitter(x[i], i, 0.01), _jitter(y[i], i, 0.01)),
                textcoords="data",
                fontsize=9,
                ha="left",
                va="bottom"
            )

        ax.set_xlabel("Total time per 1000 jobs (mean)")
        ax.set_ylabel("Total waste per 1000 jobs (GB*hours)) (mean)")
        ax.yaxis.set_major_formatter(_fmt_thousands())
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.set_title("Trade-off: Time vs Waste (weighted MC means)")
        plt.tight_layout()
        plt.show()


__all__ = [
    "HPCMemoryEvaluator",
    "WeightingConfig",
    "MCResult",
    "default_total_waste",
    "default_total_wall_time",
]
