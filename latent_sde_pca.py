# -*- coding: utf-8 -*-
"""
latent_sde_pca.py
=================
Modelo gaussiano dinamico minimo en PCA space.

Este script carga embeddings cacheados de AntigenLM. No carga AntigenLM,
no recalcula embeddings, no genera secuencias y no propone variantes.

Los modelos discretos evaluados son equivalentes a discretizaciones
gaussianas simples de una SDE lineal sobre centroides mensuales en PCA.
"""

import argparse
import os
import pickle
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np
from scipy.stats import chi2


FIGURES_DIR = "figures/gisaid"
RESULTS_DIR = "results"
SUBTYPES = ("H1N1", "H3N2")
POINT_MODELS = ("persistence",)
GAUSSIAN_MODELS = ("gaussian_rw", "linear_drift_var1", "linear_drift_var2")
ALL_MODELS = POINT_MODELS + GAUSSIAN_MODELS
SUBTYPE_COLORS = {"H1N1": "#457B9D", "H3N2": "#E63946"}
MODEL_COLORS = {
    "persistence": "#6B7280",
    "gaussian_rw": "#059669",
    "linear_drift_var1": "#2563EB",
    "linear_drift_var2": "#7C3AED",
}
EXPORT_DPI = 300


def load_cache(path):
    with open(path, "rb") as f:
        payload = pickle.load(f)

    required = ("embeddings", "years", "months", "types", "records")
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(f"Cache invalido, faltan claves: {missing}")

    embeddings = np.asarray(payload["embeddings"], dtype=np.float64)
    years = np.asarray(payload["years"], dtype=int)
    months = np.asarray(payload["months"], dtype=int)
    types = np.asarray(payload["types"])
    records = payload["records"]

    lengths = {
        "embeddings": len(embeddings),
        "years": len(years),
        "months": len(months),
        "types": len(types),
        "records": len(records),
    }
    if len(set(lengths.values())) != 1:
        raise ValueError(f"Cache desalineado: {lengths}")

    return embeddings, years, months, types, records, payload.get("metadata", {})


def month_index(years, months):
    return years * 12 + (np.clip(months, 1, 12) - 1)


def month_label(idx):
    year = int(idx // 12)
    month = int(idx % 12) + 1
    return f"{year}-{month:02d}"


def pca_from_covariance(embeddings, max_dim):
    X = np.asarray(embeddings, dtype=np.float64)
    finite_mask = np.isfinite(X).all(axis=1)
    X = X[finite_mask]
    if len(X) < max_dim + 2:
        raise ValueError("Insuficientes embeddings finitos para PCA")

    mean = X.mean(axis=0, keepdims=True)
    Xc = X - mean
    cov = (Xc.T @ Xc) / max(len(Xc) - 1, 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = np.clip(eigvals[order], 0.0, None)
    eigvecs = eigvecs[:, order]
    positive = eigvals[eigvals > 1e-12]
    ratios = positive / positive.sum()
    return {
        "mean": mean,
        "components": eigvecs[:, :max_dim],
        "ratios": ratios,
    }


def project(embeddings, pca, dim):
    return (np.asarray(embeddings, dtype=np.float64) - pca["mean"]) @ pca["components"][:, :dim]


def centroid_dict(projected, month_idx):
    grouped = defaultdict(list)
    for z, idx in zip(projected, month_idx):
        grouped[int(idx)].append(z)
    centroids = {
        key: np.mean(values, axis=0)
        for key, values in grouped.items()
    }
    counts = {
        key: len(values)
        for key, values in grouped.items()
    }
    return centroids, counts


def rolling_targets(month_idx, test_start_year, test_end_year):
    start = test_start_year * 12
    end = (test_end_year + 1) * 12 - 1
    available = set(int(m) for m in month_idx)
    return [
        m for m in range(start, end + 1)
        if m in available and (m - 1) in available and (m - 2) in available
    ]


def train_pairs(centroids, origin_month):
    return [
        m for m in sorted(centroids)
        if m <= origin_month and (m - 1) in centroids
    ]


def train_triplets(centroids, origin_month):
    return [
        m for m in sorted(centroids)
        if m <= origin_month and (m - 1) in centroids and (m - 2) in centroids
    ]


def fit_ridge(features, targets, alpha):
    features = np.asarray(features, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    if len(features) == 0:
        return None
    design = np.column_stack([features, np.ones(len(features))])
    penalty = np.eye(design.shape[1]) * float(alpha)
    penalty[-1, -1] = 0.0
    lhs = design.T @ design + penalty
    rhs = design.T @ targets
    return np.linalg.solve(lhs, rhs)


def predict_ridge(coef, features):
    if coef is None:
        return None
    features = np.asarray(features, dtype=np.float64).reshape(1, -1)
    design = np.column_stack([features, np.ones(len(features))])
    return (design @ coef)[0]


def regularized_cov(residuals, dim, cov_reg, cov_type="full", cov_inflation=1.0):
    residuals = np.asarray(residuals, dtype=np.float64)
    if len(residuals) < 2:
        return np.eye(dim) * cov_reg * cov_inflation
    cov = np.cov(residuals, rowvar=False)
    cov = np.atleast_2d(cov)
    if cov.shape != (dim, dim):
        cov = np.eye(dim) * float(np.var(residuals))
    if cov_type == "diagonal":
        cov = np.diag(np.diag(cov))
    elif cov_type != "full":
        raise ValueError(f"Tipo de covarianza desconocido: {cov_type}")
    return (cov + np.eye(dim) * float(cov_reg)) * float(cov_inflation)


def gaussian_logpdf(y, mean, cov):
    diff = np.asarray(y, dtype=np.float64) - np.asarray(mean, dtype=np.float64)
    sign, logdet = np.linalg.slogdet(cov)
    if sign <= 0:
        return np.nan, np.nan, np.nan
    inv = np.linalg.inv(cov)
    md2 = float(diff.T @ inv @ diff)
    dim = len(diff)
    logp = -0.5 * (dim * np.log(2 * np.pi) + logdet + md2)
    return float(logp), float(md2), float(np.sqrt(max(md2, 0.0)))


def fit_models(centroids, origin_month, dim, ridge_alpha, cov_reg,
               cov_type="full", cov_inflation=1.0):
    pairs = train_pairs(centroids, origin_month)
    triplets = train_triplets(centroids, origin_month)

    increments = np.array([centroids[m] - centroids[m - 1] for m in pairs])
    mu_delta = increments.mean(axis=0) if len(increments) else np.zeros(dim)
    q_rw = regularized_cov(
        increments - mu_delta, dim, cov_reg,
        cov_type=cov_type, cov_inflation=cov_inflation,
    )

    x_prev = np.array([centroids[m - 1] for m in pairs])
    y_next = np.array([centroids[m] for m in pairs])
    coef_var1 = fit_ridge(x_prev, y_next, ridge_alpha)
    if coef_var1 is None:
        residual_var1 = np.empty((0, dim))
    else:
        pred_train = np.column_stack([x_prev, np.ones(len(x_prev))]) @ coef_var1
        residual_var1 = y_next - pred_train
    q_var1 = regularized_cov(
        residual_var1, dim, cov_reg,
        cov_type=cov_type, cov_inflation=cov_inflation,
    )

    if triplets:
        feat_var2 = np.array([
            np.concatenate([centroids[m - 1], centroids[m - 2]])
            for m in triplets
        ])
        y_var2 = np.array([centroids[m] for m in triplets])
        coef_var2 = fit_ridge(feat_var2, y_var2, ridge_alpha)
        if coef_var2 is None:
            residual_var2 = np.empty((0, dim))
        else:
            pred_train = np.column_stack([feat_var2, np.ones(len(feat_var2))]) @ coef_var2
            residual_var2 = y_var2 - pred_train
    else:
        coef_var2 = None
        residual_var2 = np.empty((0, dim))
    q_var2 = regularized_cov(
        residual_var2, dim, cov_reg,
        cov_type=cov_type, cov_inflation=cov_inflation,
    )

    return {
        "gaussian_rw": {"mu_delta": mu_delta, "Q": q_rw, "n_fit": len(increments)},
        "linear_drift_var1": {"coef": coef_var1, "Q": q_var1, "n_fit": len(pairs)},
        "linear_drift_var2": {"coef": coef_var2, "Q": q_var2, "n_fit": len(triplets)},
    }


def evaluate_one_target(y_true, x_t, x_tm1, fitted):
    predictions = {
        "persistence": {"mean": x_t, "Q": None},
        "gaussian_rw": {
            "mean": x_t + fitted["gaussian_rw"]["mu_delta"],
            "Q": fitted["gaussian_rw"]["Q"],
        },
        "linear_drift_var1": {
            "mean": predict_ridge(fitted["linear_drift_var1"]["coef"], x_t),
            "Q": fitted["linear_drift_var1"]["Q"],
        },
        "linear_drift_var2": {
            "mean": predict_ridge(
                fitted["linear_drift_var2"]["coef"],
                np.concatenate([x_t, x_tm1]),
            ),
            "Q": fitted["linear_drift_var2"]["Q"],
        },
    }

    out = {}
    for model, payload in predictions.items():
        mean = payload["mean"]
        if mean is None:
            mean = np.full_like(y_true, np.nan)
        err = mean - y_true
        record = {
            "mean": mean,
            "err": err,
            "loglik": np.nan,
            "nll": np.nan,
            "md2": np.nan,
            "mahalanobis": np.nan,
            "trace_Q": np.nan,
            "logdet_Q": np.nan,
        }
        Q = payload["Q"]
        if Q is not None and np.isfinite(mean).all():
            logp, md2, md = gaussian_logpdf(y_true, mean, Q)
            sign, logdet = np.linalg.slogdet(Q)
            record.update({
                "loglik": logp,
                "nll": -logp if np.isfinite(logp) else np.nan,
                "md2": md2,
                "mahalanobis": md,
                "trace_Q": float(np.trace(Q)),
                "logdet_Q": float(logdet) if sign > 0 else np.nan,
            })
        out[model] = record
    return out


def evaluate_subtype(embeddings, years, months, dim, subtype,
                     test_start_year, test_end_year, ridge_alpha, cov_reg,
                     max_dim, cov_type="full", cov_inflation=1.0):
    m_idx = month_index(years, months)
    targets = rolling_targets(m_idx, test_start_year, test_end_year)
    rows = []
    predictions_for_plot = []
    skipped = 0

    per_model = {model: [] for model in ALL_MODELS}
    for target_month in targets:
        origin = target_month - 1
        train_mask = m_idx <= origin
        eval_mask = m_idx <= target_month
        if np.sum(train_mask) <= max_dim + 2:
            skipped += 1
            continue

        pca = pca_from_covariance(embeddings[train_mask], max_dim=max_dim)
        projected = project(embeddings[eval_mask], pca, dim)
        centroids, counts = centroid_dict(projected, m_idx[eval_mask])
        if any(m not in centroids for m in (target_month, origin, origin - 1)):
            skipped += 1
            continue

        pairs = train_pairs(centroids, origin)
        if len(pairs) < 3:
            skipped += 1
            continue

        fitted = fit_models(
            centroids, origin, dim, ridge_alpha, cov_reg,
            cov_type=cov_type, cov_inflation=cov_inflation,
        )
        y_true = centroids[target_month]
        x_t = centroids[origin]
        x_tm1 = centroids[origin - 1]
        evals = evaluate_one_target(y_true, x_t, x_tm1, fitted)

        predictions_for_plot.append({
            "target_month": target_month,
            "centroids": centroids,
            "counts": counts,
            "y_true": y_true,
            "x_t": x_t,
            "evaluations": evals,
            "fitted": fitted,
        })
        for model in ALL_MODELS:
            per_model[model].append(evals[model])

    thresholds = {p: chi2.ppf(p, dim) for p in (0.50, 0.90, 0.95)}
    persistence_rmse = None
    for model in ALL_MODELS:
        payloads = per_model[model]
        errors = np.array([p["err"] for p in payloads], dtype=np.float64)
        valid = np.isfinite(errors).all(axis=1) if len(errors) else np.array([], dtype=bool)
        if len(errors) == 0 or not np.any(valid):
            row = {
                "subtype": subtype,
                "dim": dim,
                "model": model,
                "n_eval": 0,
                "rmse": np.nan,
                "mae": np.nan,
                "mean_euclidean": np.nan,
                "rmse_vs_persistence": np.nan,
                "relative_improvement": np.nan,
                "mean_loglik": np.nan,
                "mean_nll": np.nan,
                "mean_mahalanobis": np.nan,
                "coverage50": np.nan,
                "coverage90": np.nan,
                "coverage95": np.nan,
                "mean_trace_Q": np.nan,
                "mean_logdet_Q": np.nan,
                "skipped_targets": skipped,
                "cov_reg": cov_reg,
                "cov_inflation": cov_inflation,
                "cov_type": cov_type,
            }
        else:
            err = errors[valid]
            md2 = np.array([p["md2"] for p in payloads], dtype=np.float64)
            loglik = np.array([p["loglik"] for p in payloads], dtype=np.float64)
            nll = np.array([p["nll"] for p in payloads], dtype=np.float64)
            mahal = np.array([p["mahalanobis"] for p in payloads], dtype=np.float64)
            trace_q = np.array([p["trace_Q"] for p in payloads], dtype=np.float64)
            logdet_q = np.array([p["logdet_Q"] for p in payloads], dtype=np.float64)
            finite_md = np.isfinite(md2)
            row = {
                "subtype": subtype,
                "dim": dim,
                "model": model,
                "n_eval": int(len(err)),
                "rmse": float(np.sqrt(np.mean(err ** 2))),
                "mae": float(np.mean(np.abs(err))),
                "mean_euclidean": float(np.mean(np.linalg.norm(err, axis=1))),
                "rmse_vs_persistence": np.nan,
                "relative_improvement": np.nan,
                "mean_loglik": float(np.nanmean(loglik)) if np.isfinite(loglik).any() else np.nan,
                "mean_nll": float(np.nanmean(nll)) if np.isfinite(nll).any() else np.nan,
                "mean_mahalanobis": float(np.nanmean(mahal)) if np.isfinite(mahal).any() else np.nan,
                "coverage50": float(np.mean(md2[finite_md] <= thresholds[0.50])) if np.any(finite_md) else np.nan,
                "coverage90": float(np.mean(md2[finite_md] <= thresholds[0.90])) if np.any(finite_md) else np.nan,
                "coverage95": float(np.mean(md2[finite_md] <= thresholds[0.95])) if np.any(finite_md) else np.nan,
                "mean_trace_Q": float(np.nanmean(trace_q)) if np.isfinite(trace_q).any() else np.nan,
                "mean_logdet_Q": float(np.nanmean(logdet_q)) if np.isfinite(logdet_q).any() else np.nan,
                "skipped_targets": skipped,
                "cov_reg": cov_reg,
                "cov_inflation": cov_inflation,
                "cov_type": cov_type,
            }
        rows.append(row)
        if model == "persistence":
            persistence_rmse = row["rmse"]

    for row in rows:
        if persistence_rmse and np.isfinite(row["rmse"]):
            row["rmse_vs_persistence"] = row["rmse"] / persistence_rmse
            row["relative_improvement"] = (persistence_rmse - row["rmse"]) / persistence_rmse

    return rows, predictions_for_plot


def evaluate_all(embeddings, years, months, types, dims,
                 test_start_year, test_end_year, ridge_alpha, cov_reg,
                 cov_type="full", cov_inflation=1.0):
    rows = []
    plot_payload = {}
    max_dim = max(dims)
    for subtype in SUBTYPES:
        mask = types == subtype
        for dim in dims:
            subtype_rows, payload = evaluate_subtype(
                embeddings[mask],
                years[mask],
                months[mask],
                dim,
                subtype,
                test_start_year,
                test_end_year,
                ridge_alpha,
                cov_reg,
                max_dim,
                cov_type=cov_type,
                cov_inflation=cov_inflation,
            )
            rows.extend(subtype_rows)
            plot_payload[(subtype, dim)] = payload
    return rows, plot_payload


def save_figure(fig, basename):
    os.makedirs(FIGURES_DIR, exist_ok=True)
    png_path = os.path.join(FIGURES_DIR, f"{basename}.png")
    pdf_path = os.path.join(FIGURES_DIR, f"{basename}.pdf")
    fig.savefig(png_path, dpi=EXPORT_DPI, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return [png_path, pdf_path]


def plot_metric_bars(rows, metric, basename, title, ylabel, models=None):
    models = models or GAUSSIAN_MODELS
    dims = sorted(set(row["dim"] for row in rows))
    fig, axes = plt.subplots(1, len(dims), figsize=(5.4 * len(dims), 5.2), sharey=False)
    if len(dims) == 1:
        axes = [axes]
    width = 0.36
    x = np.arange(len(models))
    for ax, dim in zip(axes, dims):
        for offset, subtype in [(-width / 2, "H1N1"), (width / 2, "H3N2")]:
            vals = [
                next(
                    row[metric] for row in rows
                    if row["dim"] == dim and row["subtype"] == subtype and row["model"] == model
                )
                for model in models
            ]
            ax.bar(x + offset, vals, width=width, color=SUBTYPE_COLORS[subtype], alpha=0.82, label=subtype)
        ax.set_title(f"d={dim}")
        ax.set_xticks(x)
        ax.set_xticklabels([m.replace("_", "\n") for m in models], rotation=0, ha="center", fontsize=8)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend()
    fig.suptitle(title, y=1.04)
    plt.tight_layout()
    return save_figure(fig, basename)


def plot_coverage(rows):
    models = GAUSSIAN_MODELS
    dims = sorted(set(row["dim"] for row in rows))
    levels = [("coverage50", 0.50), ("coverage90", 0.90), ("coverage95", 0.95)]
    fig, axes = plt.subplots(len(levels), len(dims), figsize=(5.3 * len(dims), 4.0 * len(levels)), sharey=True)
    if len(dims) == 1:
        axes = axes[:, None]
    width = 0.36
    x = np.arange(len(models))
    for r, (metric, nominal) in enumerate(levels):
        for c, dim in enumerate(dims):
            ax = axes[r, c]
            for offset, subtype in [(-width / 2, "H1N1"), (width / 2, "H3N2")]:
                vals = [
                    next(
                        row[metric] for row in rows
                        if row["dim"] == dim and row["subtype"] == subtype and row["model"] == model
                    )
                    for model in models
                ]
                ax.bar(x + offset, vals, width=width, color=SUBTYPE_COLORS[subtype], alpha=0.82, label=subtype)
            ax.axhline(nominal, color="black", linewidth=1.0, linestyle="--")
            ax.set_ylim(0, 1.05)
            ax.set_title(f"d={dim}, nominal {int(nominal * 100)}%")
            ax.set_xticks(x)
            ax.set_xticklabels([m.replace("_", "\n") for m in models], fontsize=8)
            ax.set_ylabel("Cobertura empirica")
            ax.grid(axis="y", alpha=0.25)
            ax.spines[["top", "right"]].set_visible(False)
            if r == 0:
                ax.legend()
    fig.suptitle("Cobertura empirica de modelos gaussianos en PCA space", y=1.01)
    plt.tight_layout()
    return save_figure(fig, "pca_sde_coverage_by_model")


def ellipse_from_cov(mean, cov2, level, **kwargs):
    vals, vecs = np.linalg.eigh(cov2)
    vals = np.clip(vals, 0, None)
    order = np.argsort(vals)[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    scale = np.sqrt(chi2.ppf(level, 2))
    width, height = 2 * scale * np.sqrt(vals)
    return Ellipse(xy=mean[:2], width=width, height=height, angle=angle, **kwargs)


def plot_h3n2_d3_ellipses(plot_payload):
    payload = plot_payload.get(("H3N2", 3), [])
    if not payload:
        return []
    item = payload[-1]
    centroids = item["centroids"]
    months = np.array(sorted(centroids))
    X = np.array([centroids[m] for m in months])
    years = months // 12
    evals = item["evaluations"]

    fig, ax = plt.subplots(figsize=(8, 6.5))
    sc = ax.scatter(X[:, 0], X[:, 1], c=years, cmap="viridis", s=18, alpha=0.55, linewidths=0)
    ax.plot(X[:, 0], X[:, 1], color="#111827", linewidth=0.7, alpha=0.35)
    plt.colorbar(sc, ax=ax, label="Anio")

    ax.scatter(item["y_true"][0], item["y_true"][1], marker="*", s=170, color="black", label="observado")
    for model in GAUSSIAN_MODELS:
        mean = evals[model]["mean"]
        Q = item["fitted"][model]["Q"]
        color = MODEL_COLORS[model]
        ax.scatter(mean[0], mean[1], s=55, color=color, label=model)
        ell = ellipse_from_cov(mean, Q[:2, :2], 0.90, edgecolor=color, facecolor="none", linewidth=1.6, alpha=0.95)
        ax.add_patch(ell)

    ax.set_xlabel("PC1 local")
    ax.set_ylabel("PC2 local")
    ax.set_title(f"H3N2 d=3: elipses predictivas 90% para {month_label(item['target_month'])}")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    return save_figure(fig, "pca_sde_h3n2_d3_ellipses")


def format_float(value, digits=4):
    if value is None or not np.isfinite(value):
        return "NA"
    return f"{value:.{digits}f}"


def best_by_metric(rows, metric, minimize=True):
    out = {}
    for subtype in SUBTYPES:
        subset = [r for r in rows if r["subtype"] == subtype and np.isfinite(r[metric])]
        if subset:
            out[subtype] = min(subset, key=lambda r: r[metric]) if minimize else max(subset, key=lambda r: r[metric])
    return out


def write_summary(path, cache_path, rows, test_start_year, test_end_year,
                  ridge_alpha, cov_reg, figures):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    best_rmse = best_by_metric(rows, "rmse", minimize=True)
    best_nll = best_by_metric([r for r in rows if r["model"] in GAUSSIAN_MODELS], "mean_nll", minimize=True)

    lines = [
        "# SDE lineal minima / modelo gaussiano dinamico en PCA space",
        "",
        f"Fuente: `{cache_path}`.",
        "No se recalcularon embeddings, no se cargo AntigenLM y no se generaron secuencias.",
        "",
        "## Modelo",
        "",
        "- Gaussian random walk: `x[t+1] = x[t] + mu_delta + eps`, con `eps ~ N(0,Q)`.",
        "- Linear drift / VAR(1): `x[t+1] = A x[t] + b + eps`.",
        "- VAR(2): `x[t+1] = A1 x[t] + A2 x[t-1] + b + eps`.",
        "- `Q` se estima con residuos de entrenamiento y se regulariza como `Q + cov_reg I`.",
        "",
        "## Evaluacion",
        "",
        f"- Rolling-origin sobre {test_start_year}-{test_end_year}.",
        "- PCA se ajusta en cada corte usando solo datos de entrenamiento.",
        f"- Ridge alpha = {ridge_alpha}.",
        f"- Covariance regularization = {cov_reg}.",
        "",
        "## RMSE/MAE",
        "",
        "| dim | subtipo | modelo | n eval | RMSE | MAE | distancia euclidiana media | RMSE/persistence | mejora vs persistence |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['dim']} | {row['subtype']} | {row['model']} | {row['n_eval']} | "
            f"{format_float(row['rmse'])} | {format_float(row['mae'])} | "
            f"{format_float(row['mean_euclidean'])} | {format_float(row['rmse_vs_persistence'])} | "
            f"{format_float(row['relative_improvement'])} |"
        )

    lines.extend([
        "",
        "## NLL, Mahalanobis y cobertura",
        "",
        "| dim | subtipo | modelo | mean loglik | mean NLL | Mahalanobis medio | cov50 | cov90 | cov95 | trace(Q) media | logdet(Q) medio |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in rows:
        if row["model"] not in GAUSSIAN_MODELS:
            continue
        lines.append(
            f"| {row['dim']} | {row['subtype']} | {row['model']} | "
            f"{format_float(row['mean_loglik'])} | {format_float(row['mean_nll'])} | "
            f"{format_float(row['mean_mahalanobis'])} | {format_float(row['coverage50'])} | "
            f"{format_float(row['coverage90'])} | {format_float(row['coverage95'])} | "
            f"{format_float(row['mean_trace_Q'])} | {format_float(row['mean_logdet_Q'])} |"
        )

    lines.extend([
        "",
        "## Mejor modelo por subtipo",
        "",
        "| criterio | subtipo | dim | modelo | valor |",
        "|---|---|---:|---|---:|",
    ])
    for subtype, row in best_rmse.items():
        lines.append(f"| RMSE | {subtype} | {row['dim']} | {row['model']} | {format_float(row['rmse'])} |")
    for subtype, row in best_nll.items():
        lines.append(f"| NLL | {subtype} | {row['dim']} | {row['model']} | {format_float(row['mean_nll'])} |")

    lines.extend([
        "",
        "## Figuras",
        "",
    ])
    for fig in figures:
        lines.append(f"- `{fig}`")

    lines.extend([
        "",
        "## Interpretacion prudente",
        "",
        "- H1N1 debe compararse cuidadosamente contra random walk/persistence, porque el rolling-origin puntual ya sugeria drift debil.",
        "- H3N2 favorece drift lineal solo si VAR(1)/VAR(2) mejora RMSE y NLL frente al random walk.",
        "- Cobertura empirica menor que la nominal sugiere subdispersion; cobertura mayor que la nominal sugiere sobredispersion.",
        "- Este modelo probabilistico sigue operando en PCA space y no es todavia una SDE final ni una evaluacion de generacion de secuencias.",
        "",
    ])
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return path


def parse_float_list(text):
    values = tuple(float(part.strip()) for part in text.split(",") if part.strip())
    if not values:
        raise ValueError("La lista numerica no puede estar vacia")
    return values


def parse_str_list(text):
    values = tuple(part.strip() for part in text.split(",") if part.strip())
    if not values:
        raise ValueError("La lista de opciones no puede estar vacia")
    return values


def calibration_error(row):
    if row["model"] not in GAUSSIAN_MODELS:
        return np.nan
    coverages = (row["coverage50"], row["coverage90"], row["coverage95"])
    if not all(np.isfinite(v) for v in coverages):
        return np.nan
    return abs(row["coverage50"] - 0.50) + abs(row["coverage90"] - 0.90) + abs(row["coverage95"] - 0.95)


def calibration_grid_evaluation(embeddings, years, months, types, dims,
                                test_start_year, test_end_year, ridge_alpha,
                                cov_regs, cov_inflations, cov_types):
    rows = []
    total = len(cov_regs) * len(cov_inflations) * len(cov_types)
    done = 0
    for cov_type in cov_types:
        for cov_reg in cov_regs:
            for cov_inflation in cov_inflations:
                done += 1
                print(
                    f"  [{done}/{total}] cov_type={cov_type} "
                    f"cov_reg={cov_reg:g} inflation={cov_inflation:g}"
                )
                run_rows, _ = evaluate_all(
                    embeddings, years, months, types, dims,
                    test_start_year, test_end_year,
                    ridge_alpha, cov_reg,
                    cov_type=cov_type,
                    cov_inflation=cov_inflation,
                )
                for row in run_rows:
                    row["calibration_error"] = calibration_error(row)
                rows.extend(run_rows)
    return rows


def best_rows(rows, metric, subtype=None, model=None, minimize=True,
              gaussian_only=True):
    subset = [
        row for row in rows
        if np.isfinite(row.get(metric, np.nan))
        and (subtype is None or row["subtype"] == subtype)
        and (model is None or row["model"] == model)
        and (not gaussian_only or row["model"] in GAUSSIAN_MODELS)
    ]
    if not subset:
        return None
    return min(subset, key=lambda row: row[metric]) if minimize else max(subset, key=lambda row: row[metric])


def best_calibration_rows_by_group(rows, subtype):
    selected = []
    for dim in sorted(set(row["dim"] for row in rows)):
        for model in GAUSSIAN_MODELS:
            row = best_rows(
                rows, "calibration_error",
                subtype=subtype, model=model, minimize=True, gaussian_only=True,
            )
            if row is not None and row["dim"] == dim:
                selected.append(row)
            elif row is None:
                continue
            else:
                # Need best for the requested dim, not global model.
                subset = [
                    r for r in rows
                    if r["subtype"] == subtype and r["dim"] == dim
                    and r["model"] == model and np.isfinite(r["calibration_error"])
                ]
                if subset:
                    selected.append(min(subset, key=lambda r: r["calibration_error"]))
    return selected


def plot_calibration_subtype(rows, subtype):
    selected = best_calibration_rows_by_group(rows, subtype)
    labels = [
        f"d{row['dim']}\n{row['model'].replace('linear_drift_', '')}"
        for row in selected
    ]
    x = np.arange(len(selected))
    width = 0.24
    fig, ax = plt.subplots(figsize=(max(10, 0.75 * len(selected)), 5.5))
    for offset, metric, nominal, color in [
        (-width, "coverage50", 0.50, "#94A3B8"),
        (0.0, "coverage90", 0.90, "#2563EB"),
        (width, "coverage95", 0.95, "#7C3AED"),
    ]:
        vals = [row[metric] for row in selected]
        ax.bar(x + offset, vals, width=width, color=color, alpha=0.82, label=metric.replace("coverage", "cov"))
        ax.axhline(nominal, color=color, linestyle="--", linewidth=0.9, alpha=0.65)
    ax.set_ylim(0, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Cobertura empirica")
    ax.set_title(f"Calibracion mejor ajustada por modelo/dimension - {subtype}")
    ax.grid(axis="y", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(ncol=3)
    plt.tight_layout()
    return save_figure(fig, f"pca_sde_calibration_{subtype.lower()}")


def plot_nll_coverage_tradeoff(rows):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4), sharey=True)
    for ax, subtype in zip(axes, SUBTYPES):
        for model in GAUSSIAN_MODELS:
            subset = [
                row for row in rows
                if row["subtype"] == subtype and row["model"] == model
                and np.isfinite(row["coverage90"]) and np.isfinite(row["mean_nll"])
            ]
            if not subset:
                continue
            x = [row["coverage90"] for row in subset]
            y = [row["mean_nll"] for row in subset]
            sizes = [28 + 12 * (row["dim"] - 3) for row in subset]
            ax.scatter(
                x, y, s=sizes, alpha=0.62,
                color=MODEL_COLORS[model], label=model,
                edgecolors="none",
            )
        ax.axvline(0.90, color="black", linewidth=1.0, linestyle="--")
        ax.set_xlabel("Cobertura empirica 90%")
        ax.set_title(subtype)
        ax.grid(alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Mean NLL (menor es mejor)")
    fig.suptitle("Trade-off entre NLL y cobertura 90% en PCA space", y=1.03)
    plt.tight_layout()
    return save_figure(fig, "pca_sde_nll_vs_coverage_tradeoff")


def cache_label(path, metadata):
    seed = metadata.get("seed")
    max_per = metadata.get("max_per_subtype")
    if seed is not None and max_per is not None:
        return f"seed{seed}_max{max_per}"
    base = os.path.basename(path)
    return os.path.splitext(base)[0].replace("embeddings_cache_", "")


def evaluate_h3n2_diagonal_only(embeddings, years, months, types, dims,
                                test_start_year, test_end_year, ridge_alpha,
                                cov_reg, cov_inflation):
    rows = []
    mask = types == "H3N2"
    max_dim = max(dims)
    for dim in dims:
        subtype_rows, _ = evaluate_subtype(
            embeddings[mask],
            years[mask],
            months[mask],
            dim,
            "H3N2",
            test_start_year,
            test_end_year,
            ridge_alpha,
            cov_reg,
            max_dim,
            cov_type="diagonal",
            cov_inflation=cov_inflation,
        )
        rows.extend(subtype_rows)
    return rows


def robustness_evaluation(cache_paths, dims, test_start_year, test_end_year,
                          ridge_alpha):
    rows = []
    for cache_path in cache_paths:
        embeddings, years, months, types, records, metadata = load_cache(cache_path)
        label = cache_label(cache_path, metadata)
        print(
            f"\nRobustez cache={label}: embeddings={embeddings.shape} "
            f"records={len(records)}"
        )

        print("  Config principal: cov_type=full cov_reg=1e-5 inflation=1.0")
        main_rows, _ = evaluate_all(
            embeddings, years, months, types, dims,
            test_start_year, test_end_year,
            ridge_alpha, 1e-5,
            cov_type="full",
            cov_inflation=1.0,
        )
        for row in main_rows:
            row["cache_label"] = label
            row["cache_path"] = cache_path
            row["robustness_config"] = "main_full_reg1e-5"
        rows.extend(main_rows)

        print("  Complementaria H3N2: cov_type=diagonal cov_reg=1e-4 inflation=1.0")
        h3n2_diag_rows = evaluate_h3n2_diagonal_only(
            embeddings, years, months, types, dims,
            test_start_year, test_end_year,
            ridge_alpha, 1e-4, 1.0,
        )
        for row in h3n2_diag_rows:
            row["cache_label"] = label
            row["cache_path"] = cache_path
            row["robustness_config"] = "h3n2_diag_reg1e-4"
        rows.extend(h3n2_diag_rows)
    return rows


def plot_robustness_metric(rows, metric, basename, ylabel):
    main_rows = [row for row in rows if row["robustness_config"] == "main_full_reg1e-5"]
    caches = sorted(set(row["cache_label"] for row in main_rows))
    dims = sorted(set(row["dim"] for row in main_rows))
    models = list(ALL_MODELS)
    fig, axes = plt.subplots(
        len(SUBTYPES), len(dims),
        figsize=(5.2 * len(dims), 4.3 * len(SUBTYPES)),
        sharey=False,
    )
    if len(SUBTYPES) == 1:
        axes = axes[None, :]
    if len(dims) == 1:
        axes = axes[:, None]

    x = np.arange(len(models))
    width = 0.78 / max(len(caches), 1)
    for r, subtype in enumerate(SUBTYPES):
        for c, dim in enumerate(dims):
            ax = axes[r, c]
            for i, label in enumerate(caches):
                vals = []
                for model in models:
                    subset = [
                        row for row in main_rows
                        if row["cache_label"] == label
                        and row["subtype"] == subtype
                        and row["dim"] == dim
                        and row["model"] == model
                    ]
                    vals.append(subset[0][metric] if subset else np.nan)
                offset = (i - (len(caches) - 1) / 2) * width
                ax.bar(x + offset, vals, width=width, label=label, alpha=0.82)
            ax.set_title(f"{subtype}, d={dim}")
            ax.set_xticks(x)
            ax.set_xticklabels([m.replace("linear_drift_", "var").replace("_", "\n") for m in models], fontsize=8)
            ax.set_ylabel(ylabel)
            ax.grid(axis="y", alpha=0.25)
            ax.spines[["top", "right"]].set_visible(False)
            if r == 0 and c == 0:
                ax.legend(fontsize=8)
    fig.suptitle(ylabel + " por cache, subtipo y dimension", y=1.01)
    plt.tight_layout()
    return save_figure(fig, basename)


def selected_model_rows(rows):
    selected = []
    for row in rows:
        if row["robustness_config"] == "main_full_reg1e-5":
            selected.append(row)
        elif (
            row["robustness_config"] == "h3n2_diag_reg1e-4"
            and row["subtype"] == "H3N2"
            and row["model"] in ("linear_drift_var1", "linear_drift_var2", "gaussian_rw")
        ):
            selected.append(row)
    return selected


def best_robustness_by_cache(rows, cache_label_value, subtype, metric, models=None):
    subset = [
        row for row in rows
        if row["cache_label"] == cache_label_value
        and row["subtype"] == subtype
        and np.isfinite(row.get(metric, np.nan))
        and (models is None or row["model"] in models)
    ]
    if not subset:
        return None
    return min(subset, key=lambda row: row[metric])


def write_robustness_summary(path, rows, cache_paths, dims, test_start_year,
                             test_end_year, ridge_alpha, figures):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    main_rows = [row for row in rows if row["robustness_config"] == "main_full_reg1e-5"]
    selected_rows = selected_model_rows(rows)
    caches = sorted(set(row["cache_label"] for row in rows))

    lines = [
        "# Robustez externa minima de modelo gaussiano dinamico en PCA space",
        "",
        "No se recalcularon embeddings, no se cargo AntigenLM y no se generaron secuencias.",
        "",
        "## Caches comparados",
        "",
    ]
    for cache_path in cache_paths:
        lines.append(f"- `{cache_path}`")
    lines.extend([
        "",
        "## Configuracion",
        "",
        f"- Rolling-origin: {test_start_year}-{test_end_year}.",
        f"- Dimensiones: {', '.join(str(d) for d in dims)}.",
        f"- Ridge alpha: {ridge_alpha}.",
        "- Config principal: `cov_type=full`, `cov_reg=1e-5`, `cov_inflation=1.0`.",
        "- Complementaria acotada H3N2: `cov_type=diagonal`, `cov_reg=1e-4`, `cov_inflation=1.0`.",
        "",
        "## Tabla comparativa principal",
        "",
        "| cache | config | subtipo | d | modelo | n | RMSE | MAE | mean NLL | cov90 | cov95 | mejora vs persistence |",
        "|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in sorted(selected_rows, key=lambda r: (r["cache_label"], r["subtype"], r["dim"], r["robustness_config"], r["model"])):
        lines.append(
            f"| {row['cache_label']} | {row['robustness_config']} | {row['subtype']} | {row['dim']} | "
            f"{row['model']} | {row['n_eval']} | {format_float(row['rmse'])} | "
            f"{format_float(row['mae'])} | {format_float(row['mean_nll'])} | "
            f"{format_float(row['coverage90'])} | {format_float(row['coverage95'])} | "
            f"{format_float(row['relative_improvement'])} |"
        )

    lines.extend([
        "",
        "## Mejores modelos por cache",
        "",
        "| cache | subtipo | mejor RMSE | d | RMSE | mejor NLL gaussiano | d | mean NLL | cov90 | cov95 |",
        "|---|---|---|---:|---:|---|---:|---:|---:|---:|",
    ])
    for label in caches:
        for subtype in SUBTYPES:
            rmse_row = best_robustness_by_cache(main_rows, label, subtype, "rmse", models=ALL_MODELS)
            nll_candidates = [
                row for row in rows
                if row["cache_label"] == label
                and row["subtype"] == subtype
                and row["model"] in GAUSSIAN_MODELS
            ]
            nll_row = min(
                [row for row in nll_candidates if np.isfinite(row["mean_nll"])],
                key=lambda r: r["mean_nll"],
                default=None,
            )
            if rmse_row is None or nll_row is None:
                continue
            lines.append(
                f"| {label} | {subtype} | {rmse_row['model']} | {rmse_row['dim']} | "
                f"{format_float(rmse_row['rmse'])} | {nll_row['model']} ({nll_row['robustness_config']}) | "
                f"{nll_row['dim']} | {format_float(nll_row['mean_nll'])} | "
                f"{format_float(nll_row['coverage90'])} | {format_float(nll_row['coverage95'])} |"
            )

    lines.extend([
        "",
        "## Figuras",
        "",
    ])
    for fig in figures:
        lines.append(f"- `{fig}`")

    lines.extend([
        "",
        "## Lectura metodologica prudente",
        "",
        "- Si H1N1 mantiene como mejor RMSE a `persistence` o queda muy cerca de `gaussian_rw`, la lectura de drift debil se conserva.",
        "- Si H3N2 mantiene mejor RMSE y NLL con `linear_drift_var2`, la evidencia de drift lineal util es robusta frente al cambio de cache/seed.",
        "- La corrida diagonal de H3N2 es una prueba acotada de sensibilidad de covarianza, no una nueva calibracion completa.",
        "- Estos resultados siguen siendo dinamica probabilistica en PCA space; no prueban generacion ni una SDE final en el espacio completo.",
        "",
    ])
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return path


def write_calibration_summary(path, cache_path, rows, cov_regs, cov_inflations,
                              cov_types, ridge_alpha, test_start_year,
                              test_end_year, figures):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    best_rmse = {subtype: best_rows(rows, "rmse", subtype=subtype, gaussian_only=False) for subtype in SUBTYPES}
    best_nll = {subtype: best_rows(rows, "mean_nll", subtype=subtype, gaussian_only=True) for subtype in SUBTYPES}
    best_cal = {subtype: best_rows(rows, "calibration_error", subtype=subtype, gaussian_only=True) for subtype in SUBTYPES}

    lines = [
        "# Calibracion de modelo gaussiano dinamico en PCA space",
        "",
        f"Fuente: `{cache_path}`.",
        "No se recalcularon embeddings, no se cargo AntigenLM y no se generaron secuencias.",
        "",
        f"Rolling-origin: {test_start_year}-{test_end_year}.",
        f"Ridge alpha: {ridge_alpha}.",
        f"cov-reg grid: {', '.join(f'{v:g}' for v in cov_regs)}.",
        f"cov-inflation grid: {', '.join(f'{v:g}' for v in cov_inflations)}.",
        f"covariance types: {', '.join(cov_types)}.",
        "",
        "## Mejores configuraciones por subtipo",
        "",
        "| criterio | subtipo | dim | modelo | cov type | cov reg | inflation | valor | cov50 | cov90 | cov95 | calib error |",
        "|---|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for criterion, mapping, metric in [
        ("RMSE", best_rmse, "rmse"),
        ("NLL", best_nll, "mean_nll"),
        ("calibracion", best_cal, "calibration_error"),
    ]:
        for subtype in SUBTYPES:
            row = mapping.get(subtype)
            if not row:
                continue
            lines.append(
                f"| {criterion} | {subtype} | {row['dim']} | {row['model']} | "
                f"{row['cov_type']} | {row['cov_reg']:.0e} | {row['cov_inflation']:.2f} | "
                f"{format_float(row[metric])} | {format_float(row['coverage50'])} | "
                f"{format_float(row['coverage90'])} | {format_float(row['coverage95'])} | "
                f"{format_float(row.get('calibration_error'))} |"
            )

    lines.extend([
        "",
        "## Tabla completa de modelos gaussianos",
        "",
        "| dim | subtipo | modelo | cov type | cov reg | inflation | RMSE | MAE | mean NLL | Mahalanobis | cov50 | cov90 | cov95 | trace(Q) | calib error |",
        "|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    gaussian_rows = [row for row in rows if row["model"] in GAUSSIAN_MODELS]
    gaussian_rows = sorted(
        gaussian_rows,
        key=lambda r: (r["subtype"], r["dim"], r["model"], r["cov_type"], r["cov_reg"], r["cov_inflation"]),
    )
    for row in gaussian_rows:
        lines.append(
            f"| {row['dim']} | {row['subtype']} | {row['model']} | {row['cov_type']} | "
            f"{row['cov_reg']:.0e} | {row['cov_inflation']:.2f} | "
            f"{format_float(row['rmse'])} | {format_float(row['mae'])} | "
            f"{format_float(row['mean_nll'])} | {format_float(row['mean_mahalanobis'])} | "
            f"{format_float(row['coverage50'])} | {format_float(row['coverage90'])} | "
            f"{format_float(row['coverage95'])} | {format_float(row['mean_trace_Q'])} | "
            f"{format_float(row.get('calibration_error'))} |"
        )

    lines.extend([
        "",
        "## Figuras",
        "",
    ])
    for fig in figures:
        lines.append(f"- `{fig}`")

    lines.extend([
        "",
        "## Interpretacion prudente",
        "",
        "- H1N1 sigue pareciendo compatible con random walk/persistence si los drifts lineales no mejoran RMSE de forma estable.",
        "- H3N2 mantiene evidencia de drift lineal util si VAR(1)/VAR(2) conserva mejor RMSE y NLL frente a random walk.",
        "- Aumentar inflacion de covarianza suele mejorar cobertura si habia subdispersion, pero puede empeorar NLL.",
        "- La mejor configuracion para una SDE minima debe balancear error puntual, NLL y cobertura; no basta escoger solo RMSE.",
        "- Todo esto sigue ocurriendo en PCA space y no implica generacion ni validacion final de una SDE completa.",
        "",
    ])
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return path


def parse_dims(text):
    dims = tuple(int(part.strip()) for part in text.split(",") if part.strip())
    if not dims:
        raise ValueError("Debe especificarse al menos una dimension")
    return dims


def main():
    parser = argparse.ArgumentParser(
        description="Modelo gaussiano dinamico minimo en PCA space"
    )
    parser.add_argument(
        "--cache-path",
        default=os.path.join(RESULTS_DIR, "embeddings_cache_10k_per_subtype_seed42.pkl"),
    )
    parser.add_argument("--dims", default="3,4,5")
    parser.add_argument("--test-start-year", type=int, default=2019)
    parser.add_argument("--test-end-year", type=int, default=2022)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--cov-reg", type=float, default=1e-5)
    parser.add_argument("--cov-inflation", type=float, default=1.0)
    parser.add_argument("--cov-type", choices=("full", "diagonal"), default="full")
    parser.add_argument("--calibration-grid", action="store_true",
                        help="Barre cov-reg, cov-inflation y tipo de covarianza")
    parser.add_argument("--cov-reg-grid", default="1e-6,1e-5,1e-4,1e-3")
    parser.add_argument("--cov-inflation-grid", default="1.0,1.1,1.25,1.5")
    parser.add_argument("--cov-types", default="full,diagonal")
    parser.add_argument("--robustness-check", action="store_true",
                        help="Compara configuraciones seleccionadas entre caches/seed sin grilla completa")
    parser.add_argument(
        "--robustness-caches",
        default="results/embeddings_cache_10k_per_subtype_seed42.pkl,results/embeddings_cache_5k_per_subtype_seed7.pkl",
    )
    args = parser.parse_args()

    dims = parse_dims(args.dims)
    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    if args.robustness_check:
        cache_paths = parse_str_list(args.robustness_caches)
        rows = robustness_evaluation(
            cache_paths, dims,
            args.test_start_year, args.test_end_year,
            args.ridge_alpha,
        )
        figures = []
        figures.extend(plot_robustness_metric(
            rows, "rmse", "pca_sde_robustness_rmse", "RMSE"
        ))
        figures.extend(plot_robustness_metric(
            rows, "mean_nll", "pca_sde_robustness_nll", "Mean NLL"
        ))
        summary_path = write_robustness_summary(
            os.path.join(RESULTS_DIR, "pca_sde_robustness_summary.md"),
            rows,
            cache_paths,
            dims,
            args.test_start_year,
            args.test_end_year,
            args.ridge_alpha,
            figures,
        )

        print("\nMejores modelos por cache:")
        main_rows = [row for row in rows if row["robustness_config"] == "main_full_reg1e-5"]
        for label in sorted(set(row["cache_label"] for row in rows)):
            for subtype in SUBTYPES:
                rmse_row = best_robustness_by_cache(main_rows, label, subtype, "rmse", models=ALL_MODELS)
                nll_row = best_robustness_by_cache(rows, label, subtype, "mean_nll", models=GAUSSIAN_MODELS)
                if rmse_row:
                    print(
                        f"  RMSE {label} {subtype}: d={rmse_row['dim']} {rmse_row['model']} "
                        f"RMSE={format_float(rmse_row['rmse'])}"
                    )
                if nll_row:
                    print(
                        f"  NLL  {label} {subtype}: d={nll_row['dim']} {nll_row['model']} "
                        f"{nll_row['robustness_config']} NLL={format_float(nll_row['mean_nll'])} "
                        f"cov90={format_float(nll_row['coverage90'])}"
                    )

        print("\nFiguras generadas:")
        for fig in figures:
            print(f"  {fig}")
        print(f"\nResumen guardado: {summary_path}")
        print("Advertencia: robustez en PCA space; no es generacion de secuencias ni SDE final.")
        return

    embeddings, years, months, types, records, metadata = load_cache(args.cache_path)
    print(f"Cache cargado: embeddings={embeddings.shape} records={len(records)}")
    print(
        "Metadata: "
        f"sampling={metadata.get('sampling_strategy')} "
        f"seed={metadata.get('seed')} "
        f"max_per_subtype={metadata.get('max_per_subtype')}"
    )

    if args.calibration_grid:
        cov_regs = parse_float_list(args.cov_reg_grid)
        cov_inflations = parse_float_list(args.cov_inflation_grid)
        cov_types = parse_str_list(args.cov_types)
        bad_types = [cov_type for cov_type in cov_types if cov_type not in ("full", "diagonal")]
        if bad_types:
            raise ValueError(f"Tipos de covarianza invalidos: {bad_types}")

        print("\nBarrido de calibracion:")
        rows = calibration_grid_evaluation(
            embeddings, years, months, types, dims,
            args.test_start_year, args.test_end_year, args.ridge_alpha,
            cov_regs, cov_inflations, cov_types,
        )
        figures = []
        figures.extend(plot_calibration_subtype(rows, "H1N1"))
        figures.extend(plot_calibration_subtype(rows, "H3N2"))
        figures.extend(plot_nll_coverage_tradeoff(rows))
        summary_path = write_calibration_summary(
            os.path.join(RESULTS_DIR, "pca_sde_calibration_summary.md"),
            args.cache_path,
            rows,
            cov_regs,
            cov_inflations,
            cov_types,
            args.ridge_alpha,
            args.test_start_year,
            args.test_end_year,
            figures,
        )

        print("\nMejores configuraciones:")
        for criterion, metric, gaussian_only in [
            ("RMSE", "rmse", False),
            ("NLL", "mean_nll", True),
            ("calibracion", "calibration_error", True),
        ]:
            for subtype in SUBTYPES:
                row = best_rows(rows, metric, subtype=subtype, gaussian_only=gaussian_only)
                if row:
                    print(
                        f"  {criterion} {subtype}: d={row['dim']} {row['model']} "
                        f"cov={row['cov_type']} reg={row['cov_reg']:.0e} "
                        f"infl={row['cov_inflation']:.2f} value={format_float(row[metric])} "
                        f"cov90={format_float(row['coverage90'])}"
                    )

        print("\nFiguras generadas:")
        for fig in figures:
            print(f"  {fig}")
        print(f"\nResumen guardado: {summary_path}")
        print("Advertencia: calibracion en PCA space; no es generacion de secuencias ni SDE final.")
        return

    rows, plot_payload = evaluate_all(
        embeddings, years, months, types, dims,
        args.test_start_year, args.test_end_year,
        args.ridge_alpha, args.cov_reg,
        cov_type=args.cov_type,
        cov_inflation=args.cov_inflation,
    )

    print("\nResultados RMSE/MAE:")
    for row in rows:
        print(
            f"  d={row['dim']} {row['subtype']} {row['model']}: "
            f"n={row['n_eval']} RMSE={format_float(row['rmse'])} "
            f"MAE={format_float(row['mae'])} dist={format_float(row['mean_euclidean'])} "
            f"NLL={format_float(row['mean_nll'])} cov90={format_float(row['coverage90'])}"
        )

    figures = []
    figures.extend(plot_metric_bars(
        rows, "mean_nll", "pca_sde_nll_by_model",
        "Mean NLL de modelos gaussianos en PCA space", "Mean NLL",
    ))
    figures.extend(plot_coverage(rows))
    figures.extend(plot_metric_bars(
        rows, "mean_mahalanobis", "pca_sde_mahalanobis_by_model",
        "Distancia de Mahalanobis promedio", "Mahalanobis promedio",
    ))
    figures.extend(plot_h3n2_d3_ellipses(plot_payload))

    summary_path = write_summary(
        os.path.join(RESULTS_DIR, "pca_sde_summary.md"),
        args.cache_path,
        rows,
        args.test_start_year,
        args.test_end_year,
        args.ridge_alpha,
        args.cov_reg,
        figures,
    )

    print("\nFiguras generadas:")
    for fig in figures:
        print(f"  {fig}")
    print(f"\nResumen guardado: {summary_path}")
    print("Advertencia: modelo gaussiano en PCA space; no es generacion de secuencias ni SDE final.")


if __name__ == "__main__":
    main()
