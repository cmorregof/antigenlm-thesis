# -*- coding: utf-8 -*-
"""
latent_dynamics_pca.py
======================
Modelos dinamicos minimos sobre centroides mensuales en PCA space.

Este script carga embeddings cacheados de AntigenLM. No carga AntigenLM,
no recalcula embeddings, no genera secuencias y no optimiza variantes.
Evalua senal dinamica retrospectiva antes de formular una SDE.
"""

import argparse
import os
import pickle
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


FIGURES_DIR = "figures/gisaid"
RESULTS_DIR = "results"
SUBTYPES = ("H1N1", "H3N2")
MODELS = ("persistence", "constant_velocity", "var1", "gaussian_rw_mean")
ROLLING_MODELS = (
    "persistence",
    "constant_velocity",
    "ridge_var1",
    "ridge_var2",
    "gaussian_rw_mean",
)
SUBTYPE_COLORS = {"H1N1": "#457B9D", "H3N2": "#E63946"}
MODEL_COLORS = {
    "persistence": "#6B7280",
    "constant_velocity": "#D97706",
    "var1": "#2563EB",
    "ridge_var1": "#2563EB",
    "ridge_var2": "#7C3AED",
    "gaussian_rw_mean": "#059669",
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
    if len(X) < 2:
        raise ValueError("Insuficientes embeddings finitos para PCA")

    mean = X.mean(axis=0, keepdims=True)
    X_centered = X - mean
    cov = (X_centered.T @ X_centered) / max(len(X_centered) - 1, 1)
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
        "n": int(len(X)),
        "dim": int(X.shape[1]),
    }


def project(embeddings, pca, dim):
    X = np.asarray(embeddings, dtype=np.float64)
    return (X - pca["mean"]) @ pca["components"][:, :dim]


def monthly_centroids(Z, years, months, types, subtype):
    mask = types == subtype
    Z_sub = Z[mask]
    month_idx = month_index(years[mask], months[mask])
    grouped = defaultdict(list)
    for z, idx in zip(Z_sub, month_idx):
        grouped[int(idx)].append(z)

    keys = np.array(sorted(grouped), dtype=int)
    X = np.array([np.mean(grouped[key], axis=0) for key in keys], dtype=np.float64)
    counts = np.array([len(grouped[key]) for key in keys], dtype=int)
    return {
        "subtype": subtype,
        "month_index": keys,
        "years": keys // 12,
        "months": keys % 12 + 1,
        "X": X,
        "counts": counts,
    }


def centroid_dict(projected, month_idx):
    grouped = defaultdict(list)
    for z, idx in zip(projected, month_idx):
        grouped[int(idx)].append(z)
    return {
        key: np.mean(values, axis=0)
        for key, values in grouped.items()
    }, {
        key: len(values)
        for key, values in grouped.items()
    }


def train_transitions_from_centroids(centroids, origin_month):
    months = sorted(m for m in centroids if m <= origin_month)
    pairs = [
        m for m in months
        if (m - 1) in centroids and m <= origin_month
    ]
    triplets = [
        m for m in months
        if (m - 1) in centroids and (m - 2) in centroids and m <= origin_month
    ]
    return pairs, triplets


def consecutive_pair_indices(series, train_end_year, test_start_year, test_end_year):
    idx = series["month_index"]
    years = series["years"]
    train_pairs = [
        i for i in range(1, len(idx))
        if idx[i] - idx[i - 1] == 1 and years[i] <= train_end_year
    ]
    # Requiere dos gaps mensuales consecutivos para evaluar todos los modelos,
    # incluida velocidad constante, sobre el mismo conjunto de targets.
    test_triplets = [
        i for i in range(2, len(idx))
        if (
            test_start_year <= years[i] <= test_end_year
            and idx[i] - idx[i - 1] == 1
            and idx[i - 1] - idx[i - 2] == 1
        )
    ]
    return train_pairs, test_triplets


def fit_var1(series, train_pairs):
    X = series["X"]
    if len(train_pairs) < X.shape[1] + 2:
        return None
    prev = np.array([X[i - 1] for i in train_pairs])
    target = np.array([X[i] for i in train_pairs])
    design = np.column_stack([prev, np.ones(len(prev))])
    coef, *_ = np.linalg.lstsq(design, target, rcond=None)
    return coef


def fit_increment_gaussian(series, train_pairs, eps=1e-6):
    X = series["X"]
    if len(train_pairs) < 2:
        return None
    increments = np.array([X[i] - X[i - 1] for i in train_pairs])
    mean = increments.mean(axis=0)
    cov = np.cov(increments, rowvar=False)
    cov = np.atleast_2d(cov)
    scale = float(np.trace(cov) / max(cov.shape[0], 1))
    cov = cov + np.eye(cov.shape[0]) * (eps * max(scale, 1.0))
    return {"mean": mean, "cov": cov, "train_increments": increments}


def fit_increment_gaussian_array(increments, eps=1e-6):
    increments = np.asarray(increments, dtype=np.float64)
    if len(increments) < 2:
        return None
    mean = increments.mean(axis=0)
    cov = np.cov(increments, rowvar=False)
    cov = np.atleast_2d(cov)
    scale = float(np.trace(cov) / max(cov.shape[0], 1))
    cov = cov + np.eye(cov.shape[0]) * (eps * max(scale, 1.0))
    return {"mean": mean, "cov": cov, "train_increments": increments}


def gaussian_log_likelihood(increments, gaussian):
    if gaussian is None or len(increments) == 0:
        return np.nan, np.nan
    mean = gaussian["mean"]
    cov = gaussian["cov"]
    sign, logdet = np.linalg.slogdet(cov)
    if sign <= 0:
        return np.nan, np.nan
    inv = np.linalg.inv(cov)
    centered = increments - mean
    quad = np.einsum("ij,jk,ik->i", centered, inv, centered)
    dim = increments.shape[1]
    logp = -0.5 * (dim * np.log(2 * np.pi) + logdet + quad)
    return float(np.mean(logp)), float(np.sum(logp))


def fit_ridge_regression(features, targets, alpha):
    features = np.asarray(features, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    if len(features) == 0:
        return None
    design = np.column_stack([features, np.ones(len(features))])
    penalty = np.eye(design.shape[1]) * float(alpha)
    penalty[-1, -1] = 0.0  # no penalizar intercepto
    lhs = design.T @ design + penalty
    rhs = design.T @ targets
    return np.linalg.solve(lhs, rhs)


def predict_ridge(coef, features):
    if coef is None:
        return None
    features = np.asarray(features, dtype=np.float64).reshape(1, -1)
    design = np.column_stack([features, np.ones(len(features))])
    return (design @ coef)[0]


def model_predictions(series, test_indices, var_coef, gaussian):
    X = series["X"]
    Y = np.array([X[i] for i in test_indices])
    prev = np.array([X[i - 1] for i in test_indices])
    prev2 = np.array([X[i - 2] for i in test_indices])

    preds = {
        "persistence": prev,
        "constant_velocity": prev + (prev - prev2),
    }
    if var_coef is None:
        preds["var1"] = np.full_like(Y, np.nan)
    else:
        design = np.column_stack([prev, np.ones(len(prev))])
        preds["var1"] = design @ var_coef

    if gaussian is None:
        preds["gaussian_rw_mean"] = np.full_like(Y, np.nan)
    else:
        preds["gaussian_rw_mean"] = prev + gaussian["mean"]

    test_increments = Y - prev
    mean_ll, total_ll = gaussian_log_likelihood(test_increments, gaussian)
    return Y, preds, test_increments, mean_ll, total_ll


def metrics(y_true, y_pred):
    valid = np.isfinite(y_pred).all(axis=1)
    if not np.any(valid):
        return {
            "n_eval": 0,
            "rmse": np.nan,
            "mae": np.nan,
            "mean_euclidean": np.nan,
        }
    diff = y_pred[valid] - y_true[valid]
    return {
        "n_eval": int(len(diff)),
        "rmse": float(np.sqrt(np.mean(diff ** 2))),
        "mae": float(np.mean(np.abs(diff))),
        "mean_euclidean": float(np.mean(np.linalg.norm(diff, axis=1))),
    }


def evaluate_dimension(projected, years, months, types, dim,
                       train_end_year, test_start_year, test_end_year):
    results = []
    series_by_subtype = {}
    increment_payload = {}

    for subtype in SUBTYPES:
        series = monthly_centroids(projected[:, :dim], years, months, types, subtype)
        train_pairs, test_indices = consecutive_pair_indices(
            series, train_end_year, test_start_year, test_end_year
        )
        var_coef = fit_var1(series, train_pairs)
        gaussian = fit_increment_gaussian(series, train_pairs)
        y_true, preds, test_increments, mean_ll, total_ll = model_predictions(
            series, test_indices, var_coef, gaussian
        )
        persistence_rmse = None
        model_rows = {}
        for model in MODELS:
            row = metrics(y_true, preds[model])
            row.update({
                "dim": dim,
                "subtype": subtype,
                "model": model,
                "months_valid": int(len(series["X"])),
                "train_pairs": int(len(train_pairs)),
                "test_targets": int(len(test_indices)),
                "gaussian_mean_loglik": mean_ll if model == "gaussian_rw_mean" else np.nan,
                "gaussian_total_loglik": total_ll if model == "gaussian_rw_mean" else np.nan,
            })
            model_rows[model] = row
            if model == "persistence":
                persistence_rmse = row["rmse"]

        for model in MODELS:
            row = model_rows[model]
            row["rmse_vs_persistence"] = (
                row["rmse"] / persistence_rmse
                if persistence_rmse and np.isfinite(row["rmse"]) else np.nan
            )
            results.append(row)

        series_by_subtype[subtype] = series
        increment_payload[subtype] = {
            "train": gaussian["train_increments"] if gaussian else np.empty((0, dim)),
            "test": test_increments,
        }

    return results, series_by_subtype, increment_payload


def save_figure(fig, basename):
    os.makedirs(FIGURES_DIR, exist_ok=True)
    png_path = os.path.join(FIGURES_DIR, f"{basename}.png")
    pdf_path = os.path.join(FIGURES_DIR, f"{basename}.pdf")
    fig.savefig(png_path, dpi=EXPORT_DPI, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return [png_path, pdf_path]


def plot_monthly_trajectory(series, pca_ratios, subtype):
    X = series["X"]
    years = series["years"]
    counts = series["counts"]
    fig, ax = plt.subplots(figsize=(8, 6.4))
    sc = ax.scatter(
        X[:, 0], X[:, 1],
        c=years,
        cmap="viridis",
        s=18 + 32 * np.sqrt(counts / max(counts.max(), 1)),
        alpha=0.72,
        linewidths=0,
    )
    ax.plot(X[:, 0], X[:, 1], color="#111827", linewidth=0.7, alpha=0.35)
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("Anio")
    ax.set_xlabel(f"PC1 ({pca_ratios[0] * 100:.2f}% var.)")
    ax.set_ylabel(f"PC2 ({pca_ratios[1] * 100:.2f}% var.)")
    ax.set_title(f"Trayectoria mensual de centroides en PCA d=3 - {subtype}")
    ax.grid(alpha=0.22)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    return save_figure(fig, f"pca_monthly_trajectory_{subtype.lower()}_d3")


def plot_rmse(results):
    dims = sorted(set(row["dim"] for row in results))
    fig, axes = plt.subplots(1, len(dims), figsize=(5.2 * len(dims), 5.0), sharey=False)
    if len(dims) == 1:
        axes = [axes]

    width = 0.36
    x = np.arange(len(MODELS))
    for ax, dim in zip(axes, dims):
        for offset, subtype in [(-width / 2, "H1N1"), (width / 2, "H3N2")]:
            vals = [
                next(
                    row["rmse"] for row in results
                    if row["dim"] == dim and row["subtype"] == subtype and row["model"] == model
                )
                for model in MODELS
            ]
            ax.bar(
                x + offset,
                vals,
                width=width,
                color=SUBTYPE_COLORS[subtype],
                alpha=0.82,
                label=subtype,
            )
        ax.set_title(f"d={dim}")
        ax.set_xticks(x)
        ax.set_xticklabels(["persist", "velocity", "VAR(1)", "RW mean"], rotation=25, ha="right")
        ax.set_ylabel("RMSE en PCA space")
        ax.grid(axis="y", alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend()
    fig.suptitle("Error one-step retrospectivo por modelo dinamico", y=1.04)
    plt.tight_layout()
    return save_figure(fig, "pca_dynamics_rmse_by_model")


def plot_increment_distribution(increment_payload):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.0), sharey=True)
    for ax, subtype in zip(axes, SUBTYPES):
        payload = increment_payload[subtype]
        train_norm = np.linalg.norm(payload["train"], axis=1)
        test_norm = np.linalg.norm(payload["test"], axis=1)
        max_val = max(
            np.percentile(train_norm, 99) if len(train_norm) else 1,
            np.percentile(test_norm, 99) if len(test_norm) else 1,
            1,
        )
        bins = np.linspace(0, max_val, 35)
        ax.hist(train_norm, bins=bins, density=True, alpha=0.50, label="train <=2018")
        ax.hist(test_norm, bins=bins, density=True, alpha=0.45, label="test 2019-2022")
        ax.set_title(subtype)
        ax.set_xlabel("Norma del incremento mensual en PCA d=3")
        ax.grid(alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend()
    axes[0].set_ylabel("Densidad")
    fig.suptitle("Distribucion de incrementos mensuales en PCA d=3", y=1.04)
    plt.tight_layout()
    return save_figure(fig, "pca_increment_distribution")


def format_float(value, digits=4):
    if value is None or not np.isfinite(value):
        return "NA"
    return f"{value:.{digits}f}"


def write_summary(path, cache_path, pca, monthly_series_d3, all_results,
                  train_end_year, test_start_year, test_end_year, figures):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    lines = [
        "# Dinamica mensual retrospectiva en PCA space",
        "",
        f"Fuente: `{cache_path}`.",
        "No se recalcularon embeddings, no se cargo AntigenLM y no se generaron secuencias.",
        "",
        f"Split temporal: entrenamiento hasta {train_end_year}, test {test_start_year}-{test_end_year}.",
        "La evaluacion es one-step retrospectiva sobre meses consecutivos observados.",
        "PCA fue ajustado de forma no supervisada sobre los embeddings cacheados completos; por tanto, este piloto no debe presentarse como evaluacion predictiva final sin controlar esa decision.",
        "",
        "## Meses validos y cepas por mes",
        "",
        "| subtipo | meses validos | cepas min | cepas mediana | cepas media | cepas max |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for subtype in SUBTYPES:
        series = monthly_series_d3[subtype]
        counts = series["counts"]
        lines.append(
            f"| {subtype} | {len(counts)} | {counts.min()} | "
            f"{np.median(counts):.1f} | {np.mean(counts):.2f} | {counts.max()} |"
        )

    lines.extend([
        "",
        "## Resultados por modelo",
        "",
        "| dim | subtipo | modelo | meses validos | train pairs | test targets | RMSE | MAE | distancia euclidiana media | RMSE/persistence | mean loglik RW |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in all_results:
        lines.append(
            f"| {row['dim']} | {row['subtype']} | {row['model']} | "
            f"{row['months_valid']} | {row['train_pairs']} | {row['test_targets']} | "
            f"{format_float(row['rmse'])} | {format_float(row['mae'])} | "
            f"{format_float(row['mean_euclidean'])} | "
            f"{format_float(row['rmse_vs_persistence'])} | "
            f"{format_float(row['gaussian_mean_loglik'])} |"
        )

    lines.extend([
        "",
        "## Cepas por mes",
        "",
        "| subtipo | mes | cepas |",
        "|---|---|---:|",
    ])
    for subtype in SUBTYPES:
        series = monthly_series_d3[subtype]
        for idx, count in zip(series["month_index"], series["counts"]):
            lines.append(f"| {subtype} | {month_label(idx)} | {int(count)} |")

    lines.extend([
        "",
        "## Figuras",
        "",
    ])
    for fig in figures:
        lines.append(f"- `{fig}`")

    lines.extend([
        "",
        "## Lectura metodologica",
        "",
        "- Esta es una dinamica minima en espacio PCA, no una SDE final.",
        "- Persistence es un baseline dificil cuando la trayectoria mensual es suave.",
        "- Constant velocity prueba si extrapolar el ultimo incremento ayuda o introduce sobreoscilacion.",
        "- VAR(1) evalua si existe senal lineal autoregresiva one-step mas alla de persistence.",
        "- El random walk gaussiano solo modela distribucion de incrementos; no genera secuencias.",
        "- Los resultados deben interpretarse como piloto retrospectivo local, no como prediccion prospectiva ni propuesta de variantes.",
        "",
    ])
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return path


def rolling_origin_targets(month_idx, test_start_year, test_end_year):
    start = test_start_year * 12
    end = (test_end_year + 1) * 12 - 1
    available = set(int(m) for m in month_idx)
    return [
        m for m in range(start, end + 1)
        if m in available and (m - 1) in available and (m - 2) in available
    ]


def rolling_origin_subtype(embeddings, years, months, dim, subtype,
                           test_start_year, test_end_year, ridge_alpha,
                           max_dim):
    month_idx = month_index(years, months)
    targets = rolling_origin_targets(month_idx, test_start_year, test_end_year)
    target_rows = []
    raw_errors = {model: [] for model in ROLLING_MODELS}
    truth_by_month = []
    pred_by_model = {model: [] for model in ROLLING_MODELS}
    ll_values = []
    n_skipped = 0

    for target_month in targets:
        origin_month = target_month - 1
        train_mask = month_idx <= origin_month
        eval_mask = month_idx <= target_month
        if np.sum(train_mask) <= max_dim + 2:
            n_skipped += 1
            continue

        pca = pca_from_covariance(embeddings[train_mask], max_dim=max_dim)
        projected_eval = project(embeddings[eval_mask], pca, dim)
        eval_months = month_idx[eval_mask]
        centroids, counts = centroid_dict(projected_eval, eval_months)
        needed = (target_month, origin_month, origin_month - 1)
        if any(m not in centroids for m in needed):
            n_skipped += 1
            continue

        train_pairs, train_triplets = train_transitions_from_centroids(
            centroids, origin_month
        )
        if len(train_pairs) < 2:
            n_skipped += 1
            continue

        y_true = centroids[target_month]
        x_t = centroids[origin_month]
        x_tm1 = centroids[origin_month - 1]

        pair_features = np.array([centroids[m - 1] for m in train_pairs])
        pair_targets = np.array([centroids[m] for m in train_pairs])
        var1_coef = fit_ridge_regression(pair_features, pair_targets, ridge_alpha)

        if train_triplets:
            var2_features = np.array([
                np.concatenate([centroids[m - 1], centroids[m - 2]])
                for m in train_triplets
            ])
            var2_targets = np.array([centroids[m] for m in train_triplets])
            var2_coef = fit_ridge_regression(var2_features, var2_targets, ridge_alpha)
        else:
            var2_coef = None

        increments = np.array([centroids[m] - centroids[m - 1] for m in train_pairs])
        gaussian = fit_increment_gaussian_array(increments)
        observed_increment = (y_true - x_t).reshape(1, -1)
        mean_ll, _ = gaussian_log_likelihood(observed_increment, gaussian)
        ll_values.append(mean_ll)

        preds = {
            "persistence": x_t,
            "constant_velocity": x_t + (x_t - x_tm1),
            "ridge_var1": predict_ridge(var1_coef, x_t),
            "ridge_var2": predict_ridge(var2_coef, np.concatenate([x_t, x_tm1])),
            "gaussian_rw_mean": x_t + gaussian["mean"] if gaussian else None,
        }

        truth_by_month.append((target_month, y_true))
        for model, pred in preds.items():
            if pred is None:
                pred = np.full_like(y_true, np.nan)
            pred_by_model[model].append((target_month, pred))
            raw_errors[model].append(pred - y_true)

        target_rows.append({
            "subtype": subtype,
            "dim": dim,
            "target_month": target_month,
            "target_label": month_label(target_month),
            "train_records": int(np.sum(train_mask)),
            "train_pairs": int(len(train_pairs)),
            "train_triplets": int(len(train_triplets)),
            "target_count": int(counts[target_month]),
        })

    rows = []
    persistence_rmse = None
    for model in ROLLING_MODELS:
        errors = np.asarray(raw_errors[model], dtype=np.float64)
        if len(errors) == 0 or not np.isfinite(errors).any():
            row = {
                "dim": dim,
                "subtype": subtype,
                "model": model,
                "n_eval": 0,
                "rmse": np.nan,
                "mae": np.nan,
                "mean_euclidean": np.nan,
                "rmse_vs_persistence": np.nan,
                "relative_improvement": np.nan,
                "mean_loglik": np.nan,
                "mean_nll": np.nan,
                "skipped_targets": n_skipped,
            }
        else:
            valid = np.isfinite(errors).all(axis=1)
            err = errors[valid]
            row = {
                "dim": dim,
                "subtype": subtype,
                "model": model,
                "n_eval": int(len(err)),
                "rmse": float(np.sqrt(np.mean(err ** 2))),
                "mae": float(np.mean(np.abs(err))),
                "mean_euclidean": float(np.mean(np.linalg.norm(err, axis=1))),
                "rmse_vs_persistence": np.nan,
                "relative_improvement": np.nan,
                "mean_loglik": float(np.nanmean(ll_values)) if model == "gaussian_rw_mean" else np.nan,
                "mean_nll": float(-np.nanmean(ll_values)) if model == "gaussian_rw_mean" else np.nan,
                "skipped_targets": n_skipped,
            }
        rows.append(row)
        if model == "persistence":
            persistence_rmse = row["rmse"]

    for row in rows:
        if persistence_rmse and np.isfinite(row["rmse"]):
            row["rmse_vs_persistence"] = row["rmse"] / persistence_rmse
            row["relative_improvement"] = (persistence_rmse - row["rmse"]) / persistence_rmse

    prediction_payload = {
        "truth": truth_by_month,
        "predictions": pred_by_model,
        "target_rows": target_rows,
    }
    return rows, prediction_payload


def rolling_origin_evaluation(embeddings, years, months, types, dims,
                              test_start_year, test_end_year, ridge_alpha):
    all_rows = []
    prediction_payload = {}
    max_dim = max(dims)
    for subtype in SUBTYPES:
        mask = types == subtype
        E_sub = embeddings[mask]
        y_sub = years[mask]
        m_sub = months[mask]
        for dim in dims:
            rows, payload = rolling_origin_subtype(
                E_sub, y_sub, m_sub, dim, subtype,
                test_start_year, test_end_year, ridge_alpha, max_dim,
            )
            all_rows.extend(rows)
            prediction_payload[(subtype, dim)] = payload
    return all_rows, prediction_payload


def plot_rolling_rmse(rows):
    dims = sorted(set(row["dim"] for row in rows))
    fig, axes = plt.subplots(1, len(dims), figsize=(5.4 * len(dims), 5.2), sharey=False)
    if len(dims) == 1:
        axes = [axes]
    width = 0.36
    x = np.arange(len(ROLLING_MODELS))
    for ax, dim in zip(axes, dims):
        for offset, subtype in [(-width / 2, "H1N1"), (width / 2, "H3N2")]:
            vals = [
                next(
                    row["rmse"] for row in rows
                    if row["dim"] == dim and row["subtype"] == subtype and row["model"] == model
                )
                for model in ROLLING_MODELS
            ]
            ax.bar(
                x + offset,
                vals,
                width=width,
                color=SUBTYPE_COLORS[subtype],
                alpha=0.82,
                label=subtype,
            )
        ax.set_title(f"d={dim}")
        ax.set_xticks(x)
        ax.set_xticklabels(
            ["persist", "velocity", "Ridge VAR(1)", "Ridge VAR(2)", "RW mean"],
            rotation=25,
            ha="right",
        )
        ax.set_ylabel("RMSE en PCA local")
        ax.grid(axis="y", alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend()
    fig.suptitle("Rolling-origin: RMSE por modelo dinamico", y=1.04)
    plt.tight_layout()
    return save_figure(fig, "pca_rolling_rmse_by_model")


def plot_rolling_relative_improvement(rows):
    dims = sorted(set(row["dim"] for row in rows))
    fig, axes = plt.subplots(1, len(dims), figsize=(5.4 * len(dims), 5.2), sharey=True)
    if len(dims) == 1:
        axes = [axes]
    width = 0.36
    models = [m for m in ROLLING_MODELS if m != "persistence"]
    x = np.arange(len(models))
    for ax, dim in zip(axes, dims):
        for offset, subtype in [(-width / 2, "H1N1"), (width / 2, "H3N2")]:
            vals = [
                100 * next(
                    row["relative_improvement"] for row in rows
                    if row["dim"] == dim and row["subtype"] == subtype and row["model"] == model
                )
                for model in models
            ]
            ax.bar(
                x + offset,
                vals,
                width=width,
                color=SUBTYPE_COLORS[subtype],
                alpha=0.82,
                label=subtype,
            )
        ax.axhline(0, color="black", linewidth=1.0)
        ax.set_title(f"d={dim}")
        ax.set_xticks(x)
        ax.set_xticklabels(["velocity", "Ridge VAR(1)", "Ridge VAR(2)", "RW mean"], rotation=25, ha="right")
        ax.set_ylabel("Mejora RMSE vs persistence (%)")
        ax.grid(axis="y", alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend()
    fig.suptitle("Rolling-origin: mejora relativa frente a persistence", y=1.04)
    plt.tight_layout()
    return save_figure(fig, "pca_rolling_relative_improvement")


def plot_rolling_predictions(payload, subtype, dim):
    truth = payload["truth"]
    preds = payload["predictions"]
    if not truth:
        return []

    months = np.array([m for m, _ in truth])
    labels = [month_label(m) for m in months]
    y_true = np.array([v for _, v in truth])
    x = np.arange(len(months))
    fig, axes = plt.subplots(2, 1, figsize=(11, 7.2), sharex=True)
    models_to_plot = ("persistence", "ridge_var1", "ridge_var2", "constant_velocity")
    for pc_idx, ax in enumerate(axes):
        ax.plot(x, y_true[:, pc_idx], color="black", linewidth=2.0, label="real")
        for model in models_to_plot:
            pred_values = np.array([v for _, v in preds[model]])
            ax.plot(
                x,
                pred_values[:, pc_idx],
                linewidth=1.2,
                alpha=0.85,
                label=model,
            )
        ax.set_ylabel(f"PC{pc_idx + 1}")
        ax.grid(alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(ncol=3, fontsize=8)
    step = max(1, len(labels) // 10)
    axes[-1].set_xticks(x[::step])
    axes[-1].set_xticklabels(labels[::step], rotation=35, ha="right")
    axes[-1].set_xlabel("Mes objetivo")
    fig.suptitle(f"Rolling-origin: predicciones one-step en PCA d={dim} - {subtype}", y=1.02)
    plt.tight_layout()
    return save_figure(fig, f"pca_rolling_predictions_{subtype.lower()}_d{dim}")


def best_models_by_subtype(rows):
    best = {}
    for subtype in SUBTYPES:
        subset = [row for row in rows if row["subtype"] == subtype]
        finite = [row for row in subset if np.isfinite(row["rmse"])]
        if finite:
            best[subtype] = min(finite, key=lambda row: row["rmse"])
    return best


def write_rolling_summary(path, cache_path, rows, prediction_payload,
                          test_start_year, test_end_year, ridge_alpha, figures):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    best = best_models_by_subtype(rows)
    lines = [
        "# Rolling-origin dynamics in PCA space",
        "",
        f"Fuente: `{cache_path}`.",
        "No se recalcularon embeddings, no se cargo AntigenLM y no se generaron secuencias.",
        "",
        "## Diferencias frente al piloto anterior",
        "",
        "- El piloto anterior ajustaba PCA una sola vez sobre todo el cache.",
        "- Esta evaluacion ajusta PCA nuevamente en cada corte, usando solo datos disponibles hasta el mes anterior al objetivo.",
        "- La evaluacion es one-step retrospectiva sobre meses 2019-2022 con centroides mensuales.",
        f"- Ridge alpha = {ridge_alpha}.",
        "",
        "## Resultados por modelo",
        "",
        "| dim | subtipo | modelo | n eval | RMSE | MAE | distancia euclidiana media | RMSE/persistence | mejora vs persistence | mean loglik RW | NLL media RW |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['dim']} | {row['subtype']} | {row['model']} | {row['n_eval']} | "
            f"{format_float(row['rmse'])} | {format_float(row['mae'])} | "
            f"{format_float(row['mean_euclidean'])} | {format_float(row['rmse_vs_persistence'])} | "
            f"{format_float(row['relative_improvement'])} | {format_float(row['mean_loglik'])} | "
            f"{format_float(row['mean_nll'])} |"
        )

    lines.extend([
        "",
        "## Mejor modelo por subtipo",
        "",
        "| subtipo | dim | modelo | RMSE | mejora vs persistence |",
        "|---|---:|---|---:|---:|",
    ])
    for subtype in SUBTYPES:
        row = best.get(subtype)
        if row:
            lines.append(
                f"| {subtype} | {row['dim']} | {row['model']} | "
                f"{format_float(row['rmse'])} | {format_float(row['relative_improvement'])} |"
            )

    lines.extend([
        "",
        "## Meses evaluados",
        "",
        "| subtipo | dim | targets | primer target | ultimo target | targets omitidos |",
        "|---|---:|---:|---|---|---:|",
    ])
    for subtype in SUBTYPES:
        for dim in sorted(set(row["dim"] for row in rows)):
            payload = prediction_payload[(subtype, dim)]
            labels = [month_label(m) for m, _ in payload["truth"]]
            skipped = next(
                row["skipped_targets"] for row in rows
                if row["subtype"] == subtype and row["dim"] == dim and row["model"] == "persistence"
            )
            lines.append(
                f"| {subtype} | {dim} | {len(labels)} | "
                f"{labels[0] if labels else 'NA'} | {labels[-1] if labels else 'NA'} | {skipped} |"
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
        "- Persistence sigue siendo un baseline fuerte si las mejoras relativas son pequenas o negativas.",
        "- Constant velocity puede empeorar cuando los incrementos mensuales no son persistentes o hay sobreoscilacion.",
        "- Ridge VAR(1)/VAR(2) solo aporta senal dinamica si mejora persistence de forma estable por subtipo y dimension.",
        "- La senal dinamica puede ser subtipo-dependiente; no debe promediarse sin revisar H1N1 y H3N2 por separado.",
        "- Esto sigue siendo dinamica en PCA space, no una SDE final ni una evaluacion de generacion de secuencias.",
        "",
    ])
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return path


def parse_dims(text):
    dims = tuple(int(part.strip()) for part in text.split(",") if part.strip())
    if not dims:
        raise ValueError("Debe especificarse al menos una dimension")
    if min(dims) < 1:
        raise ValueError("Las dimensiones PCA deben ser positivas")
    return dims


def main():
    parser = argparse.ArgumentParser(
        description="Dinamicas minimas retrospectivas en PCA space"
    )
    parser.add_argument(
        "--cache-path",
        default=os.path.join(RESULTS_DIR, "embeddings_cache_10k_per_subtype_seed42.pkl"),
    )
    parser.add_argument("--dims", default="3,4,5")
    parser.add_argument("--train-end-year", type=int, default=2018)
    parser.add_argument("--test-start-year", type=int, default=2019)
    parser.add_argument("--test-end-year", type=int, default=2022)
    parser.add_argument("--rolling-origin", action="store_true",
                        help="Evalua rolling-origin ajustando PCA solo con train en cada corte")
    parser.add_argument("--ridge-alpha", type=float, default=1.0,
                        help="Regularizacion L2 para Ridge VAR(1)/VAR(2)")
    args = parser.parse_args()

    dims = parse_dims(args.dims)
    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    embeddings, years, months, types, records, metadata = load_cache(args.cache_path)
    print(f"Cache cargado: embeddings={embeddings.shape} records={len(records)}")
    print(
        "Metadata: "
        f"sampling={metadata.get('sampling_strategy')} "
        f"seed={metadata.get('seed')} "
        f"max_per_subtype={metadata.get('max_per_subtype')}"
    )

    if args.rolling_origin:
        rows, rolling_payload = rolling_origin_evaluation(
            embeddings, years, months, types, dims,
            args.test_start_year, args.test_end_year, args.ridge_alpha,
        )
        print("\nResultados rolling-origin:")
        for row in rows:
            print(
                f"  d={row['dim']} {row['subtype']} {row['model']}: "
                f"n={row['n_eval']} RMSE={format_float(row['rmse'])} "
                f"MAE={format_float(row['mae'])} dist={format_float(row['mean_euclidean'])} "
                f"ratio={format_float(row['rmse_vs_persistence'])} "
                f"impr={format_float(row['relative_improvement'])} "
                f"NLL={format_float(row['mean_nll'])}"
            )

        figures = []
        figures.extend(plot_rolling_rmse(rows))
        figures.extend(plot_rolling_relative_improvement(rows))
        figures.extend(plot_rolling_predictions(rolling_payload[("H1N1", 3)], "H1N1", 3))
        figures.extend(plot_rolling_predictions(rolling_payload[("H3N2", 3)], "H3N2", 3))

        summary_path = write_rolling_summary(
            os.path.join(RESULTS_DIR, "pca_rolling_dynamics_summary.md"),
            args.cache_path,
            rows,
            rolling_payload,
            args.test_start_year,
            args.test_end_year,
            args.ridge_alpha,
            figures,
        )

        print("\nFiguras generadas:")
        for fig in figures:
            print(f"  {fig}")
        print(f"\nResumen guardado: {summary_path}")
        print("Advertencia: rolling-origin en PCA space; no es una SDE ni genera secuencias.")
        return

    pca = pca_from_covariance(embeddings, max_dim=max(dims))
    projected = project(embeddings, pca, max(dims))
    print(
        "PCA global: "
        f"PC1={pca['ratios'][0]:.4f} PC2={pca['ratios'][1]:.4f} "
        f"PC3={pca['ratios'][2]:.4f} PC4={pca['ratios'][3]:.4f}"
    )

    all_results = []
    series_by_dim = {}
    increment_by_dim = {}
    for dim in dims:
        results, series_by_subtype, increments = evaluate_dimension(
            projected, years, months, types, dim,
            args.train_end_year, args.test_start_year, args.test_end_year,
        )
        all_results.extend(results)
        series_by_dim[dim] = series_by_subtype
        increment_by_dim[dim] = increments

    print("\nResultados:")
    for row in all_results:
        print(
            f"  d={row['dim']} {row['subtype']} {row['model']}: "
            f"RMSE={format_float(row['rmse'])} MAE={format_float(row['mae'])} "
            f"dist={format_float(row['mean_euclidean'])} "
            f"ratio={format_float(row['rmse_vs_persistence'])} "
            f"ll={format_float(row['gaussian_mean_loglik'])}"
        )

    figures = []
    for subtype in SUBTYPES:
        figures.extend(plot_monthly_trajectory(series_by_dim[3][subtype], pca["ratios"], subtype))
    figures.extend(plot_rmse(all_results))
    figures.extend(plot_increment_distribution(increment_by_dim[3]))

    summary_path = write_summary(
        os.path.join(RESULTS_DIR, "pca_dynamics_summary.md"),
        args.cache_path,
        pca,
        series_by_dim[3],
        all_results,
        args.train_end_year,
        args.test_start_year,
        args.test_end_year,
        figures,
    )

    print("\nFiguras generadas:")
    for fig in figures:
        print(f"  {fig}")
    print(f"\nResumen guardado: {summary_path}")
    print("Advertencia: piloto dinamico en PCA space; no es una SDE ni genera secuencias.")


if __name__ == "__main__":
    main()
