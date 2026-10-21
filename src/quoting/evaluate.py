"""Validación cruzada temporal, métricas, enrutamiento por confianza y simulación de tiempo."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, mean_absolute_error, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit

from .model import LineaBasePrecio, ModeloCotizacion


@dataclass
class Supuestos:
    """Supuestos operativos de la simulación. NO son mediciones: cámbialos por los de tu equipo."""

    solicitudes_por_dia: int = 40
    minutos_cotizacion_manual: float = 12.0  # buscar historial, calcular precio y plazo, redactar
    minutos_revision_auto: float = 2.0  # validar una sugerencia de alta confianza y enviarla
    minutos_revision_baja_confianza: float = 9.0  # revisión completa, aunque parte con la sugerencia
    umbral_ancho_relativo: float = 0.20  # intervalo P10–P90 máximo, relativo al precio, para auto-sugerir
    tope_monto_auto_clp: float = 20_000_000  # sobre este monto total siempre revisa una persona


def mape(y, pred) -> float:
    return float(np.mean(np.abs(pred - y) / y))


def ece(y, p, bins: int = 10) -> float:
    """Error de calibración esperado: brecha promedio, ponderada, entre probabilidad y frecuencia."""
    cortes = np.linspace(0, 1, bins + 1)
    grupo = np.clip(np.digitize(p, cortes[1:-1]), 0, bins - 1)
    total = 0.0
    for g in range(bins):
        m = grupo == g
        if m.any():
            total += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(total)


def _baseline_adjudicacion(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """Línea base de adjudicación: regresión logística solo con descuento y plazo."""
    cols = ["descuento", "plazo_dias"]
    lr = LogisticRegression().fit(train[cols], train["ganada"])
    return lr.predict_proba(test[cols])[:, 1]


def validar(df: pd.DataFrame, n_splits: int = 5, seed: int = 0, max_iter: int = 600) -> dict:
    """Entrena y evalúa en pliegues temporales: siempre se entrena con el pasado y se prueba con el futuro."""
    df = df.sort_values("fecha").reset_index(drop=True)
    pliegues, oof = [], []
    for k, (tr, te) in enumerate(TimeSeriesSplit(n_splits=n_splits).split(df)):
        train, test = df.iloc[tr], df.iloc[te]
        base = LineaBasePrecio().fit(train).predict(test)
        modelo = ModeloCotizacion(seed=seed, max_iter=max_iter).fit(train)
        pred = modelo.predict(test)
        p_base_adj = _baseline_adjudicacion(train, test)

        y = test["precio_unitario"].to_numpy()
        g = test["ganada"].to_numpy()
        pliegues.append(
            {
                "pliegue": k + 1,
                "n_train": len(train),
                "n_test": len(test),
                "desde": str(test["fecha"].min().date()),
                "hasta": str(test["fecha"].max().date()),
                "mae_base": mean_absolute_error(y, base),
                "mae_modelo": mean_absolute_error(y, pred["precio_sugerido"]),
                "mape_base": mape(y, base),
                "mape_historico": mape(y, test["precio_historico"].to_numpy()),
                "mape_modelo": mape(y, pred["precio_sugerido"].to_numpy()),
                "cobertura_p10_p90": float(np.mean((y >= pred["p10"]) & (y <= pred["p90"]))),
                "auc_base": roc_auc_score(g, p_base_adj),
                "auc_modelo": roc_auc_score(g, pred["prob_adjudicacion"]),
                # Techo: AUC de la probabilidad real del generador (sin ruido no observado)
                "auc_techo": roc_auc_score(g, test["_prob_real"]),
                "brier_base": brier_score_loss(g, p_base_adj),
                "brier_modelo": brier_score_loss(g, pred["prob_adjudicacion"]),
                "ece_base": ece(g, p_base_adj),
                "ece_modelo": ece(g, pred["prob_adjudicacion"].to_numpy()),
            }
        )
        oof.append(pd.concat([test.reset_index(drop=True), pred.reset_index(drop=True)], axis=1).assign(pliegue=k + 1))
    return {"pliegues": pd.DataFrame(pliegues), "oof": pd.concat(oof, ignore_index=True)}


def enrutar(oof: pd.DataFrame, s: Supuestos) -> pd.Series:
    """True si la cotización queda auto-sugerida (alta confianza); False si va a revisión."""
    monto = oof["precio_sugerido"] * oof["cantidad"]
    return (oof["ancho_relativo"] <= s.umbral_ancho_relativo) & (monto <= s.tope_monto_auto_clp)


def simular_tiempo(oof: pd.DataFrame, s: Supuestos) -> dict:
    """Estima el ahorro diario de tiempo con los supuestos dados y la tasa de auto-sugerencia medida."""
    auto = enrutar(oof, s)
    tasa_auto = float(auto.mean())
    error = np.abs(oof["precio_sugerido"] - oof["precio_unitario"]) / oof["precio_unitario"]
    antes = s.solicitudes_por_dia * s.minutos_cotizacion_manual
    despues = s.solicitudes_por_dia * (
        tasa_auto * s.minutos_revision_auto + (1 - tasa_auto) * s.minutos_revision_baja_confianza
    )
    return {
        "supuestos": asdict(s),
        "tasa_auto_sugerida": tasa_auto,
        "tasa_revision": 1 - tasa_auto,
        "mape_auto_sugeridas": float(error[auto].mean()) if auto.any() else None,
        "mape_revision": float(error[~auto].mean()) if (~auto).any() else None,
        "auto_con_error_mayor_10pct": float((error[auto] > 0.10).mean()) if auto.any() else None,
        "minutos_dia_antes": antes,
        "minutos_dia_despues": despues,
        "reduccion_tiempo": 1 - despues / antes,
    }


def tabla_umbrales(oof: pd.DataFrame, s: Supuestos, umbrales=(0.10, 0.15, 0.20, 0.25, 0.30, 0.40)) -> pd.DataFrame:
    """Compromiso entre automatizar más y equivocarse más, para elegir el umbral con datos."""
    filas = []
    for u in umbrales:
        r = simular_tiempo(oof, Supuestos(**{**asdict(s), "umbral_ancho_relativo": u}))
        filas.append(
            {
                "umbral": u,
                "auto_sugeridas": r["tasa_auto_sugerida"],
                "mape_auto": r["mape_auto_sugeridas"],
                "mape_revision": r["mape_revision"],
                "auto_error_>10%": r["auto_con_error_mayor_10pct"],
                "reduccion_tiempo": r["reduccion_tiempo"],
            }
        )
    return pd.DataFrame(filas)


def curva_calibracion(oof: pd.DataFrame, bins: int = 10):
    return calibration_curve(oof["ganada"], oof["prob_adjudicacion"], n_bins=bins, strategy="quantile")
