"""Modelos: línea base de precio, precio sugerido con intervalo y probabilidad de adjudicación."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

CATEGORICAS = ["producto", "familia", "tipo_cliente", "region", "canal", "urgencia"]
NUMERICAS_PRECIO = ["log_cantidad", "log_precio_lista", "log_precio_historico", "cliente_recurrente", "mes", "mes_del_anio"]
# El modelo de adjudicación además ve el precio propuesto (como descuento) y el plazo
NUMERICAS_ADJ = NUMERICAS_PRECIO + ["descuento", "plazo_dias"]


def preparar(df: pd.DataFrame, categorias: dict[str, list[str]] | None = None) -> tuple[pd.DataFrame, dict]:
    """Convierte la tabla cruda en atributos numéricos para los modelos.

    Las categóricas se codifican como enteros con el vocabulario visto en entrenamiento; una
    categoría nueva en prueba queda como NaN, que HistGradientBoosting trata como faltante.
    """
    x = pd.DataFrame(index=df.index)
    if categorias is None:
        categorias = {c: sorted(df[c].unique()) for c in CATEGORICAS}
    for c in CATEGORICAS:
        mapa = {v: i for i, v in enumerate(categorias[c])}
        x[c] = df[c].map(mapa).astype(float)
    x["log_cantidad"] = np.log1p(df["cantidad"])
    x["log_precio_lista"] = np.log(df["precio_lista"])
    x["log_precio_historico"] = np.log(df["precio_historico"])
    x["cliente_recurrente"] = df["cliente_recurrente"]
    fecha = pd.to_datetime(df["fecha"])
    x["mes"] = (fecha - pd.Timestamp("2024-01-01")).dt.days / 30.0
    x["mes_del_anio"] = fecha.dt.month
    x["descuento"] = df["descuento"]
    x["plazo_dias"] = df["plazo_dias"]
    return x, categorias


class LineaBasePrecio:
    """Lo que haría una planilla: precio promedio por producto (y por familia si no hay historia)."""

    def fit(self, df: pd.DataFrame) -> "LineaBasePrecio":
        self.por_producto = df.groupby("producto")["precio_unitario"].mean()
        self.por_familia = df.groupby("familia")["precio_unitario"].mean()
        self.global_ = df["precio_unitario"].mean()
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        pred = df["producto"].map(self.por_producto)
        pred = pred.fillna(df["familia"].map(self.por_familia)).fillna(self.global_)
        return pred.to_numpy()


class ModeloCotizacion:
    """Precio sugerido (mediana), intervalo P10–P90 y probabilidad de adjudicación.

    El precio se modela en escala logarítmica: los productos van de miles a cientos de miles de
    pesos y el error relevante para el negocio es porcentual.
    """

    def __init__(self, seed: int = 0, max_iter: int = 600):
        self.seed = seed
        self.max_iter = max_iter

    def _hgb_reg(self, **kw):
        cat_mask = [c in CATEGORICAS for c in CATEGORICAS + NUMERICAS_PRECIO]
        return HistGradientBoostingRegressor(
            categorical_features=cat_mask, max_iter=self.max_iter, learning_rate=0.05,
            # árboles poco profundos y hojas grandes: la estructura del precio es casi multiplicativa
            # y con pocos miles de filas los árboles profundos memorizan el ruido de negociación
            max_leaf_nodes=7, min_samples_leaf=40, l2_regularization=1.0, random_state=self.seed, **kw,
        )

    def fit(self, df: pd.DataFrame) -> "ModeloCotizacion":
        df = df.sort_values("fecha")
        x, self.categorias = preparar(df)
        y = np.log(df["precio_unitario"].to_numpy())
        xp = x[CATEGORICAS + NUMERICAS_PRECIO]
        # Error cuadrático en log: el precio sugerido es la media geométrica esperada
        self.mediana = self._hgb_reg(loss="squared_error").fit(xp, y)

        # Intervalo P10–P90 con calibración conformal (CQR): los cuantiles se ajustan con el 80 % más
        # antiguo y el 20 % más reciente mide cuánto hay que ensanchar el intervalo para que cubra
        # de verdad el 80 % de los casos. Sin esto, los cuantiles de boosting quedan demasiado estrechos.
        corte = int(len(df) * 0.8)
        self.p10 = self._hgb_reg(loss="quantile", quantile=0.10).fit(xp.iloc[:corte], y[:corte])
        self.p90 = self._hgb_reg(loss="quantile", quantile=0.90).fit(xp.iloc[:corte], y[:corte])
        lo, hi = self._cuantiles(xp.iloc[corte:])
        puntaje = np.maximum(lo - y[corte:], y[corte:] - hi)
        n_cal = len(puntaje)
        nivel = min(1.0, np.ceil((n_cal + 1) * 0.80) / n_cal)
        self.ajuste_conformal = float(np.quantile(puntaje, nivel))

        # Adjudicación: modelo más simple (pocas hojas, hojas grandes) porque la señal es ruidosa;
        # sin el código de producto para no memorizar productos.
        cols = [c for c in CATEGORICAS + NUMERICAS_ADJ if c != "producto"]
        self.cols_adj = cols
        self.adjudicacion = HistGradientBoostingClassifier(
            categorical_features=[c in CATEGORICAS for c in cols], max_iter=150, learning_rate=0.05,
            max_leaf_nodes=8, min_samples_leaf=40, l2_regularization=2.0, random_state=self.seed,
        ).fit(x[cols], df["ganada"])
        return self

    def _cuantiles(self, xp: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        a, b = self.p10.predict(xp), self.p90.predict(xp)
        return np.minimum(a, b), np.maximum(a, b)  # los cuantiles pueden cruzarse

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        x, _ = preparar(df, self.categorias)
        xp = x[CATEGORICAS + NUMERICAS_PRECIO]
        mediana = np.exp(self.mediana.predict(xp))
        lo, hi = self._cuantiles(xp)
        lo, hi = np.exp(lo - self.ajuste_conformal), np.exp(hi + self.ajuste_conformal)
        return pd.DataFrame(
            {
                "precio_sugerido": mediana,
                "p10": lo,
                "p90": hi,
                # Ancho relativo del intervalo: la medida de confianza del precio
                "ancho_relativo": (hi - lo) / mediana,
                "prob_adjudicacion": self.adjudicacion.predict_proba(x[self.cols_adj])[:, 1],
            },
            index=df.index,
        )
