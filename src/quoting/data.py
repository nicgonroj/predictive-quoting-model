"""Generación de un historial sintético de solicitudes de cotización.

Todo es inventado y reproducible con una semilla. Las relaciones son deliberadamente no lineales
(descuento por volumen con saturación, recargo logístico por región que depende de la familia,
interacciones entre canal y cantidad, inflación mensual) y llevan ruido, para que el problema no sea
trivial para un modelo ni para la línea base.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Familias de producto: (precio de lista base en CLP, dispersión lognormal, plazo base en días)
FAMILIAS = {
    "descartables": (4_500, 0.45, 3),
    "instrumental": (38_000, 0.50, 7),
    "monitoreo": (420_000, 0.40, 15),
    "implantes": (910_000, 0.35, 12),
    "diagnostico": (95_000, 0.55, 9),
}
PRODUCTOS_POR_FAMILIA = 8

TIPOS_CLIENTE = ["hospital_publico", "clinica_privada", "distribuidor", "centro_medico"]
P_TIPO_CLIENTE = [0.30, 0.30, 0.20, 0.20]
FACTOR_TIPO = {"hospital_publico": 0.94, "clinica_privada": 1.03, "distribuidor": 0.88, "centro_medico": 1.00}

REGIONES = ["metropolitana", "centro", "norte", "sur", "austral"]
P_REGION = [0.40, 0.20, 0.15, 0.17, 0.08]
# Recargo logístico por región; se aplica con más fuerza a familias voluminosas (monitoreo)
RECARGO_REGION = {"metropolitana": 0.00, "centro": 0.02, "norte": 0.05, "sur": 0.04, "austral": 0.09}

CANALES = ["directo", "licitacion", "portal_cliente", "ejecutivo_terreno"]
P_CANAL = [0.35, 0.25, 0.20, 0.20]

URGENCIAS = ["normal", "alta", "critica"]
P_URGENCIA = [0.65, 0.25, 0.10]
PRIMA_URGENCIA = {"normal": 0.00, "alta": 0.03, "critica": 0.08}

FECHA_INICIO = pd.Timestamp("2024-01-01")
MESES = 24


def _catalogo(rng: np.random.Generator) -> pd.DataFrame:
    """Crea el catálogo de productos con un precio de lista por producto."""
    filas = []
    for familia, (base, sigma, plazo) in FAMILIAS.items():
        for i in range(PRODUCTOS_POR_FAMILIA):
            filas.append(
                {
                    "producto": f"{familia[:3].upper()}-{i + 1:02d}",
                    "familia": familia,
                    "precio_lista": float(base * rng.lognormal(0.0, sigma)),
                    "plazo_base": plazo,
                    # popularidad: algunos productos concentran la mayoría de las solicitudes
                    "peso": float(rng.pareto(1.5) + 0.2),
                }
            )
    return pd.DataFrame(filas)


def _precio_historico(df: pd.DataFrame, ventana: int = 10) -> pd.Series:
    """Promedio de las últimas `ventana` cotizaciones previas del mismo producto (sin fuga).

    Si el producto no tiene historia, se usa el precio de lista. Es lo que el ejecutivo mira hoy
    en una planilla antes de cotizar.
    """
    previo = (
        df.groupby("producto")["precio_unitario"]
        .transform(lambda s: s.shift(1).rolling(ventana, min_periods=1).mean())
    )
    return previo.fillna(df["precio_lista"])


def generar_solicitudes(n: int = 6000, seed: int = 42) -> pd.DataFrame:
    """Genera `n` solicitudes históricas ordenadas por fecha.

    Columnas principales: producto, familia, cantidad, tipo_cliente, region, canal, urgencia,
    cliente_recurrente, plazo_dias, precio_lista (vigente), precio_historico (promedio de las últimas
    cotizaciones del producto), precio_unitario (lo que cotizó el equipo), descuento (sobre la lista)
    y ganada (1 si se adjudicó).
    """
    rng = np.random.default_rng(seed)
    cat = _catalogo(rng)

    # --- atributos de la solicitud -------------------------------------------------------------
    idx = rng.choice(len(cat), size=n, p=cat["peso"] / cat["peso"].sum())
    df = cat.iloc[idx].reset_index(drop=True).drop(columns="peso")

    dias = np.sort(rng.integers(0, MESES * 30, size=n))
    df.insert(0, "fecha", FECHA_INICIO + pd.to_timedelta(dias, unit="D"))
    df.insert(0, "id_solicitud", [f"S{i:05d}" for i in range(n)])
    mes = dias / 30.0

    df["tipo_cliente"] = rng.choice(TIPOS_CLIENTE, size=n, p=P_TIPO_CLIENTE)
    df["region"] = rng.choice(REGIONES, size=n, p=P_REGION)
    df["canal"] = rng.choice(CANALES, size=n, p=P_CANAL)
    df["urgencia"] = rng.choice(URGENCIAS, size=n, p=P_URGENCIA)
    df["cliente_recurrente"] = (rng.random(n) < 0.55).astype(int)

    # Cantidad: los descartables se piden en volumen; los implantes y equipos, de a pocos
    escala_cant = df["familia"].map(
        {"descartables": 400, "instrumental": 25, "monitoreo": 2, "implantes": 4, "diagnostico": 30}
    )
    mult_dist = np.where(df["tipo_cliente"] == "distribuidor", 3.0, 1.0)
    df["cantidad"] = np.maximum(1, np.round(escala_cant * mult_dist * rng.lognormal(0, 0.9, n))).astype(int)

    # --- precio que habría cotizado el equipo (variable objetivo a) ---------------------------
    # Descuento por volumen con saturación: crece con log(cantidad relativa) y se aplana cerca de 18 %
    cant_rel = df["cantidad"] / escala_cant
    desc_volumen = 0.18 * (1 - np.exp(-0.6 * np.log1p(cant_rel)))
    # Interacción: distribuidores con pedidos grandes negocian un descuento adicional
    desc_dist = np.where((df["tipo_cliente"] == "distribuidor") & (cant_rel > 4), 0.05, 0.0)
    # Licitaciones públicas: precio más agresivo, salvo urgencia crítica
    desc_licit = np.where((df["canal"] == "licitacion") & (df["urgencia"] != "critica"), 0.04, 0.0)
    # Recargo logístico: pesa el doble en equipos de monitoreo e implantes
    peso_log = df["familia"].map({"monitoreo": 2.0, "implantes": 1.5}).fillna(1.0)
    recargo = df["region"].map(RECARGO_REGION) * peso_log
    inflacion = (1 + 0.004) ** mes  # ~5 % anual
    estacional = 1 + 0.015 * np.sin(2 * np.pi * mes / 12)  # cierre de año fiscal

    factor = (
        df["tipo_cliente"].map(FACTOR_TIPO)
        * (1 - desc_volumen - desc_dist - desc_licit)
        * (1 + recargo)
        * (1 + df["urgencia"].map(PRIMA_URGENCIA))
        * (1 - 0.02 * df["cliente_recurrente"])
        * inflacion
        * estacional
    )
    # Ruido del criterio humano, heterocedástico: los productos de catálogo con cliente recurrente
    # se cotizan casi igual siempre; las licitaciones, los implantes y los clientes nuevos se negocian
    # caso a caso. Esa diferencia es la que el intervalo de confianza debe detectar.
    sigma = (
        0.03
        + 0.05 * (df["canal"] == "licitacion")
        + 0.04 * (df["familia"] == "implantes")
        + 0.03 * (1 - df["cliente_recurrente"])
        + 0.03 * (cant_rel > 6)
    )
    ruido = np.exp(rng.normal(0, 1, n) * sigma)
    df["precio_unitario"] = np.round(df["precio_lista"] * factor * ruido, 0)

    # --- plazo de entrega comprometido ---------------------------------------------------------
    extra_region = df["region"].map({"metropolitana": 0, "centro": 1, "norte": 3, "sur": 3, "austral": 6})
    sin_stock = rng.random(n) < 0.18
    df["plazo_dias"] = (
        df["plazo_base"] + extra_region + np.where(sin_stock, rng.integers(7, 25, n), 0) + rng.integers(0, 3, n)
    ).astype(int)

    # --- precio de lista vigente, precio histórico y descuento ----------------------------------
    # La lista se reajusta por inflación; el descuento se mide contra la lista vigente.
    df["precio_lista"] = np.round(df["precio_lista"] * inflacion, 0)
    df["precio_historico"] = np.round(_precio_historico(df), 0)
    df["descuento"] = np.round(1 - df["precio_unitario"] / df["precio_lista"], 4)

    # --- adjudicación (variable objetivo b) ----------------------------------------------------
    # La competencia castiga el precio alto (no lineal: saturación con tanh), los plazos largos cuando
    # hay urgencia, y favorece la relación con clientes recurrentes.
    urg = df["urgencia"].map({"normal": 0.0, "alta": 1.0, "critica": 2.0})
    logit = (
        -0.2
        + 2.2 * np.tanh(6 * (df["descuento"] - 0.08))
        - 0.06 * df["plazo_dias"] * (1 + 0.6 * urg)
        + 0.7 * df["cliente_recurrente"]
        - 0.5 * (df["canal"] == "licitacion")
        + 0.4 * (df["canal"] == "ejecutivo_terreno")
        + 0.3 * urg
    )
    # Probabilidad "verdadera" sin el ruido no observado. Solo existe porque los datos son sintéticos:
    # sirve para medir el techo de AUC alcanzable y nunca se usa como atributo.
    df["_prob_real"] = 1 / (1 + np.exp(-logit))
    logit = logit + rng.normal(0, 0.6, n)  # factores no observados (competidor, relación personal)
    df["ganada"] = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)

    return df.drop(columns=["plazo_base"])
