"""Pipeline completo: `PYTHONPATH=src python -m quoting`.

Genera el dataset sintético, valida en pliegues temporales, simula el ahorro de tiempo y deja
métricas en output/ y la figura en docs/.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data import generar_solicitudes
from .evaluate import Supuestos, curva_calibracion, simular_tiempo, tabla_umbrales, validar

RAIZ = Path(__file__).resolve().parents[2]


def figura(pliegues, oof, ruta: Path) -> None:
    """Figura pequeña: error de precio por pliegue y calibración de la probabilidad."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4), dpi=110)
    x = pliegues["pliegue"]
    ax[0].plot(x, pliegues["mape_base"] * 100, "o-", color="#9aa0a6", label="Promedio por producto")
    ax[0].plot(x, pliegues["mape_historico"] * 100, "s--", color="#c58f3d", label="Últimas 10 cotizaciones")
    ax[0].plot(x, pliegues["mape_modelo"] * 100, "o-", color="#1a73e8", label="Gradient boosting")
    ax[0].set(title="Error de precio (MAPE) por pliegue temporal", xlabel="Pliegue", ylabel="MAPE %")
    ax[0].set_xticks(x)
    ax[0].legend(fontsize=7, frameon=False)
    ax[0].grid(alpha=0.3)

    frac, media = curva_calibracion(oof)
    ax[1].plot([0, 1], [0, 1], ":", color="#9aa0a6", label="Calibración perfecta")
    ax[1].plot(media, frac, "o-", color="#1a73e8", label="Modelo")
    ax[1].set(title="Calibración de la probabilidad de adjudicación",
              xlabel="Probabilidad predicha", ylabel="Frecuencia observada")
    ax[1].legend(fontsize=7, frameon=False)
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    ruta.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ruta)
    plt.close(fig)


def main(argv=None) -> dict:
    p = argparse.ArgumentParser(description="Modelo predictivo de cotizaciones (datos sintéticos)")
    p.add_argument("--n", type=int, default=6000, help="solicitudes sintéticas")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--splits", type=int, default=5, help="pliegues de validación temporal")
    p.add_argument("--umbral", type=float, default=Supuestos.umbral_ancho_relativo)
    p.add_argument("--max-iter", type=int, default=600)
    p.add_argument("--out", type=Path, default=RAIZ / "output")
    p.add_argument("--figura", type=Path, default=RAIZ / "docs" / "resultados.png")
    p.add_argument("--sin-figura", action="store_true")
    a = p.parse_args(argv)

    df = generar_solicitudes(a.n, a.seed)
    a.out.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out / "solicitudes_sinteticas.csv", index=False)

    r = validar(df, n_splits=a.splits, seed=a.seed, max_iter=a.max_iter)
    pl, oof = r["pliegues"], r["oof"]
    s = Supuestos(umbral_ancho_relativo=a.umbral)
    sim = simular_tiempo(oof, s)
    umbrales = tabla_umbrales(oof, s)

    resumen = {
        "n_solicitudes": len(df),
        "tasa_adjudicacion": float(df["ganada"].mean()),
        "precio": {
            "mae_base": float(pl["mae_base"].mean()),
            "mae_modelo": float(pl["mae_modelo"].mean()),
            "mape_base": float(pl["mape_base"].mean()),
            "mape_historico": float(pl["mape_historico"].mean()),
            "mape_modelo": float(pl["mape_modelo"].mean()),
            "mape_modelo_std": float(pl["mape_modelo"].std()),
            "cobertura_p10_p90": float(pl["cobertura_p10_p90"].mean()),
        },
        "adjudicacion": {
            "auc_base": float(pl["auc_base"].mean()),
            "auc_modelo": float(pl["auc_modelo"].mean()),
            "auc_modelo_std": float(pl["auc_modelo"].std()),
            "auc_techo": float(pl["auc_techo"].mean()),
            "brier_base": float(pl["brier_base"].mean()),
            "brier_modelo": float(pl["brier_modelo"].mean()),
            "ece_base": float(pl["ece_base"].mean()),
            "ece_modelo": float(pl["ece_modelo"].mean()),
        },
        "simulacion": sim,
        "umbrales": umbrales.to_dict(orient="records"),
    }
    pl.to_csv(a.out / "metricas_por_pliegue.csv", index=False)
    oof.to_csv(a.out / "predicciones_fuera_de_muestra.csv", index=False)
    (a.out / "metricas.json").write_text(json.dumps(resumen, indent=2, ensure_ascii=False))
    if not a.sin_figura:
        figura(pl, oof, a.figura)

    fmt = lambda v: "n/d" if v is None else f"{v:.1%}"  # noqa: E731
    print(f"Solicitudes: {len(df)} | adjudicación histórica: {fmt(resumen['tasa_adjudicacion'])}")
    print(pl.round(4).to_string(index=False))
    pr, ad = resumen["precio"], resumen["adjudicacion"]
    print(f"\nPrecio  MAPE base {fmt(pr['mape_base'])} | histórico {fmt(pr['mape_historico'])} | "
          f"modelo {fmt(pr['mape_modelo'])} ± {fmt(pr['mape_modelo_std'])} | cobertura P10–P90 {fmt(pr['cobertura_p10_p90'])}")
    print(f"        MAE base {pr['mae_base']:,.0f} CLP | modelo {pr['mae_modelo']:,.0f} CLP")
    print(f"Adjud.  AUC base {ad['auc_base']:.3f} | modelo {ad['auc_modelo']:.3f} | techo {ad['auc_techo']:.3f}\n"
          f"        "
          f"Brier base {ad['brier_base']:.3f} / modelo {ad['brier_modelo']:.3f} | "
          f"ECE base {ad['ece_base']:.3f} / modelo {ad['ece_modelo']:.3f}")
    print(f"\nEnrutamiento (umbral {s.umbral_ancho_relativo:.0%}): auto-sugeridas {fmt(sim['tasa_auto_sugerida'])} | "
          f"MAPE auto {fmt(sim['mape_auto_sugeridas'])} | MAPE revisión {fmt(sim['mape_revision'])}")
    print(f"Tiempo simulado: {sim['minutos_dia_antes']:.0f} → {sim['minutos_dia_despues']:.0f} min/día "
          f"({fmt(sim['reduccion_tiempo'])} menos, con los supuestos de Supuestos)")
    print("\n" + umbrales.round(3).to_string(index=False))
    return resumen


if __name__ == "__main__":
    main()
