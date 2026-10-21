import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from quoting.__main__ import main  # noqa: E402
from quoting.data import generar_solicitudes  # noqa: E402
from quoting.evaluate import Supuestos, enrutar, simular_tiempo, validar  # noqa: E402
from quoting.model import CATEGORICAS, NUMERICAS_ADJ, LineaBasePrecio  # noqa: E402


class DatosTests(unittest.TestCase):
    def test_reproducible_con_semilla(self):
        a = generar_solicitudes(500, seed=7)
        b = generar_solicitudes(500, seed=7)
        pd.testing.assert_frame_equal(a, b)
        c = generar_solicitudes(500, seed=8)
        self.assertFalse(a["precio_unitario"].equals(c["precio_unitario"]))

    def test_columnas_y_rangos(self):
        df = generar_solicitudes(800, seed=1)
        for col in ["producto", "familia", "cantidad", "tipo_cliente", "region", "canal", "urgencia",
                    "precio_historico", "descuento", "ganada"]:
            self.assertIn(col, df.columns)
        self.assertTrue((df["cantidad"] >= 1).all())
        self.assertTrue((df["precio_unitario"] > 0).all())
        self.assertTrue(set(df["ganada"].unique()) <= {0, 1})
        self.assertTrue(df["fecha"].is_monotonic_increasing)
        self.assertGreater(df["ganada"].mean(), 0.1)
        self.assertLess(df["ganada"].mean(), 0.9)

    def test_la_probabilidad_real_no_es_atributo(self):
        # La columna de oráculo solo sirve para medir el techo; ningún modelo puede verla
        self.assertNotIn("_prob_real", CATEGORICAS + NUMERICAS_ADJ)


class ModeloTests(unittest.TestCase):
    def test_linea_base_usa_familia_si_no_hay_historia(self):
        df = generar_solicitudes(300, seed=3)
        base = LineaBasePrecio().fit(df)
        nuevo = df.head(1).assign(producto="NUEVO-99")
        self.assertAlmostEqual(base.predict(nuevo)[0], base.por_familia[nuevo["familia"].iloc[0]])

    def test_pipeline_corre_en_pequeno(self):
        df = generar_solicitudes(900, seed=5)
        r = validar(df, n_splits=3, seed=0, max_iter=40)
        self.assertEqual(len(r["pliegues"]), 3)
        oof = r["oof"]
        self.assertTrue(np.isfinite(oof["precio_sugerido"]).all())
        self.assertTrue(((oof["prob_adjudicacion"] >= 0) & (oof["prob_adjudicacion"] <= 1)).all())
        self.assertTrue((oof["p10"] <= oof["p90"]).all())
        # entrena con el pasado y prueba con el futuro
        for _, f in r["pliegues"].iterrows():
            self.assertGreater(f["n_train"], 0)

    def test_simulacion_respeta_supuestos(self):
        df = generar_solicitudes(900, seed=5)
        oof = validar(df, n_splits=3, max_iter=40)["oof"]
        todo_auto = simular_tiempo(oof, Supuestos(umbral_ancho_relativo=99, tope_monto_auto_clp=1e18))
        self.assertAlmostEqual(todo_auto["tasa_auto_sugerida"], 1.0)
        # si todo se auto-sugiere, el tiempo baja a minutos_revision_auto / minutos_manual
        self.assertAlmostEqual(todo_auto["reduccion_tiempo"], 1 - 2.0 / 12.0)
        nada = enrutar(oof, Supuestos(umbral_ancho_relativo=0.0))
        self.assertFalse(nada.any())

    def test_cli_escribe_metricas(self):
        with tempfile.TemporaryDirectory() as tmp:
            r = main(["--n", "700", "--splits", "2", "--max-iter", "30", "--out", tmp, "--sin-figura"])
            self.assertTrue((Path(tmp) / "metricas.json").exists())
            self.assertIn("auc_modelo", r["adjudicacion"])


if __name__ == "__main__":
    unittest.main()
