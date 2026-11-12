"""Per-metre shadow-victim identity (shadow_check.identita_metri): same set of stamps and poses
within 1e-6 m in every metre. Real data from a recorded run (tests/dati/ombra_s1001,
excerpt) and synthetic data; no simulation.
"""
import csv
import importlib.util
import json
import shutil
from pathlib import Path

from src.optimization import shadow_check as shc
from src.optimization import window_record as wrec

ROOT = Path(__file__).resolve().parent.parent
DATI = ROOT / "tests" / "dati" / "ombra_s1001"
spec = importlib.util.spec_from_file_location("coda", ROOT / "scripts" / "coda_serieB_stato.py")
coda = importlib.util.module_from_spec(spec)
spec.loader.exec_module(coda)

# synthetic metres: 5 windows, openings at 11.0, 12.0, ... (metre k = (open_k, open_k+1])
HIST = [{"window": k, "victim_window": {"open_stamp": 10.9 + k, "drive_stamps": [11.5 + k, 11.9 + k]}}
        for k in range(5)]


def _righe(t0=10.0, t1=15.9, **mod):
    """Rows like shadow_check_scans.csv, all identical; mod: stamp -> differing fields."""
    out = []
    n = int(round((t1 - t0) * 10)) + 1
    for i in range(n):
        t = round(t0 + 0.1 * i, 1)
        r = {"stamp_s": t, "elaborata": "entrambe", "posa": "entrambe", "dxy_m": "0", "dyaw_rad": "0"}
        r.update(mod.get(f"{t:.1f}", {}))
        out.append(r)
    return out


def _esito(rows, hist=HIST):
    metri = shc.identita_metri(rows, wrec.confini_metri(hist))
    return metri, shc.esito_identita(metri)


def test_confini_metri_dalla_history():
    c = wrec.confini_metri(HIST)
    assert [(x["finestra"], x["da_s"], x["a_s"]) for x in c] == [
        (0, None, 11.9), (1, 11.9, 12.9), (2, 12.9, 13.9), (3, 13.9, 14.9), (4, 14.9, None)]


def test_prima_run_serieB_divergente_nei_metri_giusti():
    r = shc.identita_da_rep(DATI)
    assert r["esito"] == "ombra_divergente"
    m = {x["finestra"]: x for x in r["metri"]}
    # stamps of one side only: metres w0 and w3 (6 victim only, 14 shadow only in total)
    assert [(k, m[k]["n_solo_vittima"], m[k]["n_solo_ombra"]) for k in range(5)] == [
        (0, 2, 4), (1, 0, 0), (2, 0, 0), (3, 4, 10), (4, 0, 0)]
    assert m[0]["solo_vittima_stamp_s"] == [13.2, 15.9]
    assert m[0]["solo_ombra_stamp_s"] == [15.5, 15.7, 16.0, 17.9]
    assert m[0]["prima_stamp_divergente_s"] == 13.2 and m[0]["motivo_prima"] == "solo vittima"
    # poses apart from 13.3 on (3.1e-6 m): all metres diverge
    assert r["metri_divergenti"] == [0, 1, 2, 3, 4]
    assert all(m[k]["motivo_prima"] == "posa diversa" for k in (1, 2, 3, 4))
    assert 0.012 < m[2]["max_dxy_m"] < 0.0124
    # shadow trace tail lost at close (38.5-39.2, pose of both) does not decide
    assert m[4]["n_posa_mancante"] == 0 and m[4]["coda_non_confrontabile"] == 0
    assert r["testo"].startswith("OMBRA DIVERGENTE nei metri w0, w1, w2, w3, w4")


def test_sintetico_identico():
    metri, e = _esito(_righe())
    assert e["esito"] == "identica" and e["testo"] == "ombra: identica in 5/5 metri"
    assert all(m["n_comuni"] == 10 for m in metri[1:4])


def test_stamp_mancante_a_una_sola():
    metri, e = _esito(_righe(**{"13.4": {"elaborata": "solo_ombra", "posa": "solo_ombra", "dxy_m": "", "dyaw_rad": ""}}))
    assert e["esito"] == "ombra_divergente" and e["metri_divergenti"] == [2]
    assert metri[2]["n_solo_ombra"] == 1 and metri[2]["prima_stamp_divergente_s"] == 13.4
    assert e["testo"].startswith("OMBRA DIVERGENTE nei metri w2 (1 su 5")


def test_stamp_assente_del_tutto_non_conta():
    # neither processed it: sets still equal
    _, e = _esito(_righe(**{"12.5": {"elaborata": "", "posa": ""}}))
    assert e["esito"] == "identica"


def test_una_sola_stamp_diversa_e_soglia_1e6():
    _, e = _esito(_righe(**{"14.2": {"dxy_m": "2e-6"}}))
    assert e["esito"] == "ombra_divergente" and e["metri_divergenti"] == [3]
    _, e = _esito(_righe(**{"14.2": {"dxy_m": "1e-6"}}))
    assert e["esito"] == "identica"                  # within the threshold (<=)


def test_soglia_angolo():
    _, e = _esito(_righe(**{"11.0": {"dyaw_rad": "2e-9"}}))
    assert e["esito"] == "ombra_divergente" and e["metri_divergenti"] == [0]


def test_posa_mancante_su_stamp_comune():
    metri, e = _esito(_righe(**{"12.0": {"posa": "solo_vittima", "dxy_m": "", "dyaw_rad": "",
                                         "csv_identico": "False"}}))
    assert e["metri_divergenti"] == [1] and metri[1]["n_posa_mancante"] == 1
    assert metri[1]["motivo_prima"] == "posa non confrontabile"
    # pose not logged by the lab, identical trace rows: confirmed
    metri, e = _esito(_righe(**{"12.0": {"posa": "", "dxy_m": "", "dyaw_rad": "", "csv_identico": "True"}}))
    assert e["esito"] == "identica" and metri[1]["n_solo_traccia"] == 1


def test_coda_di_fine_run_non_confrontabile():
    # after the end of the driven segment of the last metre (15.9): closing, does not decide
    mod = {f"{t / 10:.1f}": {"elaborata": "solo_vittima", "posa": "solo_vittima"} for t in range(160, 163)}
    metri, e = _esito(_righe(t1=16.2, **mod))
    assert e["esito"] == "identica" and metri[4]["coda_non_confrontabile"] == 3
    # stamps of both after driving ends: compared like the others
    _, e = _esito(_righe(t1=16.2, **{"16.1": {"dxy_m": "3e-6"}}))
    assert e["esito"] == "ombra_divergente" and e["metri_divergenti"] == [4]
    # before driving ends the same tail is "one side only": divergent
    mod = {f"{t / 10:.1f}": {"elaborata": "solo_vittima", "posa": "solo_vittima"} for t in range(157, 160)}
    _, e = _esito(_righe(**mod))
    assert e["esito"] == "ombra_divergente" and e["metri_divergenti"] == [4]


def test_mola_che_smette_a_meta_run():
    # the shadow stops after 12.0, the victim goes on (and vice versa after 13.0)
    mod = {f"{t / 10:.1f}": {"elaborata": "solo_vittima", "posa": "solo_vittima"} for t in range(121, 160)}
    metri, e = _esito(_righe(**mod))
    assert e["esito"] == "ombra_divergente" and e["metri_divergenti"] == [1, 2, 3, 4]
    assert metri[1]["prima_stamp_divergente_s"] == 12.1 and metri[0]["esito"] == "identica"
    mod = {f"{t / 10:.1f}": {"elaborata": "solo_ombra", "posa": "solo_ombra"} for t in range(131, 160)}
    _, e = _esito(_righe(**mod))
    assert e["metri_divergenti"] == [2, 3, 4]
    # without driving end in the history, no closing exemption
    h = [dict(x, victim_window={"open_stamp": x["victim_window"]["open_stamp"]}) for x in HIST]
    mod = {f"{t / 10:.1f}": {"elaborata": "solo_vittima", "posa": "solo_vittima"} for t in range(160, 163)}
    _, e = _esito(_righe(t1=16.2, **mod), h)
    assert e["esito"] == "ombra_divergente"


def test_confine_assente_non_misurato():
    h = [dict(x, victim_window=dict(x["victim_window"])) for x in HIST]
    h[2]["victim_window"]["open_stamp"] = None
    metri, e = _esito(_righe(), h)
    assert [m["esito"] for m in metri] == ["identica", "non_misurata", "non_misurata", "identica", "identica"]
    assert e["esito"] == "non_misurata" and e["testo"].startswith("OMBRA NON MISURATA nei metri w1, w2")


def test_rep_senza_confronto(tmp_path):
    r = shc.identita_da_rep(tmp_path)
    assert r["esito"] == "non_misurata" and "nessun confronto" in r["testo"]
    assert r["testo"].startswith("OMBRA NON MISURATA") and r["testo"].endswith("run non valida")
    shutil.copyfile(DATI / "shadow_check_scans.csv", tmp_path / "shadow_check_scans.csv")
    (tmp_path / "history.json").write_text("[{tronca")
    r = shc.identita_da_rep(tmp_path)
    assert r["esito"] == "non_misurata" and "history.json illeggibile" in r["testo"]


def test_riga_di_comando(tmp_path, capsys):
    assert shc.main([str(DATI), "--json", str(tmp_path / "o.json")]) == 3
    assert json.loads((tmp_path / "o.json").read_text())["esito"] == "ombra_divergente"
    rep = tmp_path / "rep_01"
    rep.mkdir()
    with open(rep / "shadow_check_scans.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(_righe()[0]))
        w.writeheader()
        w.writerows(_righe())
    (rep / "history.json").write_text(json.dumps(HIST))
    assert shc.main([str(rep)]) == 0
    assert "ombra: identica in 5/5 metri" in capsys.readouterr().out


def test_registro_dal_vivo_scrive_identita(tmp_path):
    sc = shc.ShadowCheck(tmp_path, None, None)
    sc.close()
    sc.final = {"errore": "prova"}
    sc.final_rows = _righe(**{"13.0": {"dxy_m": "5e-6"}})
    hist = [dict(x) for x in HIST]
    sc.fill_history(hist)
    j = json.loads((tmp_path / "shadow_check.json").read_text())
    assert j["identita_per_metro"]["esito"] == "ombra_divergente"
    assert j["identita_per_metro"]["metri"]["2"]["prima_stamp_divergente_s"] == 13.0
    assert hist[-1]["shadow_check_fine"]["identita_per_metro"]["metri_divergenti"] == [2]


def test_coda_ombra_divergente_come_guasto_catena():
    base = {"rc": 0, "completata": True, "adattatore_vivo": True, "storia": {}, "ritardo_rc": 0}
    div = coda.ombra_run(str(DATI))
    e = coda.classifica({**base, "ombra": div})
    assert e["esito"] == "guasto_catena" and "OMBRA DIVERGENTE" in e["motivo"]
    assert e["esito"] in coda.ESITI_RIPETIBILI
    assert coda.classifica({**base, "ombra": {"esito": "identica"}})["ombra"] == "identica"
    # comparison not done (missing field, empty output, not measured, error): run not valid
    for om in (None, {}, {"esito": "non_misurata", "testo": "OMBRA NON MISURATA (x): run non valida"},
               coda.ombra_run("/percorso/che/non/esiste")):
        e = coda.classifica({**base, "ombra": om})
        assert e["esito"] == "guasto_catena" and e["motivo"].startswith("ombra non misurata")
    assert coda.classifica({**base, "ombra": div, "baseline": True})["esito"] == "fatta"
    assert coda.classifica({**base, "baseline": True})["esito"] == "fatta"
    # the [4] that stops the queue comes first
    assert coda.classifica({**base, "ritardo_rc": 4, "ombra": div})["esito"] == "ritardo_configurazione"
    # repetitions: same limit as chain failures
    v = {"tentativi": [{"esito": "guasto_catena"}] * (coda.RIPETIZIONI_MAX + 1)}
    assert coda.stato_run(v) == "guasto_definitivo"


def test_riepilogo_run_con_riga_ombra(tmp_path):
    run = tmp_path / "s1001"
    rep = run / "campagna" / "straight" / "x" / "rep_01"
    rep.mkdir(parents=True)
    for f in ("shadow_check_scans.csv", "history.json"):
        shutil.copyfile(DATI / f, rep / f)
    d = coda.riepilogo_run(str(run), str(rep), usa_bag=False)
    t = coda.testo_riepilogo_run(d)
    assert "  OMBRA DIVERGENTE nei metri w0, w1, w2, w3, w4 (5 su 5; identica in 0)" in t
    assert "    w3 (28.7, 34.1]: solo vittima 4, solo ombra 10" in t


def test_randpuro_eps5_un_metro_divergente():
    # "1 episode: to declare" without the rule; with it the metre is not valid
    r = shc.identita_da_rep(ROOT / "tests" / "dati" / "ombra_randpuro_eps5_s1001")
    assert r["esito"] == "ombra_divergente" and r["metri_divergenti"] == [6] and r["n_metri"] == 1
    m = r["metri"][0]
    assert (m["n_solo_vittima"], m["n_solo_ombra"]) == (4, 2)
    assert abs(m["max_dxy_m"] - 2.44e-4) < 1e-6


def test_partenza_sfasata_divergente_dal_primo_metro():
    # recorded pilot run (s4001): victim-only stamps at start (class "avvio", not counted
    # as an episode) and poses apart by more than 0.1 m from the start
    r = shc.identita_da_rep(ROOT / "tests" / "dati" / "ombra_pilotaT5_s4001")
    assert r["esito"] == "ombra_divergente" and r["metri_divergenti"] == [0, 1, 2, 3, 4, 5]
    m0 = r["metri"][0]
    assert m0["n_solo_vittima"] == 9 and m0["motivo_prima"] == "solo vittima" and m0["max_dxy_m"] > 0.1
