"""Run delay check (scripts/controllo_ritardo_run.py): synthetic cases for the code and,
read only, the 5 seeds of a pilot run against the reference table."""
import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "controllo_ritardo_run.py"
sys.path.insert(0, str(ROOT / "scripts"))
import controllo_ritardo_run as cr  # noqa: E402

COL = ("t_arr_wall", "t_arr_mono", "stamp_in", "victim_frame_idx", "victim_stream", "stamp_victim", "proc_ms",
       "t_release_target_mono", "t_release_mono", "d_min_ms", "delta_victim_ms", "delta_shadow_ms")
VIT = "/carter/lidar_victim"
PER_FINESTRA = 50


def run_finta(d, delta, *, shadow_topic=None, delta_ombra=None, omesse=(), proc=None, dmin=140.0,
              senza_ombra=False, colonne=COL, modifica=None, nuvole_per_genoma=None):
    """Folder of one seed: frame_log.csv with one victim row per delta value (plus one
    victim_stream row without stamp_victim), MOLA log and w<k>_applied traces."""
    rep = d / "campagna" / "straight" / "nsga3_finta" / "rep_01"
    (rep / "logs").mkdir(parents=True)
    (rep / "traces").mkdir()
    sep = "-" * 20 + "\n"
    (rep / "logs" / "mola_victim.log").write_text(sep + f"  topic LiDAR : {VIT}\n")
    if not senza_ombra:
        (rep / "logs" / "mola_shadow.log").write_text(sep + f"  topic LiDAR : {shadow_topic or VIT}\n")
    righe = []
    for i, x in enumerate(delta):
        ta = 100.0 + 0.1 * i
        r = {"t_arr_wall": 1000.0 + 0.1 * i, "t_arr_mono": ta, "stamp_in": round(5 + 0.1 * i, 1),
             "victim_frame_idx": i % (nuvole_per_genoma or len(delta)),
             "victim_stream": 1, "stamp_victim": round(5 + 0.1 * i, 1),
             "proc_ms": 80.0 if proc is None else proc[i], "t_release_target_mono": ta + dmin / 1000.0,
             "t_release_mono": ta + dmin / 1000.0, "d_min_ms": dmin, "delta_victim_ms": x,
             "delta_shadow_ms": "" if delta_ombra is None or i in omesse else delta_ombra[i]}
        righe.append(r)
    righe.append({"t_arr_wall": 2000.0, "t_arr_mono": 900.0, "stamp_in": 99.0, "victim_stream": 1})
    if modifica:
        modifica(righe)
    with open(d / "frame_log.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=colonne, extrasaction="ignore")
        w.writeheader()
        w.writerows(righe)
    if shadow_topic:
        with open(d / "shadow_crc.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["slot", "stamp", "victim_frame_idx", "shadow_omitted"])
            for i in range(len(delta)):
                w.writerow([i, 5 + 0.1 * i, i % (nuvole_per_genoma or len(delta)), int(i in omesse)])
    for k in range(0, len(delta), PER_FINESTRA):
        with open(rep / "traces" / f"w{k // PER_FINESTRA}_applied.csv", "w") as fh:
            fh.write("wall\n")
            fh.write(f"{1000.0 + 0.1 * k}\n{1000.0 + 0.1 * (min(k + PER_FINESTRA, len(delta)) - 1)}\n")
    return d


def sparsi(n, k, valore=6.0):
    """n on-time delays with k isolated mild clouds (distinct episodes)."""
    v = [1.0] * n
    for j in range(k):
        v[j * (n // k)] = valore
    return v


# ------------------------------------------------------------------ share and episodes
def test_quota_2_5_conforme(tmp_path):
    res = cr.analizza(run_finta(tmp_path, sparsi(200, 5)))
    assert res["vittima"]["quota"] == pytest.approx(0.025)
    assert res["vittima"]["conforme"] and res["codice"] == cr.ESITO_CONFORME


def test_quota_2_6_non_conforme(tmp_path):
    res = cr.analizza(run_finta(tmp_path, sparsi(500, 13)))
    assert res["vittima"]["quota"] == pytest.approx(0.026)
    assert not res["vittima"]["conforme"] and res["codice"] == cr.ESITO_RIT4


def test_un_episodio_grave_ammesso(tmp_path):
    v = [1.0] * 200
    v[10], v[11] = 60.0, 7.0          # one 2-cloud episode, maximum above 50 ms
    res = cr.analizza(run_finta(tmp_path, v))
    assert (res["vittima"]["episodi"], res["vittima"]["gravi"]) == (1, 1)
    assert res["codice"] == cr.ESITO_CONFORME


def test_due_episodi_gravi_rit4(tmp_path):
    v = [1.0] * 200
    v[10], v[100] = 60.0, 51.0
    res = cr.analizza(run_finta(tmp_path, v))
    assert res["vittima"]["gravi"] == 2 and res["vittima"]["quota"] <= 0.025
    assert res["codice"] == cr.ESITO_RIT4


def test_causa_calcolo(tmp_path):
    v = sparsi(200, 4)
    proc = [80.0] * 200
    proc[0], proc[50] = 136.0, 135.0  # only the first above D_min - 5
    s = cr.analizza(run_finta(tmp_path, v, proc=proc))["vittima"]
    assert (s["causa_calcolo"], s["causa_catena"]) == (1, 3)


def test_avviso_calcolo_in_run_rit4(tmp_path):
    v = sparsi(200, 8)
    proc = [140.0] * 200
    res = cr.analizza(run_finta(tmp_path, v, proc=proc))
    assert res["codice"] == cr.ESITO_RIT4 and res["avviso_calcolo"]


def test_victim_stream_senza_stamp_fuori_dal_denominatore(tmp_path):
    res = cr.analizza(run_finta(tmp_path, [1.0] * 100))
    assert res["victim_stream_senza_stamp_victim"] == 1 and res["vittima"]["n"] == 100


# ------------------------------------------------------------------ C1-C5
def _cambia(i, col, val, **altre):
    def f(righe):
        righe[i][col] = val
        righe[i].update(altre)
    return f


@pytest.mark.parametrize("modifica,ck", [
    (_cambia(3, "d_min_ms", 139.98, t_release_target_mono=100.3 + 0.13998), "C1"),
    (_cambia(3, "t_release_target_mono", 100.3 + 0.14002), "C2"),
    (_cambia(3, "delta_victim_ms", -0.2), "C3"),
    (_cambia(3, "delta_victim_ms", ""), "C4"),
])
def test_configurazione(tmp_path, modifica, ck):
    res = cr.analizza(run_finta(tmp_path, [1.0] * 100, modifica=modifica))
    assert not res["controlli"][ck]["ok"]
    assert all(v["ok"] for k, v in res["controlli"].items() if k != ck)
    assert res["codice"] == cr.ESITO_CONFIG


def test_c3_al_limite_ammesso(tmp_path):
    res = cr.analizza(run_finta(tmp_path, [1.0] * 100, modifica=_cambia(3, "delta_victim_ms", -0.1)))
    assert res["controlli"]["C3"]["ok"] and res["codice"] == cr.ESITO_CONFORME


def test_c5_colonna_assente(tmp_path):
    col = tuple(c for c in COL if c != "delta_victim_ms")
    res = cr.analizza(run_finta(tmp_path, [1.0] * 100, colonne=col))
    assert not res["controlli"]["C5"]["ok"] and res["codice"] == cr.ESITO_NON_MISURABILE


def test_registro_assente(tmp_path):
    assert cr.analizza(tmp_path)["codice"] == cr.ESITO_NON_MISURABILE


def test_ombra_non_letta(tmp_path):
    res = cr.analizza(run_finta(tmp_path, [1.0] * 100, senza_ombra=True))
    assert res["ombra"] is None and res["codice"] == cr.ESITO_NON_MISURABILE


# ------------------------------------------------------------------ shadow on its own topic
def test_ombra_stesso_messaggio(tmp_path):
    res = cr.analizza(run_finta(tmp_path, sparsi(200, 3)))
    assert not res["ombra_propria"] and res["ombra"] == res["vittima"]
    assert res["ombra_modo"] == "ombra = stesso messaggio della vittima"


def test_ombra_propria_non_conforme(tmp_path):
    res = cr.analizza(run_finta(tmp_path, [1.0] * 200, shadow_topic="/carter/lidar_shadow",
                                delta_ombra=sparsi(200, 10)))
    assert res["ombra_propria"] and res["vittima"]["conforme"] and not res["ombra"]["conforme"]
    assert res["codice"] == cr.ESITO_RIT4


def test_ombra_propria_omessa_non_conta(tmp_path):
    res = cr.analizza(run_finta(tmp_path, [1.0] * 100, shadow_topic="/carter/lidar_shadow",
                                delta_ombra=[1.0] * 100, omesse={7}))
    assert res["controlli"]["C4"]["ok"] and res["ombra"]["n"] == 99 and res["codice"] == cr.ESITO_CONFORME


def test_ombra_omessa_unita_su_stamp_non_su_indice(tmp_path):
    # two genomes of 50 clouds: victim_frame_idx restarts from 0, index 7 appears twice
    v = [1.0] * 100
    v[57] = 60.0                      # same index as the omitted cloud, in the second genome
    res = cr.analizza(run_finta(tmp_path, v, shadow_topic="/carter/lidar_shadow", delta_ombra=v,
                                omesse={7}, nuvole_per_genoma=50))
    assert res["controlli"]["C4"]["ok"]
    assert res["ombra"]["n"] == 99 and res["ombra"]["gravi"] == 1


def test_ombra_propria_senza_delta_configurazione(tmp_path):
    res = cr.analizza(run_finta(tmp_path, [1.0] * 100, shadow_topic="/carter/lidar_shadow"))
    assert not res["controlli"]["C4"]["ok"] and res["codice"] == cr.ESITO_CONFIG


# ------------------------------------------------------------------ windows
def test_finestre_marcate(tmp_path):
    v = [1.0] * 200
    v[5] = v[10] = v[15] = 6.0        # w0: 3 mild
    v[60] = v[70] = 6.0               # w1: 2 mild, allowed
    v[120] = 55.0                     # w2: 1 severe
    f = cr.analizza(run_finta(tmp_path, v))["finestre"]
    assert f["finestre"] == 4 and f["nuvole_senza_finestra"] == 0
    assert [m["finestra"] for m in f["marcate"]] == [0, 2]


def test_finestre_da_victim_senza_attacco(tmp_path):
    v = [1.0] * 200
    v[120] = 55.0
    d = run_finta(tmp_path, v)
    for p in (d / "campagna").rglob("w*_applied.csv"):
        p.rename(p.with_name(p.name.replace("applied", "victim")))
    f = cr.analizza(d)["finestre"]
    assert f["fonte"].startswith("w<k>_victim.csv") and f["finestre"] == 4
    assert [m["finestra"] for m in f["marcate"]] == [2]


def test_finestre_applied_prima_di_victim(tmp_path):
    d = run_finta(tmp_path, [1.0] * 100)
    rep = next((d / "campagna").glob("*/*/rep_01"))
    (rep / "traces" / "w9_victim.csv").write_text("wall\n1000.0\n1010.0\n")
    f = cr.analizza(d)["finestre"]
    assert f["fonte"] == "w<k>_applied.csv" and f["finestre"] == 2


def test_finestre_non_attribuibili(tmp_path):
    d = run_finta(tmp_path, [1.0] * 100)
    for p in (d / "campagna").rglob("w*_applied.csv"):
        p.rename(p.with_name(p.name.replace("applied", "nominal")))
    res = cr.analizza(d)
    assert res["finestre"] is None and "victim" in res["finestre_motivo"]
    assert res["codice"] == cr.ESITO_CONFORME


def test_cartella_della_rep_e_cli(tmp_path):
    d = run_finta(tmp_path, sparsi(500, 13))
    rep = next((d / "campagna").glob("*/*/rep_01"))
    out = tmp_path / "u.json"
    p = subprocess.run([sys.executable, str(SCRIPT), str(rep), "--json", str(out)], capture_output=True, text=True)
    assert p.returncode == cr.ESITO_RIT4, p.stdout + p.stderr
    assert "ESITO [4]: rit4" in p.stdout
    assert json.load(open(out))["codice"] == cr.ESITO_RIT4


# ------------------------------------------------------------------ real data, read only
G3 = (Path(__file__).resolve().parents[1] / "data/attack/verifica/2026-10-26_pilota_G3_dmin140")
# reference values: n, over 5 ms, max delta [ms], episodes, severe
TABELLA_S6 = {"s1001": (345, 1, 14.64, 1, 0), "s2001": (338, 3, 9.16, 3, 0), "s3001": (345, 1, 5.36, 1, 0),
              "s4001": (343, 1, 5.42, 1, 0), "s5001": (350, 3, 5.68, 3, 0)}


@pytest.mark.skipif(not G3.is_dir(), reason="dati del pilota G3 assenti")
@pytest.mark.parametrize("seme", sorted(TABELLA_S6))
def test_pilota_G3_tabella_s6(seme):
    res = cr.analizza(G3 / seme)
    s = res["vittima"]
    n, oltre, dmax, ep, g = TABELLA_S6[seme]
    assert (s["n"], s["oltre_5ms"], round(s["max"], 2), s["episodi"], s["gravi"]) == (n, oltre, dmax, ep, g)
    assert all(v["ok"] for v in res["controlli"].values())
    assert res["ombra_modo"] == "ombra = stesso messaggio della vittima"
    assert res["codice"] == cr.ESITO_CONFORME


PROVVISORIA = (Path(__file__).resolve().parents[1] / "data/attack/verifica/"
                   "2026-10-25_baseline_direction_s1001_sosta0")


@pytest.mark.skipif(not PROVVISORIA.is_dir(), reason="dati della provvisoria assenti")
def test_provvisoria_finestre_da_victim():
    # run without attack at D_min 85: windows from w<k>_victim.csv; code 5 for the missing shadow
    res = cr.analizza(PROVVISORIA, d_min=85.0)
    s, f = res["vittima"], res["finestre"]
    assert (s["n"], s["oltre_5ms"], round(s["max"], 2), s["gravi"]) == (378, 2, 7.56, 0)
    assert f is not None and f["fonte"].startswith("w<k>_victim.csv")
    assert (f["finestre"], f["nuvole_senza_finestra"], f["marcate"]) == (6, 0, [])
    assert res["codice"] == cr.ESITO_NON_MISURABILE and res["ombra"] is None
