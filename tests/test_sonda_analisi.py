"""Offline probe analysis on a synthetic run (frame_log, MOLA log, probe CSV)."""

import csv
import datetime
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import analisi_sonda as an  # noqa: E402
import sonda_nuvole as sn  # noqa: E402

W0 = datetime.datetime(2026, 10, 29, 10, 0, 0).timestamp()
M0 = 5000.0
N = 20
FL_COL = ("t_arr_wall", "t_arr_mono", "stamp_victim", "stamp_main_out", "t_pub_main_end_mono",
          "t_pub_victim_end_mono", "t_pub_attacker_end_mono", "t_dds_src")


def ora(t):
    d = datetime.datetime.fromtimestamp(t)
    return d.strftime("%H:%M:%S.") + f"{d.microsecond // 100:04d}"


def riga_presa(t, stamp):
    return (f"[{ora(t)}|DEBUG|Generator (raw)] Processing observation "
            f"type='mrpt::obs::CObservationPointCloud' label='lidar' stamp={stamp:.4f}")


def costruisci(tmp_path):
    run = tmp_path / "run"
    (run / "logs").mkdir(parents=True)
    (run / "sonda").mkdir()
    pub = []
    with open(run / "frame_log.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(FL_COL)
        for k in range(N):
            tw, tm, st = W0 + 0.1 * k, M0 + 0.1 * k, 10.0 + 0.1 * k
            vit = k < 10   # victim in the first 10 frames
            w.writerow((tw, tm, st if vit else "", st, tm + 0.141,
                        tm + 0.142 if vit else "", tm + 0.143 if vit else "", tw - 0.002))
            pub.append((k, tw, st))
    # probe on the victim topic: k=5 arrives late, k=8 is missing
    with open(run / "sonda" / sn.nome_file("/carter/lidar_victim"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(sn.COLONNE)
        for k, tw, st in pub[:10]:
            if k == 8:
                continue
            src = tw + 0.142
            rec = src + (0.5 if k == 5 else 0.001)
            w.writerow((rec, 0, f"{st:.9f}", f"{src:.9f}", f"{rec:.9f}", (rec - src) * 1e3, 722864, k, k))
    log = [f"[{ora(W0 - 1)}|INFO |MolaLauncherApp] avvio"]
    for k, tw, st in pub[:10]:
        if k == 7:
            continue                       # never taken
        ritardo = {3: 0.6, 5: 0.7, 8: 0.6}.get(k, 0.15)
        log.append(riga_presa(tw + ritardo, st))
    log.append(f"[{ora(W0 + 3)}|INFO |MolaLauncherApp] fine")
    (run / "logs" / "mola_victim.log").write_text("\n".join(log) + "\n")
    # twin alive between frames 12 and 17: takes 12..17 except 14
    g = [f"[{ora(W0 + 1.0)}|INFO |MolaLauncherApp] avvio"]
    g += [riga_presa(W0 + 0.1 * k + 0.16, 10.0 + 0.1 * k) for k in range(12, 18) if k != 14]
    g.append(f"[{ora(W0 + 0.1 * 17 + 0.6)}|INFO |MolaLauncherApp] fine")
    (run / "logs" / "mola_w0_e0.log").write_text("\n".join(g) + "\n")
    return run


def test_leggi_log_mola_giorno_e_prese():
    t = W0 + 12.34
    prese, t0, t1 = an.leggi_log_mola(riga_presa(t, 12.4) + "\n[xx] riga senza ora\n", W0)
    assert prese == [(pytest.approx(t, abs=1e-4), 12.4)]
    assert t0 == t1 == pytest.approx(t, abs=1e-4)


def test_fine_vita_alla_riga_di_shutdown():
    testo = "\n".join([riga_presa(W0 + 1, 1.0), "Caught signal 2. Shutting down...",
                       f"[{ora(W0 + 2)}|INFO |MolaLauncherApp] shutdown(): Shutting down 3 module threads...",
                       riga_presa(W0 + 2.1, 1.1),
                       f"[{ora(W0 + 2.5)}|INFO |MolaLauncherApp] shutdown(): Done."])
    prese, t0, t1 = an.leggi_log_mola(testo, W0)
    assert len(prese) == 2
    assert t1 == pytest.approx(W0 + 2, abs=1e-4)


def test_classe():
    assert an.classe(0.1, 0.2, 1.0) is None
    assert an.classe(0.1, 0.6, 1.0) == "arriva in tempo e aspetta dentro MOLA"
    assert an.classe(0.5, 0.6, 1.0) == "arriva tardi al processo"
    assert an.classe(float("nan"), 0.6, 1.0) == "non vista dalla sonda"
    assert an.classe(0.1, float("nan"), None) == "arriva in tempo e aspetta dentro MOLA"


def test_analisi_run_sintetica(tmp_path):
    run = costruisci(tmp_path)
    righe, mancanti = an.analizza(run, run / "frame_log.csv", run / "sonda")
    assert mancanti == ["/carter/lidar_perturbed"]
    v = {round(r["stamp"], 1): r for r in righe if r["mola"] == "victim"}
    assert len(v) == 10
    assert v[10.3]["classe"] == "arriva in tempo e aspetta dentro MOLA"
    assert v[10.3]["ritardo_dds_s"] == pytest.approx(0.001, abs=1e-6)
    assert v[10.3]["attesa_mola_s"] == pytest.approx(0.6 - 0.002 - 0.142 - 0.001 + 0.002, abs=2e-4)
    assert v[10.5]["classe"] == "arriva tardi al processo"
    assert v[10.7]["stato"] == "non elaborata"
    assert v[10.7]["classe"] == "arriva in tempo e aspetta dentro MOLA"
    assert v[10.8]["classe"] == "non vista dalla sonda"
    assert v[10.0]["classe"] is None and v[10.0]["stato"] == "presa"
    g = [r for r in righe if r["mola"] == "w0_e0" and r["stato"] != "fuori finestra"]
    assert sorted(round(r["stamp"], 1) for r in g) == [11.2, 11.3, 11.4, 11.5, 11.6, 11.7]
    assert [round(r["stamp"], 1) for r in g if r["stato"] == "non elaborata"] == [11.4]
    out = tmp_path / "out"
    rip = an.scrivi(righe, mancanti, out)
    assert rip["mola"]["victim"]["non_elaborate"] == 1
    assert rip["mola"]["victim"]["classi"] == {"arriva in tempo e aspetta dentro MOLA": 2,
                                               "arriva tardi al processo": 1,
                                               "non vista dalla sonda": 1}
    assert json.loads((out / "riepilogo_sonda.json").read_text())["limite"] == an.LIMITE
    ne = list(csv.DictReader(open(out / "non_elaborate.csv")))
    assert sorted(r["mola"] for r in ne) == ["victim", "w0_e0"]
    with pytest.raises(SystemExit):
        an.scrivi(righe, mancanti, out)   # non-empty folder: no overwrite
