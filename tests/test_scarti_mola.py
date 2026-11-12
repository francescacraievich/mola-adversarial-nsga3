"""Tests of src/analysis/scarti_mola.py: synthetic parsing and, when present, some real logs
and runs.

Run:  python3 -m pytest tests/test_scarti_mola.py -q
"""
import math
import os
import sys

import pytest

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO, "src", "analysis"))
import scarti_mola as sm  # noqa: E402

DATI = os.path.expanduser("~/mola-adversarial-nsga3/data/attack/campaign/straight")
VERIFICA_A = os.path.join(DATI, "none_g1_sel2_warm_nomreps1_verifica_A")
SOGLIA70 = os.path.join(DATI, "nsga3_g3_p32x16_eps70_lag0.5_sel2_warm_H1.0_soglia70")

PRINTF_BUONA = ("[LidarOdometry] pathStep=2 timestamp=1970/01/01,00:00:12.400000 goodness=0.984 "
                "minRequired=0.200 iters=1 termReason=IterTermReason::Stalled isGood=1 "
                "adapt_thres_sigma=0.1975 consecutive_bad=0")
PRINTF_RIFIUTATA = ("[LidarOdometry] pathStep=3 timestamp=1970/01/01,00:00:12.500000 goodness=0.000 "
                    "minRequired=0.200 iters=10 termReason=IterTermReason::MaxIterations isGood=0 "
                    "adapt_thres_sigma=0.1975 consecutive_bad=1")
# Warning row containing "pathStep=" that is not an ICP printf: ignored.
AVVISO = ("[11:31:29.4967|WARN |LidarOdometry] Not able to use velocity motion model for this "
          "timestep (pathStep=5, timestamp=1970/01/01,00:00:13.100000 UTC)")
TF_OLD = ("[WARN] [1792406625.887163418] []: TF_OLD_DATA ignoring data from the past for frame "
          "base_link at time 29.600000 according to authority Authority undetectable")

COLONNE = ["ADAPTIVE_THRESHOLD_SIGMA", "current_relative_timestamp", "icp_iterations",
           "icp_quality", "robot_x", "robot_y", "time_onLidar", "timestamp"]


def _csv_sintetico(path, righe, tronca=False):
    with open(path, "w", encoding="utf-8") as f:
        f.write(",".join(f'"{c}"' for c in COLONNE) + ",\n")
        for r in righe:
            f.write(",".join(f"{v:.6f}" for v in r) + ",\n")
        if tronca:
            f.write("0.040000,7.700000,2.000000,0.78")   # row broken by SIGKILL
    return path


def _riga(rel, it, q, x, stamp, sigma=0.2, y=0.0, tl=0.01):
    return [sigma, rel, it, q, x, y, tl, stamp]


# ----------------------------------------------------------------------------- printf
def test_leggi_printf_buona_e_rifiutata(tmp_path):
    log = tmp_path / "mola_victim.log"
    log.write_text("\n".join(["[INFO] avvio", PRINTF_BUONA, AVVISO, PRINTF_RIFIUTATA, TF_OLD]) + "\n")
    rec = sm.leggi_printf(str(log))
    assert len(rec) == 2
    b, r = rec
    assert b["stamp_s"] == 12.4 and b["goodness"] == 0.984 and b["is_good"] == 1
    assert b["iters"] == 1 and b["term_reason"] == "Stalled" and b["pathstep"] == 2
    assert b["min_required"] == 0.2 and b["sigma"] == 0.1975 and b["consecutive_bad"] == 0
    assert r["stamp_s"] == 12.5 and r["goodness"] == 0.0 and r["is_good"] == 0
    assert r["iters"] == 10 and r["term_reason"] == "MaxIterations" and r["consecutive_bad"] == 1
    assert sm.ultimo_stamp_tf_old_data(str(log)) == 29.6


MOTIVI = "Possible reasons are listed at http://wiki.ros.org/tf/Errors%20explained"
SEGNALE = "[INFO] [1792732784.467582587] [rclcpp]: signal_handler(SIGINT/SIGTERM)"
PRINTF_SPEZZATA = ("[LidarOdometry] pathStep=4 timestamp=1970/01/01,00:00:12.600000 goodness=0.000 "
                   "minRequired=0.200 iters=10 termReason=IterTermReason::MaxIterations isGood=0 "
                   "adapt_thres_sigma=0.1975 consecutive_bad=12")
PRINTF_ATTACCATA = ("[LidarOdometry] pathStep=5 timestamp=1970/01/01,00:00:12.700000 goodness=0.950 "
                    "minRequired=0.200 iters=1 termReason=IterTermReason::Stalled isGood=1 "
                    "adapt_thres_sigma=0.1950 consecutive_bad=0")


def _leggi_printf_riga_per_riga(log_path):
    """leggi_printf before the stderr clean-up: reference for the comparisons."""
    out = []
    with open(log_path, encoding="utf-8", errors="replace") as f:
        for i, riga in enumerate(f, 1):
            m = sm.RX_PRINTF.search(riga)
            if not m:
                continue
            out.append(dict(
                stamp_s=sm._stamp(sm._stamp_da_data(m.group(2))), goodness=float(m.group(3)),
                min_required=float(m.group(4)), iters=int(m.group(5)),
                term_reason=m.group(6).replace("IterTermReason::", ""), is_good=int(m.group(7)),
                sigma=float(m.group(8)), consecutive_bad=int(m.group(9)), pathstep=int(m.group(1)),
                riga=i))
    return out


def _stderr_a_meta_riga(log_path):
    with open(log_path, encoding="utf-8", errors="replace") as f:
        t = f.read()
    return sum(1 for m in sm.RX_STDERR.finditer(t) if m.start() > 0 and t[m.start() - 1] != "\n")


def test_leggi_printf_righe_spezzate_da_stderr(tmp_path):
    # As in real logs: block-buffered stdout is cut mid-row by TF_OLD_DATA warnings
    # or by the signal_handler row, and the printf row continues after them.
    righe = [
        "[INFO] avvio",                                    # 1
        PRINTF_BUONA,                                      # 2
        TF_OLD, MOTIVI,                                    # 3-4: full-row stderr
        PRINTF_RIFIUTATA[:30] + TF_OLD,                    # 5: broken record
        MOTIVI, TF_OLD, MOTIVI,                            # 6-8
        PRINTF_RIFIUTATA[30:],                             # 9: rest of the record
        AVVISO,                                            # 10
        PRINTF_SPEZZATA[:-1] + TF_OLD,                     # 11: broken inside consecutive_bad=12
        MOTIVI,                                            # 12
        PRINTF_SPEZZATA[-1:],                              # 13
        PRINTF_BUONA[:60] + SEGNALE,                       # 14: broken by signal_handler
        PRINTF_BUONA[60:],                                 # 15
        TF_OLD[:120] + PRINTF_ATTACCATA,                   # 16: warning cut, whole record after
    ]
    log = tmp_path / "mola_victim.log"
    log.write_text("\n".join(righe) + "\n")
    rec = sm.leggi_printf(str(log))
    assert [r["pathstep"] for r in rec] == [2, 3, 4, 2, 5]
    assert [r["riga"] for r in rec] == [2, 5, 11, 14, 16]
    assert rec[1]["is_good"] == 0 and rec[1]["iters"] == 10 and rec[1]["consecutive_bad"] == 1
    assert rec[2]["consecutive_bad"] == 12 and rec[2]["stamp_s"] == 12.6
    assert rec[3] == dict(rec[0], riga=14)
    assert rec[4]["stamp_s"] == 12.7 and rec[4]["goodness"] == 0.95
    # row-by-row reading loses records 3 and the second 2 and reads consecutive_bad=1 instead of 12
    vecchia = _leggi_printf_riga_per_riga(str(log))
    assert [(r["pathstep"], r["consecutive_bad"]) for r in vecchia] == [(2, 0), (4, 1), (5, 0)]
    assert sm.ultimo_stamp_tf_old_data(str(log)) == 29.6


def test_testo_log_numeri_di_riga(tmp_path):
    log = tmp_path / "mola_victim.log"
    log.write_text("\n".join(["a", TF_OLD, MOTIVI, "b", "c" + SEGNALE, "d"]) + "\n")
    testo, riga_nel_file = sm.testo_log(str(log))
    assert testo == "a\nb\ncd\n"
    assert [riga_nel_file(p) for p in range(len(testo))] == [1, 1, 4, 4, 5, 6, 6]


def test_stamp_da_data_oltre_il_minuto():
    assert sm._stamp(sm._stamp_da_data("1970/01/01,00:01:05.300000")) == 65.3


# ----------------------------------------------------------------------------- csv
def test_leggi_csv_prima_riga_esclusa_e_riga_tronca(tmp_path):
    p = _csv_sintetico(tmp_path / "traces_victim.csv", [
        _riga(0.0, 0, 0.0, 0.0, 12.3),            # first scan, no ICP: excluded
        _riga(0.1, 1, 0.9836, 0.0, 12.4),
        _riga(0.2, 10, 0.0, 0.05, 12.5),          # rejected: fresh q at zero
        _riga(0.3, 2, 0.1995, 0.10, 12.6),        # just below threshold
        _riga(0.4, 2, 0.2000, 0.15, 12.7),        # on the threshold: not dropped
    ], tronca=True)
    rec = sm.leggi_csv_mola(str(p), 0.20)
    assert [r["stamp_s"] for r in rec] == [12.4, 12.5, 12.6, 12.7]
    assert [r["scartato"] for r in rec] == [False, True, True, False]
    assert rec[1]["iters"] == 10 and rec[1]["q"] == 0.0 and rec[1]["robot_x"] == 0.05
    assert rec[0]["time_onLidar"] == 0.01 and rec[0]["sigma"] == 0.2
    assert sm._righe_csv(str(p))[2] == 1


# ----------------------------------------------------------------------------- threshold
def test_soglia_dallo_yaml_con_e_senza_variabile(tmp_path):
    y = tmp_path / "pipeline.yaml"
    y.write_text("params:\n  # commento\n  min_icp_goodness: ${MOLA_MINIMUM_ICP_QUALITY|0.20} # nota\n")
    v, fonte = sm.soglia_min_icp_goodness(str(y))
    assert v == 0.20 and fonte.endswith(":3 (default di ${MOLA_MINIMUM_ICP_QUALITY|0.20})")
    v, fonte = sm.soglia_min_icp_goodness(str(y), {"MOLA_MINIMUM_ICP_QUALITY": "0.35"})
    assert v == 0.35 and "dall'ambiente" in fonte
    y.write_text("  min_icp_goodness: 0.5\n")
    assert sm.soglia_min_icp_goodness(str(y))[0] == 0.5
    y.write_text("altro: 1\n")
    with pytest.raises(ValueError):
        sm.soglia_min_icp_goodness(str(y))


def test_soglia_dalla_configurazione_della_repo():
    cfg = os.path.join(REPO, "config", "mola", "lidar3d-gicp-optimize-twist-local.yaml")
    if not os.path.exists(cfg):
        pytest.skip("configurazione di MOLA assente")
    v, fonte = sm.soglia_min_icp_goodness(cfg)
    assert v == 0.20 and ":72 " in fonte


# ----------------------------------------------------------------------------- join
def test_unisci_con_disallineamento():
    printf = [
        dict(stamp_s=12.4, goodness=0.984, min_required=0.2, iters=1, term_reason="Stalled",
             is_good=1, sigma=0.2, consecutive_bad=0, pathstep=2, riga=10),
        dict(stamp_s=12.5, goodness=0.000, min_required=0.2, iters=10, term_reason="MaxIterations",
             is_good=0, sigma=0.2, consecutive_bad=1, pathstep=3, riga=11),
        dict(stamp_s=12.6, goodness=0.700, min_required=0.2, iters=2, term_reason="Stalled",
             is_good=1, sigma=0.2, consecutive_bad=0, pathstep=4, riga=12),
        dict(stamp_s=12.8, goodness=0.650, min_required=0.2, iters=3, term_reason="Stalled",
             is_good=1, sigma=0.2, consecutive_bad=0, pathstep=5, riga=13),   # tail: CSV lost
    ]
    csv_rec = [
        dict(stamp_s=12.4, q=0.9836, iters=1, robot_x=0.0, robot_y=0.0, sigma=0.2,
             time_onLidar=0.01, scartato=False, riga=3),
        dict(stamp_s=12.5, q=0.0, iters=10, robot_x=0.05, robot_y=0.0, sigma=0.2,
             time_onLidar=0.02, scartato=True, riga=4),
        dict(stamp_s=12.6, q=0.7100, iters=2, robot_x=0.1, robot_y=0.0, sigma=0.2,
             time_onLidar=0.01, scartato=False, riga=5),                     # inconsistent q
        dict(stamp_s=12.7, q=0.6000, iters=2, robot_x=0.15, robot_y=0.0, sigma=0.2,
             time_onLidar=0.01, scartato=False, riga=6),                     # printf missing
    ]
    rec = sm.unisci(printf, csv_rec, soglia=0.2)
    assert [r["stamp_s"] for r in rec] == [12.4, 12.5, 12.6, 12.7, 12.8]
    assert [r["fonte"] for r in rec] == ["printf+csv", "printf+csv", "printf+csv", "csv", "printf"]
    assert rec[0]["incoerente"] is False and rec[0]["scartato"] is False
    assert rec[1]["scartato"] is True and rec[1]["scartato_csv"] is True and rec[1]["is_good"] == 0
    assert rec[2]["incoerente"] is True and "0.7100 vs goodness 0.700" in rec[2]["nota"]
    assert rec[3]["is_good"] is None and rec[3]["scartato"] is False
    assert math.isnan(rec[4]["robot_x"]) and rec[4]["scartato"] is False
    rp = sm.riepilogo(rec, printf, csv_rec, n_righe_tronche=1, stamp_tf_old=13.0, soglia=0.2)
    assert rp["fonte_principale"].startswith("printf")
    assert rp["n_scan_processati"] == 4 and rp["n_righe_printf"] == 4 and rp["n_record"] == 5
    assert rp["n_scartati"] == 1 and rp["n_incoerenti"] == 1 and rp["n_solo_csv"] == 1
    assert rp["n_solo_printf"] == 1 and rp["n_attesi_10hz"] == 5 and rp["n_buchi"] == 0
    assert rp["coda_persa_csv_s"] == pytest.approx(0.3) and rp["coda_persa_printf_s"] == pytest.approx(0.2)
    assert rp["n_righe_csv_tronche"] == 1


def test_unisci_senza_printf_usa_il_csv():
    csv_rec = [dict(stamp_s=1.0, q=0.1, iters=5, robot_x=0.0, robot_y=0.0, sigma=0.2,
                    time_onLidar=0.01, scartato=True, riga=2),
               dict(stamp_s=1.3, q=0.5, iters=1, robot_x=0.1, robot_y=0.0, sigma=0.2,
                    time_onLidar=0.01, scartato=False, riga=3)]
    rec = sm.unisci([], csv_rec, soglia=0.2)
    assert [r["scartato"] for r in rec] == [True, False] and all(r["fonte"] == "csv" for r in rec)
    rp = sm.riepilogo(rec, [], csv_rec)
    assert rp["fonte_principale"].startswith("csv") and rp["n_buchi"] == 1 and rp["buchi"] == [(1.0, 1.3, 2)]
    assert rp["n_attesi_10hz"] == 4 and math.isnan(rp["coda_persa_printf_s"]) and rp["coda_persa_csv_s"] == 0.0


# ----------------------------------------------------------------------------- per metre
def test_scarti_per_metro():
    def r(x, stamp, scartato, q=0.5, csv=None):
        return dict(robot_x=x, stamp_s=stamp, scartato=scartato, q=q,
                    scartato_csv=(scartato if csv is None else csv), fonte="csv", incoerente=False)
    rec = [r(0.0, 1.0, False), r(0.5, 1.1, True, 0.0), r(0.9, 1.3, False), r(1.2, 1.4, False),
           r(2.7, 1.5, True, 0.1, csv=False), r(2.95, 1.9, False, 0.3),
           dict(robot_x=float("nan"), stamp_s=2.0, scartato=True, q=0.0, scartato_csv=True,
                fonte="printf", incoerente=False)]
    bins = sm.scarti_per_metro(rec, 1.0)
    assert [(b["bin_inizio_m"], b["n_scan"]) for b in bins] == [(0.0, 3), (1.0, 1), (2.0, 2)]
    assert [b["n_scartati"] for b in bins] == [1, 0, 1]
    assert [b["n_scartati_csv"] for b in bins] == [1, 0, 0]
    assert [b["n_attesi_10hz"] for b in bins] == [4, 1, 5]
    assert bins[0]["q_min"] == 0.0 and bins[2]["q_min"] == 0.1
    assert sm.scarti_per_metro([], 1.0) == []


def test_allinea_q():
    rec = [dict(q=v) for v in (0.9, 0.6728, 0.6484, 0.0, 0.6304, 0.6304, 0.5428)]
    idx = sm.allinea_q(rec, [0.6728, 0.6484, 0.6304, 0.5428, 0.1234])
    assert idx == [1, 2, 4, 6, None]


def test_allinea_q_con_q_ripetuta_e_tempi():
    # 0.9948 occurs several times before the first trace sample; with the times the
    # occurrence consistent with the offset stamp - t is chosen, not the first.
    rec = [dict(q=q, stamp_s=s) for q, s in ((0.9, 13.0), (0.9948, 13.3), (0.5, 13.4), (0.9948, 13.7),
                                              (0.6, 13.8), (0.9948, 14.0), (0.9944, 14.1), (0.7, 14.2))]
    seq, tempi = [0.9948, 0.9944, 0.7], [0.0, 0.1, 0.2]
    assert sm.allinea_q(rec, seq) == [1, 6, 7]                      # greedy: first occurrence
    assert sm.allinea_q(rec, seq, tempi=tempi) == [5, 6, 7]         # with times: constant offset
    assert sm.allinea_q(rec, [0.9948, 0.1234, 0.7], tempi=tempi) == [5, None, 7]
    assert sm.allinea_q(rec, [0.1234], tempi=[0.0]) == [None]


def test_statistiche_time_on_lidar():
    rec = [dict(time_onLidar=v / 1000.0) for v in range(1, 101)] + [dict(time_onLidar=float("nan"))]
    s = sm.statistiche_time_on_lidar(rec)
    assert s["time_onLidar_n"] == 100 and s["time_onLidar_media_ms"] == pytest.approx(50.5)
    assert s["time_onLidar_p99_ms"] == 99.0 and s["time_onLidar_p99_metodo"].startswith("nearest")


# ----------------------------------------------------------------------------- real runs
# Real logs with whole ROS stderr rows but no broken row (attacker of a live baseline
# run and of a threshold run) and a victim log with four broken records.
LOG_SENZA_SPEZZATE = [
    os.path.join(DATI, "none_g3_sel2_H1.0_baseline_live_r3", "rep_01", "logs", "mola_attacker.log"),
    os.path.join(DATI, "nsga3_g3_p32x16_eps30_lag0.5_sel3_warm_H1.0_soglia30_dannobaseline", "rep_01", "logs",
                 "mola_attacker.log"),
]
LOG_36A = os.path.expanduser(
    "~/mola-adversarial-nsga3/data/attack/verifica/2026-10-24_prova_3.6_rampa/straight/"
    "none_g3_sel2_stopramp_k1.0_vmin0.05_reliable_serieB_s1001_pt1_oprandom_H1.0_baseline_rampa/"
    "rep_01/logs/mola_victim.log")


@pytest.mark.parametrize("log", LOG_SENZA_SPEZZATE)
def test_log_reale_senza_righe_spezzate_come_prima(log):
    if not os.path.exists(log):
        pytest.skip("log assente")
    assert _stderr_a_meta_riga(log) == 0
    with open(log, encoding="utf-8", errors="replace") as f:
        assert len(list(sm.RX_STDERR.finditer(f.read()))) > 5
    vecchia = _leggi_printf_riga_per_riga(log)
    assert len(vecchia) > 50 and sm.leggi_printf(log) == vecchia


@pytest.mark.skipif(not os.path.exists(LOG_36A), reason="log della 3.6a assente")
def test_log_reale_36a_record_spezzati():
    vecchia, nuova = _leggi_printf_riga_per_riga(LOG_36A), sm.leggi_printf(LOG_36A)
    assert len(vecchia) == 218 and len(nuova) == 222
    assert [r for r in nuova if r in vecchia] == vecchia
    agg = [r for r in nuova if r not in vecchia]
    assert [(r["pathstep"], r["stamp_s"], r["riga"]) for r in agg] == [
        (106, 21.1, 292), (135, 24.0, 437), (184, 28.9, 682), (223, 32.8, 875)]
    assert [r["pathstep"] for r in nuova] == list(range(2, 224))


@pytest.mark.skipif(not os.path.isdir(VERIFICA_A), reason="run verifica_A assente")
def test_run_reale_verifica_a_con_printf():
    rec, rp, bins = sm.analizza_run(VERIFICA_A, 0.20)
    assert rp["n_righe_printf"] == 76 and rp["fonte_principale"].startswith("printf")
    assert rp["n_scan_processati"] == 66 and rp["n_incoerenti"] == 0
    assert rp["n_scartati"] == 0 and rp["n_scartati_csv"] == 0
    assert rp["n_solo_printf"] == 10 and rp["ultimo_stamp_csv_s"] == 19.9 and rp["ultimo_stamp_printf_s"] == 21.1
    assert rp["n_righe_csv_tronche"] == 1 and rp["coda_persa_csv_s"] == pytest.approx(1.2)
    assert sum(b["n_scan"] for b in bins) == 66


@pytest.mark.skipif(not os.path.isdir(SOGLIA70), reason="run soglia70 assente")
def test_run_reale_soglia70_senza_printf():
    rec, rp, bins = sm.analizza_run(SOGLIA70, 0.20)
    assert rp["n_righe_printf"] == 0 and rp["fonte_principale"].startswith("csv")
    assert rp["n_scan_processati"] == 169 and rp["n_scartati"] == 1
    scarti = [r for r in rec if r["scartato"]]
    assert len(scarti) == 1
    s = scarti[0]
    assert s["stamp_s"] == 23.2 and s["q"] == 0.0 and s["iters"] == 10 and s["robot_x"] == pytest.approx(2.708, abs=1e-3)
    assert rp["ultimo_stamp_tf_old_data_s"] == 29.6 and rp["coda_persa_csv_s"] == pytest.approx(0.5)
    assert [b["n_scartati"] for b in bins if b["bin_inizio_m"] == 2.0] == [1]
