"""Baseline in direction mode: building the polyline from a run without attack
(scripts/costruisci_baseline_direction.py) and launch options
(scripts/prova_banco_serieB_baseline_direction.sh).

Fake data in tmp_path: one repetition with history.json, traces and the `completata` file
as written by the orchestrator and run_campaign.sh. The dry-run step of the launch script
starts no process; it is skipped if a bench run is in progress. No ROS node.
"""

import csv
import json
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src" / "analysis"))

import costruisci_baseline_direction as cb  # noqa: E402
from baseline_distance import load_baseline  # noqa: E402

COLONNE = ["t", "est_x", "est_y", "true_x", "true_y", "true_vx", "true_vy", "wall"]


def _traccia(p, xs, ys):
    with open(p, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(COLONNE)
        for i, (x, y) in enumerate(zip(xs, ys)):
            w.writerow([i * 0.1, x, y, x, y, 0.8, 0.0, 1.0e9 + i])


def _rep(tmp_path, n=6, passo=1.0, x0=0.011, seme=1001, ritocchi=None):
    """Fake repetition: n windows of `passo` m in direction, repeated poses (as in real
    traces) and a small lateral offset."""
    base = tmp_path / "campagna" / "straight" / "none_g3_baseline_direction_N6"
    rep = base / "rep_01"
    (rep / "traces").mkdir(parents=True)
    hist = []
    for k in range(n):
        a, b = x0 + k * passo, x0 + (k + 1) * passo
        xs = np.repeat(np.linspace(a, b, 26), 2)          # consecutive duplicates
        ys = 0.002 * np.sin(xs)
        _traccia(rep / "traces" / f"w{k}_victim.csv", xs, ys)
        _traccia(rep / "traces" / f"w{k}_nominal.csv", xs + 0.01, ys)
        hist.append({"window": k, "status": "ok", "victim": "continuous", "victim_status": "ok",
                     "victim_lost": False, "victim_travelled": passo, "victim_track_ratio": 0.99,
                     "window_time_s": 1821.0})
    hist[-1]["run_end"] = {"reason": "finestre_completate", "windows_driven": n, "windows_held": 0}
    if ritocchi:
        ritocchi(rep, hist)
    (rep / "history.json").write_text(json.dumps(hist))
    (rep / "completata").write_text("2026-10-25T03:00:00\n")
    (base / "rep_01.log").write_text(f"  tracce        : x\n  seed          : {seme}\n")
    return rep


def _costruisci(rep, out, *extra):
    csv_p, js = out / "b.csv", out / "b.json"
    rc = cb.main([str(rep), "--csv", str(csv_p), "--json", str(js), "--seme", "1001", *extra])
    return rc, csv_p, js


def test_polilinea_formato_e_lettore(tmp_path):
    rep = _rep(tmp_path)
    rc, csv_p, js = _costruisci(rep, tmp_path)
    assert rc == 0 and csv_p.exists() and js.exists()
    righe = csv_p.read_text().splitlines()
    assert righe[0] == "s_m,x_m,y_m"
    # same digits as the older baseline
    assert re.fullmatch(r"\d+\.\d{4},-?\d+\.\d{6},-?\d+\.\d{6}", righe[1])
    poly = load_baseline(csv_p)
    s = np.array([float(r.split(",")[0]) for r in righe[1:]])
    # the last point is at s = L: it may coincide with the last step multiple at 4 digits
    assert poly.shape == (len(s), 2) and np.all(np.diff(s) >= 0)
    assert np.allclose(np.diff(s)[:-1], 0.05, atol=1e-4)      # 0.05 step, last step shorter
    assert abs(poly[0, 0] - 0.011) < 1e-6 and abs(poly[-1, 0] - 6.011) < 1e-6
    meta = json.loads(js.read_text())
    assert meta["controlli_superati"] is True
    assert meta["finestre_guidate"] == list(range(6))
    assert meta["pose_vere_distinte"] == 6 * 26 - 5             # duplicates removed at joins too
    assert meta["scarto_pose_dalla_polilinea_cm"]["max"] < 0.01
    assert meta["csv_sha256"] == cb.sha256(csv_p)


def test_uguale_alla_ricetta_della_serie_A(tmp_path):
    # a single run: averaging changes nothing and the last point is at s = L
    rep = _rep(tmp_path, n=4, passo=1.17)
    xy, _, _ = cb.percorso_vero(rep, range(4))
    s, poly, L = cb.polilinea(xy, 0.05)
    s_raw = cb.arc_length(xy)
    sq = np.arange(0.0, s_raw[-1] + 1e-9, 0.05)
    sq = sq[sq <= s_raw[-1]]
    assert np.allclose(s[:-1], sq) and abs(s[-1] - L) < 1e-12 and L > sq[-1]
    assert np.allclose(poly[:, 0], np.interp(s, s_raw, xy[:, 0]))


def test_non_sovrascrive(tmp_path):
    rep = _rep(tmp_path)
    rc, csv_p, js = _costruisci(rep, tmp_path)
    assert rc == 0
    prima = csv_p.read_bytes()
    js.unlink()  # the CSV alone is enough to stop
    rc2, _, _ = _costruisci(rep, tmp_path)
    assert rc2 == 2 and csv_p.read_bytes() == prima and not js.exists()


def _tieni(k):
    def f(rep, hist):
        hist[k].update({"victim_held": True})
    return f


def _senza_posa(rep, hist):
    hist[3].update({"victim_status": "victim_no_pose", "victim_lost": True})


def _arrivato(rep, hist):
    hist[-1]["victim_status"] = "arrived"


def _nominale_oltre(rep, hist):
    _traccia(rep / "traces" / "w5_nominal.csv", [6.0, 7.2], [0.0, 0.0])


def _senza_run_end(rep, hist):
    hist[-1].pop("run_end")


def _traccia_mancante(rep, hist):
    (rep / "traces" / "w2_victim.csv").unlink()


@pytest.mark.parametrize("caso,kw,controllo", [
    ("cinque finestre", dict(n=5), "finestre guidate"),
    ("finestra trattenuta", dict(ritocchi=_tieni(2)), "finestre trattenute (victim_held)"),
    ("perdita di posa", dict(ritocchi=_senza_posa), "esiti della vittima (nessuna perdita di posa, nessun arrived)"),
    ("arrived", dict(ritocchi=_arrivato), "esiti della vittima (nessuna perdita di posa, nessun arrived)"),
    ("x finale oltre 7", dict(passo=1.2), "x odom vera finale della vittima [m]"),
    ("nominale oltre 7", dict(ritocchi=_nominale_oltre), "x odom vera massima su tutte le tracce [m]"),
    ("run_end assente", dict(ritocchi=_senza_run_end), "run_end.reason"),
    ("traccia mancante", dict(ritocchi=_traccia_mancante), "traccia della vittima per ogni finestra guidata (>= 2 pose)"),
    ("seme diverso", dict(seme=2001), "seme della run (log della ripetizione)"),
])
def test_controlli_falliti_niente_csv(tmp_path, caso, kw, controllo):
    rep = _rep(tmp_path, **kw)
    rc, csv_p, js = _costruisci(rep, tmp_path)
    assert rc == 1, caso
    assert not csv_p.exists()
    meta = json.loads(js.read_text())
    assert meta["controlli_superati"] is False and meta["csv"] is None
    ko = [c["controllo"] for c in meta["controlli"] if not c["ok"]]
    assert controllo in ko, (caso, ko)


def test_cartella_non_completata(tmp_path):
    rep = _rep(tmp_path)
    (rep / "completata").unlink()
    rc, csv_p, _ = _costruisci(rep, tmp_path)
    assert rc == 1 and not csv_p.exists()


def test_controllo_x(tmp_path, capsys):
    rep = _rep(tmp_path)
    assert cb.main(["--controllo-x", str(rep)]) == 0
    assert capsys.readouterr().out.startswith("x_max 6.0210 ")
    assert cb.main(["--controllo-x", str(rep), "--x-max", "6.0"]) == 1


# --- launch script, dry run ---------------------------------------------------

SCRIPT = ROOT / "scripts" / "prova_banco_serieB_baseline_direction.sh"


def test_sintassi_script():
    assert subprocess.run(["bash", "-n", str(SCRIPT)]).returncode == 0


@pytest.fixture(scope="module")
def a_secco(tmp_path_factory):
    if (ROOT / "data" / "attack" / "verifica" / "PROVA_IN_CORSO.json").exists():
        pytest.skip("prova al banco in corso: niente passo a secco")
    out = tmp_path_factory.mktemp("bd") / "uscita"
    r = subprocess.run(["bash", str(SCRIPT), "--a-secco"], capture_output=True, text=True, timeout=120,
                       cwd=ROOT, env={**os.environ, "OUT": str(out)})
    return r.stdout


def _env_campagna(testo):
    riga = next(l for l in testo.splitlines() if l.startswith("+ env ") and "run_campaign.sh straight none 3 1" in l)
    tok = shlex.split(riga[2:].split(" scripts/run_campaign.sh")[0])[1:]
    return dict(t.split("=", 1) for t in tok)


def test_a_secco_opzioni(a_secco):
    env = _env_campagna(a_secco)
    assert env["WINDOW_STOP"] == "ramp" and env["STOP_RAMP_K"] == "1.0"
    assert env["TRANSPORT"] == "reliable" and env["STRUCTURAL"] == "gross"
    assert env["ICP_QUALITY_LOG"] == "1" and env["SERIE_B"] == "1" and env["SEEDS"] == "1001"
    assert env["VICTIM"] == "continuous" and env["MIN_LATENCY_MS"] == "85"
    nodo = env["NODE_EXTRA"].split()
    assert "--frame-log-delay" in nodo and "--fresh-cloud-after-restore" in nodo and "--frame-log" in nodo
    orch = env["ORCH_EXTRA"].split()
    for a, v in [("--goal-mode", "direction"), ("--max-windows", "6"), ("--extra-windows", "3"),
                 ("--settle-sec", "2.0"), ("--pause-sec", "1800"), ("--settle-clouds", "20"),
                 ("--reopen-clouds", "20")]:
        assert orch[orch.index(a) + 1] == v
    assert "--fresh-cloud-after-restore" in orch
    assert "data/attack/baseline/rettilineo_direction_N6_s1001_dmin85_sosta1800.csv" in a_secco
    assert "--controllo-x" in a_secco


def test_a_secco_riga_dell_orchestratore_accettata(a_secco, monkeypatch, tmp_path):
    """The line run_campaign.sh would build (run_campaign.sh:421-425, none arm, with
    SERIEB_FLAGS --structural gross) passes the orchestrator checks, and --settle-sec from
    ORCH_EXTRA wins over the 5.0 of run_campaign.sh."""
    ao = pytest.importorskip("src.optimization.attack_orchestrator")
    env = _env_campagna(a_secco)
    argv = ["attack_orchestrator.py", "--goals", "5,0", f"--start-pose={env['START_POSE']}", "--horizon", "1.0",
            "--settle-sec", "5.0", "--stop-on", env["STOP_ON"], "--window-stop", "ramp", "--stop-ramp-k",
            env["STOP_RAMP_K"], "--attack-from", env["ATTACK_FROM"], "--drift-reset-on", "genome",
            "--victim", env["VICTIM"], "--select", env["SELECT"], "--topk", env["TOPK"], "--no-warm-start",
            "--seed", env["SEEDS"], "--trace", "--reeval", "--out", str(tmp_path / "rep_01"),
            "--passthrough", "--structural", "gross", *env["ORCH_EXTRA"].split()]
    preso = {}

    class Fermo(Exception):
        pass

    orig = ao.check_goal_mode_args

    def spia(args, follower=None):
        preso["args"], preso["err"] = args, orig(args, follower)
        raise Fermo

    monkeypatch.setattr(ao, "check_goal_mode_args", spia)
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(Fermo):
        ao.main()
    a = preso["args"]
    assert preso["err"] is None
    assert a.goal_mode == "direction" and a.max_windows == 6 and a.extra_windows == 3
    assert a.settle_sec == 2.0 and a.pause_sec == 1800.0 and a.passthrough
    assert a.fresh_cloud_after_restore and a.window_stop == "ramp" and a.stop_ramp_k == 1.0
    assert a.damage_ref == "window" and not a.victim_never_clean
    assert not (tmp_path / "rep_01").exists()
