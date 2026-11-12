"""Victim MOLA time as an attack outcome. With --rallentamento-mola risultato, mola_vittima never
counts as a chain failure (guasto and scoperto unchanged); run_campaign.sh with SERIE_B=1
passes risultato on the attack arms. p50, p99 and max of c = 1000 * time_onLidar of the next
row and ICP restarts from the DEBUG log, per window (history) and per run (queue summary),
also on a recorded run read-only.
"""

import importlib.util
import itertools
import json
import subprocess
import sys
import textwrap
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.optimization import attack_orchestrator as ao  # noqa: E402
from src.optimization import window_record as wrec  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def _modulo(nome, percorso):
    spec = importlib.util.spec_from_file_location(nome, percorso)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


RN = _modulo("_t_regole_notte_rr", ROOT / "tests" / "test_regole_notte.py")
R5 = _modulo("_t_regola5_rr", ROOT / "tests" / "test_regola5_catena.py")
CODA = _modulo("_coda_rr", ROOT / "scripts" / "coda_serieB_stato.py")


# --- chain rule ---

def _guasto_catena_prima(controlli, rallentamento="guasto"):
    """guasto_catena as it was before "risultato" (frozen copy, deciding fields)."""
    if rallentamento not in ("guasto", "scoperto"):
        raise ValueError(rallentamento)
    guasti = [n for n, c in controlli.items() if c.get("esito") == "guasto"]
    motivi = [f"{n}: {controlli[n].get('motivo')}" for n in guasti]
    rall = (guasti == ["mola_vittima"]
            and all(controlli.get(n, {}).get("esito") == "ok" for n in ("isaac", "nuvole_vittima", "perturbation")))
    segnali = list(motivi)
    if rall and rallentamento == "scoperto":
        motivi = []
    return {"guasto": bool(motivi), "motivi": motivi, "segnali": segnali, "rallentamento_mola": rall,
            "non_determinati": [n for n, c in controlli.items() if c.get("esito") == "non_determinato"]}


NOMI = ("isaac", "adattatore", "nuvole_vittima", "posa_vittima", "mola_vittima", "perturbation")
ESITI = ("ok", "guasto", "non_determinato")


def _tutte():
    for es in itertools.product(ESITI, repeat=len(NOMI)):
        yield {n: {"esito": e, "motivo": f"{n} {e}"} for n, e in zip(NOMI, es)}


@pytest.mark.parametrize("regola", ["guasto", "scoperto"])
def test_guasto_e_scoperto_invariati(regola):
    # all 729 outcome combinations of the six checks
    for c in _tutte():
        nuovo = wrec.guasto_catena(c, regola)
        assert {k: nuovo[k] for k in ("guasto", "motivi", "segnali", "rallentamento_mola",
                                      "non_determinati")} == _guasto_catena_prima(c, regola)
        assert nuovo["regola_rallentamento"] == regola


def test_risultato_mola_vittima_mai_guasto():
    for c in _tutte():
        r = wrec.guasto_catena(c, "risultato")
        altri = [n for n in NOMI if n != "mola_vittima" and c[n]["esito"] == "guasto"]
        assert r["guasto"] is bool(altri)
        assert r["motivi"] == [f"{n}: {n} guasto" for n in altri]
        assert r["segnali"] == _guasto_catena_prima(c)["segnali"]       # stays among the signals
        assert r["regola_rallentamento"] == "risultato" and r["controlli"] is c


def test_risultato_caso_della_decisione():
    # mola_vittima failed and adapter undetermined: with risultato it is not a failure
    lento = {"esito": "guasto", "motivo": "time_onLidar p99 250 ms oltre 200 ms (n 50)"}
    c = {"isaac": {"esito": "ok"}, "nuvole_vittima": {"esito": "ok"}, "perturbation": {"esito": "ok"},
         "posa_vittima": {"esito": "ok"}, "adattatore": {"esito": "non_determinato"}, "mola_vittima": lento}
    r = wrec.guasto_catena(c, "risultato")
    assert r["guasto"] is False and r["motivi"] == [] and r["non_determinati"] == ["adattatore"]
    assert r["segnali"] == ["mola_vittima: " + lento["motivo"]]
    # undetermined clouds: "scoperto" stays a failure, "risultato" does not
    nd = {**c, "nuvole_vittima": {"esito": "non_determinato"}}
    assert wrec.guasto_catena(nd, "scoperto")["guasto"] is True
    assert wrec.guasto_catena(nd, "risultato")["guasto"] is False
    # a real chain failure stays a failure, without MOLA time among the reasons
    rotta = wrec.guasto_catena({**c, "isaac": {"esito": "guasto", "motivo": "stallo"}}, "risultato")
    assert rotta["guasto"] is True and rotta["motivi"] == ["isaac: stallo"]
    assert wrec.guasto_catena(c, "guasto")["guasto"] is True


def test_risultato_da_controlla_catena(tmp_path, monkeypatch):
    lento = RN._tracce([0.3] * 20)
    r = RN._catena(tmp_path, monkeypatch, tracce=lento, opz={"rallentamento_mola": "risultato"})
    assert r["guasto"] is False and r["controlli"]["mola_vittima"]["esito"] == "guasto"
    assert r["regola_rallentamento"] == "risultato"
    g = RN._catena(tmp_path, monkeypatch, tracce=lento)
    assert g["guasto"] is True and g["regola_rallentamento"] == "guasto"


def test_risultato_chiude_come_attacco_scoperto(tmp_path):
    c = {**R5.SANI, "mola_vittima": R5.LENTO, "nuvole_vittima": {"esito": "non_determinato"}}
    _, _, h = RN._chiusura(tmp_path, wrec.guasto_catena(c, "risultato"))
    assert h[2]["run_end"]["reason"] == "attacco_scoperto"
    assert not (tmp_path / "da_rifare.json").exists()


def _args_orchestratore(argv):
    codice = textwrap.dedent(f"""
        import json, sys
        from src.optimization import attack_orchestrator as ao
        class Stop(Exception):
            pass
        orig = ao.argparse.ArgumentParser.parse_args
        preso = {{}}
        def parse(self, *a, **k):
            preso["a"] = orig(self, *a, **k)
            raise Stop
        ao.argparse.ArgumentParser.parse_args = parse
        sys.argv = ["attack_orchestrator.py"] + {argv!r}
        try:
            ao.main()
        except Stop:
            pass
        print(json.dumps(vars(preso["a"]), default=str))
    """)
    r = subprocess.run([sys.executable, "-c", codice], cwd=ROOT, capture_output=True, text=True, timeout=120)
    return r.returncode, r.stdout, r.stderr


def test_opzione_orchestratore():
    rc, out, err = _args_orchestratore([])
    assert rc == 0, err[-2000:]
    assert json.loads(out.strip().splitlines()[-1])["rallentamento_mola"] == "guasto"   # default
    rc, out, err = _args_orchestratore(["--rallentamento-mola", "risultato"])
    assert rc == 0 and json.loads(out.strip().splitlines()[-1])["rallentamento_mola"] == "risultato"
    rc, _, err = _args_orchestratore(["--rallentamento-mola", "forse"])
    assert rc != 0 and "invalid choice" in err


# --- run_campaign.sh (fake orchestrator and processes, as in test_regola5_catena) ---

@pytest.mark.parametrize("serie_b,arm,valore,atteso", [
    ("1", "nsga3", "", "risultato"), ("1", "random", "", "risultato"), ("1", "nsga3", "risultato", "risultato"),
    ("1", "nsga3", "guasto", "guasto"), ("1", "nsga3", "scoperto", "scoperto"),
    ("1", "none", "", None), ("1", "none", "risultato", None), ("", "nsga3", "", None)])
def test_campagna_rallentamento_risultato(tmp_path, serie_b, arm, valore, atteso):
    env = {"RALLENTAMENTO_MOLA": valore}
    if serie_b:
        env.update(SERIE_B=serie_b, SEEDS="1001")
    r, righe = R5._campagna(tmp_path, env, arm)
    assert r.returncode == 0, r.stdout[-3000:]
    a = righe[-1]
    got = a[a.index("--rallentamento-mola") + 1] if "--rallentamento-mola" in a else None
    assert got == atteso


# --- times and restarts ---

COL = ["timestamp", "icp_iterations", "icp_quality", "time_onLidar"]


def _righe(valori, t0=20.0):
    """Trace rows with stamp t0 + 0.1 i; time_onLidar of row i = valori[i]."""
    return [{"timestamp": str(round(t0 + 0.1 * i, 1)), "icp_iterations": "5", "icp_quality": "0.8",
             "time_onLidar": str(v)} for i, v in enumerate(valori)]


def test_tempi_c_riga_successiva():
    pc = wrec.tempi_c(_righe([0.0, 0.010, 0.020, 0.030]))
    assert pc == [(20.0, 10.0), (20.1, 20.0), (20.2, 30.0)]
    # broken row (half written): no pair touching it
    r = _righe([0.0, 0.010, 0.020, 0.030])
    r[2]["time_onLidar"] = None
    assert wrec.tempi_c(r) == [(20.0, 10.0)]
    assert wrec.tempi_c(None) == []


def _log(tmp_path, scansioni):
    """Fake DEBUG log: for each (stamp, restarts) the generator and hook rows."""
    L = ["[10:00:00.0000|INFO|LidarOdometry] avvio"]
    for s, n in scansioni:
        L.append(f"[10:00:01.0000|DEBUG|Generator (raw)] Processing observation type='x' label='lidar' stamp={s:.4f}")
        L.append(f"[10:00:01.0001|DEBUG|Generator (obs)] Processing observation type='x' label='lidar' stamp={s:.4f}")
        L += ["[10:00:01.0002|DEBUG|LidarOdometry] ICP hook dt=0.1 deltaTrans=0.2 deltaRot=0.8"] * n
        L.append("[10:00:01.0003|DEBUG|LidarOdometry] Since last KF: updateLocalMap=NO")
    p = tmp_path / "mola_victim.log"
    p.write_text("\n".join(L) + "\n")
    return p


def test_ripartenze_icp(tmp_path):
    p = _log(tmp_path, [(20.0, 0), (20.1, 2), (20.2, 5), (20.1, 7)])    # repeated stamp: once
    assert wrec.ripartenze_icp(p) == {20.0: 0, 20.1: 2, 20.2: 5}
    senza = tmp_path / "info.log"
    senza.write_text("[10:00:00.0000|INFO|LidarOdometry] nessuna riga DEBUG\n")
    assert wrec.ripartenze_icp(senza) is None and wrec.ripartenze_icp(tmp_path / "manca.log") is None


def test_tempi_mola_finestra(tmp_path):
    rows = _righe([0.0] + [0.010] * 9 + [0.100])           # c: 9 x 10 ms, then 100 ms (stamp 20.9)
    rip = wrec.ripartenze_icp(_log(tmp_path, [(20.0 + 0.1 * i, 3 if i == 9 else 0) for i in range(11)]))
    t = wrec.tempi_mola_finestra(rows, None, None, rip)
    assert t["n"] == 10 and t["p50_ms"] == 10.0 and t["max_ms"] == 100.0 and t["p99_ms"] == 91.9
    assert t["ripartenze"] == 3 and t["scansioni_con_ripartenze"] == 1 and t["ripartenze_max"] == 3
    assert t["n_scansioni_debug"] == 11 and t["ripartenze_su_c"] == 3 and t["completo"] is None
    w = wrec.tempi_mola_finestra(rows, 20.5, 20.8, rip)
    assert w["n"] == 4 and w["max_ms"] == 10.0 and w["ripartenze"] == 0 and w["completo"] is True
    # traces not yet written past the window end
    assert wrec.tempi_mola_finestra(rows, 20.5, 21.5, rip)["completo"] is False
    nd = wrec.tempi_mola_finestra(rows, 20.5, 20.8, None)
    assert nd["ripartenze"] is None and nd["n"] == 4
    assert wrec.tempi_mola_finestra(None, 20.5, 20.8)["n"] == 0


def _scrivi(p, rows):
    with open(p, "w") as f:
        f.write(",".join(COL) + ",\n")
        for r in rows:
            f.write(",".join(r[c] for c in COL) + ",\n")


def test_record_e_aggiornamento_per_finestra(tmp_path):
    tr, log = tmp_path / "traces_victim.csv", tmp_path / "mola_victim.log"
    victim = types.SimpleNamespace(traces_path=tr, log_path=log)
    # window 0: reopening at 20.0, traces written up to 20.6
    tutte = _righe([0.0] + [0.010] * 10 + [0.050] * 10, t0=20.0)
    _scrivi(tr, tutte[:7])
    _log(tmp_path, [(round(20.0 + 0.1 * i, 1), 1) for i in range(7)])
    node = types.SimpleNamespace(last_trace=None, last_drive_stamps=(20.2, 20.6), last_victim_open_stamp=20.0)
    w0 = ao._victim_window_record(node, victim, None)
    assert "error" not in w0
    t0 = w0["tempi_mola"]
    assert t0["t0"] == 20.1 and t0["t1"] is None and t0["completo"] is None and t0["n"] == 5
    h = [{"window": 0, "victim_window": w0}]
    ao._refresh_victim_scans(h, victim)                    # end of window 0 not yet known
    assert h[0]["victim_window"]["tempi_mola"]["t1"] is None
    # window 1: reopening at 21.0 (scan 21.0 belongs to window 0), traces up to 22.0
    _scrivi(tr, tutte)
    _log(tmp_path, [(round(20.0 + 0.1 * i, 1), 1) for i in range(21)])
    node.last_drive_stamps, node.last_victim_open_stamp = (21.2, 21.8), 21.0
    h.append({"window": 1, "victim_window": ao._victim_window_record(node, victim, None)})
    ao._refresh_victim_scans(h, victim)
    t0 = h[0]["victim_window"]["tempi_mola"]
    assert (t0["t0"], t0["t1"], t0["n"], t0["completo"]) == (20.1, 21.0, 10, True)
    assert t0["max_ms"] == 50.0 and t0["ripartenze"] == 10      # c of 21.0 is the first 50 ms one
    t1 = h[1]["victim_window"]["tempi_mola"]
    assert t1["t0"] == 21.1 and t1["n"] == 9 and t1["p50_ms"] == 50.0 and t1["ripartenze"] == 10


# --- recorded run (read-only) ---

BREVE_B = (Path(__file__).resolve().parents[1] / "data/attack/verifica/2026-10-27_prova_breve_versioneI_b")


@pytest.mark.skipif(not BREVE_B.exists(), reason="prova breve b assente")
def test_riepilogo_coda_su_prova_registrata():
    # reference values from the timing checks of the recorded run: victim per run n 274 p50 13.2
    # p99 54.4 max 103.7; n 274 p50 12.4 p99 45.4 max 50.5; DEBUG log (victim, short run b):
    # 306 restarts on the matched scans
    attesi = {"s1001": (274, 13.24, 54.38, 103.75), "s2001": (274, 12.39, 45.4, 50.53)}
    su_c = 0
    for s, (n, p50, p99, mx) in attesi.items():
        cart = str(BREVE_B / s)
        rep = CODA.rep_della_run(cart)
        hist = json.load(open(Path(rep) / "history.json"))
        tm = CODA.tempi_mola_run(rep, hist)
        run = tm["run"]
        assert (run["n"], run["p50_ms"], run["p99_ms"], run["max_ms"]) == (n, p50, p99, mx)
        su_c += run["ripartenze_su_c"]
        fin = tm["finestre"]
        assert [f["finestra"] for f in fin] == [0, 1, 2, 3, 4]
        # windows partition the run: only the scan before the first reopening stays out
        assert sum(f["ripartenze"] for f in fin) == run["ripartenze"]
        assert run["n"] - sum(f["n"] for f in fin) == 1
        testo = CODA.testo_riepilogo_run(CODA.riepilogo_run(cart, rep, usa_bag=False))
        assert "tempo di MOLA della vittima" in testo and f"run: n {n} p50 {p50} p99 {p99} max {mx} ms" in testo
    assert su_c == 306


# --- end of run, time errors per window, broken rows ---

def test_fine_run_rende_definitive_tutte_le_finestre(tmp_path):
    tr = tmp_path / "traces_victim.csv"
    victim = types.SimpleNamespace(traces_path=tr, log_path=tmp_path / "mola_victim.log")
    _scrivi(tr, _righe([0.0] + [0.010] * 30, t0=20.0))
    _log(tmp_path, [(round(20.0 + 0.1 * i, 1), 1) for i in range(31)])
    h = []
    for k, (a, d) in enumerate([(20.0, (20.2, 20.8)), (21.0, (21.2, 21.8)), (22.0, (22.2, 22.8))]):
        node = types.SimpleNamespace(last_trace=None, last_drive_stamps=d, last_victim_open_stamp=a)
        h.append({"window": k, "victim_window": ao._victim_window_record(node, victim, None)})
        ao._refresh_victim_scans(h[:-1], victim)               # as in the window loop
    assert h[1]["victim_window"]["tempi_mola"]["t1"] is None   # second to last still open
    assert h[1]["victim_window"]["tempi_mola"]["n"] == 19
    ao._refresh_victim_scans(h, victim, fine_run=True)
    tm = [x["victim_window"]["tempi_mola"] for x in h]
    assert [(t["t0"], t["t1"], t["n"], t["completo"]) for t in tm] == [
        (20.1, 21.0, 10, True), (21.1, 22.0, 10, True), (22.1, None, 9, None)]
    assert tm[2]["fino_a_fine_run"] is True and "fino_a_fine_run" not in tm[1]
    # the end-of-run call is in the finally of main, after the MOLA processes stop
    src = Path(ao.__file__).read_text()
    i = src.index("_refresh_victim_scans(history, victim, fine_run=True)")
    assert 0 < src.rfind("finally:", 0, i) < src.rfind("victim.stop()", 0, i) < i


def test_errore_dei_tempi_isolato_per_finestra(tmp_path, monkeypatch):
    victim = types.SimpleNamespace(traces_path=tmp_path / "t.csv", log_path=tmp_path / "mola_victim.log")
    _log(tmp_path, [(20.0, 0)])

    def rotta(rows, t0=None, t1=None, rip=None):
        raise RuntimeError("tracce illeggibili")
    monkeypatch.setattr(wrec, "tempi_mola_finestra", rotta)
    node = types.SimpleNamespace(last_trace=None, last_drive_stamps=(20.2, 20.6), last_victim_open_stamp=20.0)
    rec = ao._victim_window_record(node, victim, None)
    assert "error" not in rec and rec["tempi_mola"]["errore"] == "tracce illeggibili"
    # window 0 with times to redo (error), window 1 with incomplete scans
    w0 = {"tempi_mola": {"t0": 20.1, "t1": None, "completo": False}, "open_stamp": 20.0}
    w1 = {"scans_drive": {"log_complete": False}, "drive_stamps": [21.2, 21.6], "open_stamp": 21.0}
    h = [{"window": 0, "victim_window": w0}, {"window": 1, "victim_window": w1}]
    ao._refresh_victim_scans(h, victim)
    assert w0["tempi_mola"]["errore"] == "tracce illeggibili"
    assert w1.get("refreshed") is True                       # scans of the next window are updated


def test_riepilogo_della_run_con_errore_dei_tempi(tmp_path, monkeypatch):
    rep = tmp_path / "campagna" / "straight" / "x" / "rep_01"
    rep.mkdir(parents=True)
    (rep / "history.json").write_text("[]")

    def rotta(rep, hist):
        raise ValueError("finto")
    monkeypatch.setattr(CODA, "tempi_mola_run", rotta)
    d = CODA.riepilogo_run(str(tmp_path), usa_bag=False)
    assert d["rep"] == str(rep) and "run_end" in d and "finestre" in d
    assert d["tempi_mola"]["note"] == ["tempi di MOLA non calcolati: ValueError: finto"]
    assert "nota sui tempi di MOLA: tempi di MOLA non calcolati" in CODA.testo_riepilogo_run(d)


def test_ripartenze_con_righe_spezzate_da_tf_old_data(tmp_path):
    warn = ("[WARN] [1793089606.123] []: TF_OLD_DATA ignoring data from the past for frame base_link\n"
            "Possible reasons are listed at http://wiki.ros.org/tf/Errors%20explained\n")
    righe = [
        "[10:00:01.0000|DEBUG|Generator (raw)] Processing observation type='x' label='lidar' stamp=20.0000",
        "[10:00:01.0001|DEBUG|LidarOdometry] ICP hook dt=0.1 deltaTrans=0.2 deltaRot=0.8",
        # broken observation row: without stitching, the next hook would land on scan 20.0
        "[10:00:01.1000|DEBUG|Generator (raw)] Processing observ" + warn + "ation type='x' label='lidar' stamp=20.1000",
        "[10:00:01.1001|DEBUG|LidarOdometry] ICP ho" + warn + "ok dt=0.1 deltaTrans=0.2 deltaRot=0.8",
        "[10:00:01.1002|DEBUG|LidarOdometry] ICP hook dt=0.1 deltaTrans=0.2 deltaRot=0.8",
    ]
    p = tmp_path / "mola_victim.log"
    p.write_text("\n".join(righe) + "\n")
    assert wrec.ripartenze_icp(p) == {20.0: 1, 20.1: 2}


def test_convenzione_dichiarata_nel_campo():
    t = wrec.tempi_mola_finestra(_righe([0.0, 0.01, 0.02]), 20.0, 20.1)
    assert "riga successiva" in t["convenzione"] and "riga stessa" in t["convenzione"]
    assert "convenzione" not in wrec.tempi_mola_vittima(RN._tracce([0.02] * 5))   # 200 ms check
