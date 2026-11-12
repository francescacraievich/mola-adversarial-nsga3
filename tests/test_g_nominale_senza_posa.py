"""--g-nominale-senza-posa: nominal of twin G that does not localize after the state load.
Off (trattieni) the flow is unchanged; on (continua) the window goes on with the
candidates and the outcome stays in the history. Fake objects, no ROS, MOLA or Isaac; the
window-loop segment is taken from the source and run as is."""

import json
import subprocess
import sys
import textwrap
import types
from pathlib import Path

import numpy as np
import pytest

from src.optimization import attack_orchestrator as ao
from src.optimization import twin_variants as tv

ROOT = Path(__file__).resolve().parents[1]
ORCH = ROOT / "src" / "optimization" / "attack_orchestrator.py"
ORIGINALE = ROOT.parent / "originale"
NO_POSE = "no_pose_from_new_mola (publisher nel grafo: 0, processi vivi: 4)"
LOG = (
    "[LidarOdometry] pathStep=229 timestamp=1970/01/01,00:00:31.800000 goodness=0.950 minRequired=0.200 "
    "iters=3 termReason=X isGood=1 adapt_thres_sigma=0.0400 consecutive_bad=0\n"
    "[04:10:00.0950|INFO |LidarOdometry] [full_state] caricato /x/ombra_w4.molastate (6654775 byte) [l]\n"
    "[LidarOdometry] pathStep=230 timestamp=1970/01/01,00:00:31.900000 goodness=0.000 minRequired=0.200 "
    "iters=7 termReason=A isGood=0 adapt_thres_sigma=0.0400 consecutive_bad=0\n"
    "[LidarOdometry] pathStep=230 timestamp=1970/01/01,00:00:32.000000 goodness=0.120 minRequired=0.200 "
    "iters=6 termReason=A isGood=0 adapt_thres_sigma=0.0400 consecutive_bad=1\n")


def _args_twin(**kw):
    base = dict(twin_state="ombra_g", victim="continuous", twin_align="attacker",
                uniform_sim_time="true", twin_init="true", drift_reset_on="genome",
                attack_from="spawn", repeat_genome=None, repeat_spec=None,
                map_gate_test=False, replay_gate=0, g_overlay=None,
                damage_ref="baseline", g_twin_lost_ends_run=False, passthrough=False)
    base.update(kw)
    return types.SimpleNamespace(**base)


def _overlay(tmp_path):
    root = tmp_path / "install_G1"
    for p in tv.G_OVERLAY_PKGS:
        (root / p / "lib" / "x86_64-linux-gnu").mkdir(parents=True)
    return str(root)


def _default_args(modulo_dir):
    """Default orchestrator arguments of a copy, in a separate process."""
    codice = textwrap.dedent("""
        import json, sys
        from src.optimization import attack_orchestrator as ao
        class Stop(Exception):
            pass
        orig = ao.argparse.ArgumentParser.parse_args
        preso = {}
        def parse(self, *a, **k):
            preso["a"] = orig(self, *a, **k)
            raise Stop
        ao.argparse.ArgumentParser.parse_args = parse
        sys.argv = ["attack_orchestrator.py"]
        try:
            ao.main()
        except Stop:
            pass
        print(json.dumps(vars(preso["a"]), default=str, sort_keys=True))
    """)
    r = subprocess.run([sys.executable, "-c", codice], cwd=modulo_dir, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    return json.loads(r.stdout.strip().splitlines()[-1])


# --- option off: arguments and flow unchanged ---------------------------------------

@pytest.mark.skipif(not (ORIGINALE / "src" / "optimization" / "attack_orchestrator.py").exists(),
                    reason="copia originale assente")
def test_argomenti_di_default_invariati():
    nuovo = _default_args(ROOT)
    vecchio = _default_args(ORIGINALE)
    assert nuovo.pop("g_nominale_senza_posa") == "trattieni"
    # the reference may be a later copy that already has the option: same default; then
    # the options added later, off by default
    assert vecchio.pop("g_nominale_senza_posa", "trattieni") == "trattieni"
    # and the limit-study options, off by default
    for k, v in {"incompleti_regola": False, "moltiplicatore_limiti": 1.0, "famiglie_limiti": None,
                 "blocca_gene": None}.items():
        if k not in vecchio:
            assert nuovo.pop(k, v) == v
    assert nuovo == vecchio


def test_helper_spento_non_fa_nulla(tmp_path):
    log = tmp_path / "mola_w4_nominal.log"
    log.write_text(LOG)
    for opz in ({}, {"g_nominale_senza_posa": "trattieni"}):
        a = _args_twin(**opz)
        assert tv.g_nominale_senza_posa(a, NO_POSE, None, str(log), 4, 2, 1.0) is None


def test_controllo_opzione(tmp_path):
    ov = _overlay(tmp_path)
    assert tv.check_twin_args(_args_twin(g_overlay=ov, g_nominale_senza_posa="continua")) is None
    assert tv.check_twin_args(_args_twin(g_overlay=ov, g_nominale_senza_posa="trattieni",
                                         damage_ref="window")) is None
    casi = [(dict(twin_state="ombra_e"), "richiede --twin-state ombra_g"),
            (dict(damage_ref="window"), "richiede --damage-ref baseline"),
            (dict(g_twin_lost_ends_run=True), "non va con --g-twin-lost-ends-run"),
            (dict(passthrough=True), "non va con --passthrough")]
    for kw, frase in casi:
        msg = tv.check_twin_args(_args_twin(g_overlay=ov, g_nominale_senza_posa="continua", **kw))
        assert msg and frase in msg


# --- option on: history entry -------------------------------------------------------

def test_helper_acceso(tmp_path):
    log = tmp_path / "mola_w4_nominal.log"
    log.write_text(LOG)
    a = _args_twin(g_nominale_senza_posa="continua", perceptibility="chamfer", settle_clouds=20)
    rec = tv.g_nominale_senza_posa(a, NO_POSE, None, str(log), 4, 2, 123.5)
    assert rec["window"] == 4 and rec["t_wall"] == 123.5 and rec["status"] == NO_POSE
    # only the ICPs after the load
    assert rec["q_n"] == 2 and rec["q_good"] == 0 and rec["q_max"] == 0.12 and rec["q_primi"] == [0.0, 0.12]
    assert rec["stamp_primo"] == "1970/01/01,00:00:31.900000"
    assert rec["mola_lines"]["n_loaded"] == 1
    rif = rec["riferimenti_non_disponibili"]
    assert rif["f2_interna"] == "non usata" and rif["settle_fixed"] == "non usato (--settle-clouds)"
    a.perceptibility = "internal"
    assert tv.g_nominale_senza_posa(a, NO_POSE, None, str(log), 4, 2)["riferimenti_non_disponibili"][
        "f2_interna"] == "non definita"
    # nominal succeeded, state not loaded or other failure: usual flow
    assert tv.g_nominale_senza_posa(a, "ok", (1.0, 0.0, 0.0), str(log), 4, 2) is None
    log2 = tmp_path / "senza_caricamento.log"
    log2.write_text(LOG.split("\n", 2)[0] + "\n")
    assert tv.g_nominale_senza_posa(a, NO_POSE, None, str(log2), 4, 2) is None
    assert tv.g_nominale_senza_posa(a, "mola_not_ready: x", None, str(log), 4, 2) is None
    assert tv.g_nominale_senza_posa(a, NO_POSE, None, str(tmp_path / "manca.log"), 4, 2) is None


# --- window-loop segment with fake objects ------------------------------------------

def _segmento():
    """From the computation of nom_assente up to 'if args.passthrough:' (excluded), inside a
    one-pass loop: 'continue' or 'break' = window not continued."""
    src = ORCH.read_text().split("\n")
    i0 = next(i for i, r in enumerate(src) if "nom_assente = tv.g_nominale_senza_posa(" in r)
    i1 = next(i for i in range(i0, len(src)) if src[i].strip() == "if args.passthrough:")
    corpo = textwrap.indent(textwrap.dedent("\n".join(src[i0:i1])), " " * 8)
    return ("def _finestra(args, st, nom_end, nom_start, nom_trav, nom_frames, attempt, mola, node, k, goal,\n"
            "              wp_idx, t_win, out, history, victim, baseline_poly, frame_verificato, saved,\n"
            "              _esito_finestra):\n"
            "    for _giro in (0,):\n" + corpo + "\n"
            "        return 'prosegue', locals()\n"
            "    return 'saltata', locals()\n")


def _esito_finto(k_corrente, nom_assente_rif):
    # same lines added to _esito_finestra
    def _esito(rec, v_status):
        nom_assente = nom_assente_rif[0]
        if nom_assente is not None and nom_assente.get("window") == k_corrente:
            rec["nominale_non_localizza"] = nom_assente
        return rec
    return _esito


def _esegui(tmp_path, opzione):
    ns = dict(vars(ao))
    exec(_segmento(), ns)
    log = tmp_path / "mola_w4_nominal.log"
    log.write_text(LOG)
    args = types.SimpleNamespace(g_nominale_senza_posa=opzione, twin_state="ombra_g",
                                 g_twin_lost_ends_run=False, victim_never_clean=False,
                                 perceptibility="internal", settle_clouds=20)
    rif = [None]
    history = []
    esito, loc = ns["_finestra"](
        args, NO_POSE, None, None, 0.0, 0, 2, types.SimpleNamespace(log_path=str(log)),
        types.SimpleNamespace(last_trace=[]), 4, np.array([1.0, 0.0]), 0, 0.0, tmp_path, history,
        None, np.array([[0.0, 0.0], [5.0, 0.0]]), True, (0.0, 0.0, 0.0), _esito_finto(4, rif))
    rif[0] = loc.get("nom_assente")
    return esito, loc, history, _esito_finto(4, rif)


def test_segmento_spento_finestra_saltata(tmp_path):
    esito, loc, history, _ = _esegui(tmp_path, "trattieni")
    assert esito == "saltata" and loc["nom_assente"] is None
    assert len(history) == 1 and history[0]["status"] == f"nominal_failed({NO_POSE})"
    assert "nominale_non_localizza" not in history[0]
    assert json.loads((tmp_path / "history.json").read_text())[0]["window"] == 4


def test_segmento_acceso_finestra_prosegue(tmp_path):
    esito, loc, history, esito_finestra = _esegui(tmp_path, "continua")
    assert esito == "prosegue" and history == []
    na = loc["nom_assente"]
    assert na["window"] == 4 and na["q_n"] == 2 and na["t_wall"] is not None
    # nominal references: missing, no candidate block
    # nom_ratio NaN, untracked check active
    assert loc["nominal_ok"] is True and loc["nom_ratio"] != loc["nom_ratio"]
    assert loc["nom_quality"] is None and loc["nom_it"] is None and loc["nom_stats"] == {}
    assert loc["nom_settle_s"] is None and loc["nom_path"] == [] and loc["nom_base_fields"] == {}
    # the entry reaches the window record
    rec = esito_finestra({"window": 4}, None)
    assert rec["nominale_non_localizza"] is na
    # next window (k = 5): the entry from window 4 is not repeated
    assert "nominale_non_localizza" not in _esito_finto(5, [na])({"window": 5}, None)


def test_danno_dalla_baseline_senza_nominale():
    poly = np.array([[0.0, 0.0], [5.0, 0.0]])
    path = np.array([[0.0, 0.1], [0.5, 0.2], [1.0, 0.3]])
    dm = ao.window_damage((1.0, 0.3, 0.0), None, (1.0, 0.0), None, path, "mean", 0.0, baseline=poly)
    assert dm["damage"] == pytest.approx(0.2) and "damage_window" not in dm
    con = ao.window_damage((1.0, 0.3, 0.0), (1.0, 0.0, 0.0), (1.0, 0.0), np.array([[0, 0], [1.0, 0.0]]),
                           path, "mean", 0.0, baseline=poly)
    assert con["damage"] == dm["damage"]   # the nominal does not enter the damage vs baseline


def test_sorgente_candidati():
    src = ORCH.read_text()
    # internal f2 undefined without nominal, objective 0; with the option off f2_int stays a float
    assert "f2_int = f2_q = f2_it = None" in src
    assert '(f2_int if f2_int is not None else 0.0) if args.perceptibility == "internal" else pert' in src
    # candidate without pose: excluded as before, labelled only with the option on
    i = src.index("guasto_g = (args.g_nominale_senza_posa == \"continua\"")
    assert 'return (np.inf, np.inf, 1e6)' in src[i:i + 2500]
    # hold branch unchanged, only conditioned on nom_assente None
    assert 'if nom_assente is None and (nom_end is None or st not in ("ok", "arrived")):' in src
