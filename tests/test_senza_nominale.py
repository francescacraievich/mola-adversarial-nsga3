"""--no-nominal-rollout: no nominal twin rollout. Off, arguments and flow are unchanged; on, no
rollout without a twin genome, damage from the baseline, evaluated candidates and twin
outcome G from the per-rollout checks. Fake objects, no ROS, MOLA or Isaac; window-loop and
eval_genome segments are taken from the source and run as they are.
"""

import json
import math
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
POLY = np.array([[0.0, 0.0], [5.0, 0.0]])
CHK_OK = {"ok": True, "firme_ok": True, "loaded_is_shadow_file": True, "hash_ok": True, "clock_ok": True}
CHK_KO = {"ok": False, "firme_ok": True, "loaded_is_shadow_file": True, "hash_ok": False, "clock_ok": True}


def _args(**kw):
    base = dict(no_nominal_rollout=True, damage_ref="baseline", perceptibility="chamfer",
                f2_budget=None, untracked="discard", settle_clouds=20, victim_never_clean=False,
                g_twin_lost_ends_run=False, passthrough=False, repeat_genome=None, repeat_spec=None,
                map_gate_test=False, victim="continuous", twin_state="clean", twin_align="attacker",
                twin_init="true", drift_reset_on="genome", attack_from="spawn",
                uniform_sim_time="off", replay_gate=0, g_nominale_senza_posa="trattieni")
    base.update(kw)
    return types.SimpleNamespace(**base)


# --- arguments ---

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


@pytest.mark.skipif(not (ORIGINALE / "src" / "optimization" / "attack_orchestrator.py").exists(),
                    reason="copia originale assente")
def test_argomenti_di_default_invariati():
    nuovo = _default_args(ROOT)
    vecchio = _default_args(ORIGINALE)
    assert nuovo.pop("no_nominal_rollout") is False
    # the reference can be a later copy that already has the option: same default;
    # then the options added later, off by default
    assert vecchio.pop("no_nominal_rollout", False) is False
    # and those of the limits study, off by default
    for k, v in {"incompleti_regola": False, "moltiplicatore_limiti": 1.0, "famiglie_limiti": None,
                 "blocca_gene": None}.items():
        if k not in vecchio:
            assert nuovo.pop(k, v) == v
    assert nuovo == vecchio


def test_spenta_nessun_controllo():
    a = _args(no_nominal_rollout=False, damage_ref="window", perceptibility="internal",
              settle_clouds=0, passthrough=True, untracked="nominal")
    assert tv.check_no_nominal_args(a) is None
    assert tv.check_twin_args(a) is None


def test_accesa_combinazione_giusta():
    assert tv.check_no_nominal_args(_args()) is None
    assert tv.check_twin_args(_args()) is None
    # --victim-never-clean implies --settle-clouds 20
    assert tv.check_no_nominal_args(_args(settle_clouds=None, victim_never_clean=True)) is None


@pytest.mark.parametrize("kw,frase", [
    (dict(damage_ref="window"), "--damage-ref window"),
    (dict(perceptibility="internal"), "--perceptibility internal"),
    (dict(f2_budget=0.5), "--f2-budget"),
    (dict(untracked="nominal"), "--untracked nominal"),
    (dict(settle_clouds=0), "--settle-clouds 0"),
    (dict(settle_clouds=None), "--settle-clouds 0"),
    (dict(g_twin_lost_ends_run=True), "--g-twin-lost-ends-run"),
    (dict(passthrough=True), "--passthrough"),
    (dict(repeat_genome="h.json:0"), "--repeat-genome"),
    (dict(repeat_spec="s.json"), "--repeat-spec"),
    (dict(map_gate_test=True), "--map-gate-test"),
    (dict(victim="restart"), "--victim restart"),
])
def test_rifiuto_combinazioni(kw, frase):
    msg = tv.check_twin_args(_args(**kw))
    assert msg and msg.startswith("--no-nominal-rollout non va con") and frase in msg


def test_rifiuto_dal_main(tmp_path):
    """The orchestrator exits with 2 before starting any process."""
    codice = textwrap.dedent(f"""
        import sys
        from src.optimization import attack_orchestrator as ao
        sys.argv = ["attack_orchestrator.py", "--no-nominal-rollout", "--damage-ref", "baseline",
                    "--settle-clouds", "20", "--out", {str(tmp_path / 'run')!r}]
        sys.exit(ao.main())
    """)
    r = subprocess.run([sys.executable, "-c", codice], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 2
    assert "--perceptibility internal" in r.stdout
    assert not (tmp_path / "run").exists()


# --- twin outcome G from the per-rollout checks ---

def test_esito_gemello_g():
    fp_ok = {"source": "ombra_g", "g_check": CHK_OK}
    fp_ko = {"source": "ombra_g", "g_check": CHK_KO}
    assert tv.esito_gemello_g("ombra_g", NO_POSE, fp_ok) == "gemello_perde_posa"
    assert tv.esito_gemello_g("ombra_g", NO_POSE, fp_ko) == "guasto_gemello"
    assert tv.esito_gemello_g("ombra_g", "ok", fp_ok) == "controlli_ok"
    assert tv.esito_gemello_g("ombra_g", "ok", fp_ko) == "guasto_gemello"
    # missing checks (rollout failed before the fingerprint) = failure
    assert tv.esito_gemello_g("ombra_g", "ombra_g_clock_not_armed", None) == "guasto_gemello"
    assert tv.esito_gemello_g("ombra_g", NO_POSE, {"source": "ombra_g"}) == "guasto_gemello"
    # twin other than G: no attribution
    assert tv.esito_gemello_g("clean", NO_POSE, None) is None
    n = tv.conta_esiti_gemello([{"esito_gemello": "guasto_gemello"}, {"esito_gemello": "gemello_perde_posa"},
                                {"esito_gemello": "gemello_perde_posa"}, {}, {"esito_gemello": None}])
    assert n == {"controlli_ok": 0, "gemello_perde_posa": 2, "guasto_gemello": 1}


# --- window-loop segment ---

def _segmento(path):
    """From the nominal (or its absence) up to 'if args.passthrough:' excluded,
    inside a one-pass loop: 'continue' = window skipped.
    """
    src = path.read_text().split("\n")
    i0 = next(i for i, r in enumerate(src) if r.strip().startswith("for attempt in (")
              and "(1, 2)" in r)
    # in the new copy the segment starts from the two initialisation lines
    while src[i0 - 1].strip().startswith(("nom_start = nom_end = None", "st, nom_trav, nom_frames, attempt")):
        i0 -= 1
    i1 = next(i for i in range(i0, len(src)) if src[i].strip() == "if args.passthrough:")
    corpo = textwrap.indent(textwrap.dedent("\n".join(src[i0:i1])), " " * 8)
    return ("def _finestra(args, isaac, mola, node, wps, out, k, frame_start, history, victim,\n"
            "              baseline_poly, frame_verificato, saved, goal, wp_idx, t_win, _esito_finestra):\n"
            "    for _giro in (0,):\n" + corpo + "\n"
            "        return 'prosegue', locals()\n"
            "    return 'saltata', locals()\n")


def _esegui_finestra(path, tmp_path, args, saved=(0.019533, 0.0, 0.0)):
    ns = dict(vars(ao))
    ns["TWIN_STATE"] = {"mode": "clean"}
    chiamate = []

    def rollout(genome, isaac, mola, node, goal, horizon, tag, **kw):
        chiamate.append((genome, tag, kw.get("victim")))
        return None, None, float("inf"), NO_POSE, 0.0, 0
    ns["rollout"] = rollout
    exec(_segmento(path), ns)
    history = []
    out = tmp_path / "run"
    (out / "traces").mkdir(parents=True)
    res = ns["_finestra"](
        args, None, types.SimpleNamespace(log_path=None), types.SimpleNamespace(last_trace=[]),
        [np.array([1.0, 0.0])], out, 4 if not args.no_nominal_rollout else 0, None, history, None,
        POLY, False, saved, np.array([1.0, 0.0]), 0, 0.0, lambda rec, v: rec)
    return res, chiamate, history


def _args_finestra(**kw):
    base = dict(no_nominal_rollout=False, g_nominale_senza_posa="trattieni", twin_state="clean",
                g_twin_lost_ends_run=False, victim_never_clean=False, perceptibility="chamfer",
                settle_clouds=20, horizon=1.0, warmup=0.0, settle_sec=5.0, catchup_sec=0.8,
                attack_from="spawn", stop_on="estimate", drift_reset_on="genome", trace=True,
                damage_ref="baseline", baseline_path="b.csv")
    base.update(kw)
    return types.SimpleNamespace(**base)


@pytest.mark.skipif(not (ORIGINALE / "src" / "optimization" / "attack_orchestrator.py").exists(),
                    reason="copia originale assente")
def test_spenta_flusso_come_prima(tmp_path):
    (tmp_path / "n").mkdir()
    (tmp_path / "o").mkdir()
    (esito_n, _), ch_n, h_n = _esegui_finestra(ORCH, tmp_path / "n", _args_finestra())
    vecchio = ORIGINALE / "src" / "optimization" / "attack_orchestrator.py"
    (esito_o, _), ch_o, h_o = _esegui_finestra(vecchio, tmp_path / "o", _args_finestra())
    # two nominals without genome, then window skipped, as in the original copy
    assert ch_n == ch_o == [(None, "w4_nominal", None), (None, "w4_nominal", None)]
    assert esito_n == esito_o == "saltata"
    for h in (h_n, h_o):
        for r in h:
            r.pop("window_time_s", None)
    assert h_n == h_o and h_n[0]["status"] == f"nominal_failed({NO_POSE})"


def test_accesa_nessun_nominale(tmp_path):
    (esito, loc), chiamate, history = _esegui_finestra(ORCH, tmp_path, _args_finestra(no_nominal_rollout=True))
    assert chiamate == [] and history == [] and esito == "prosegue"
    na = loc["nom_assente"]
    assert na["senza_nominale"] is True and na["window"] == 0
    assert na["riferimenti"]["track_ref"] == tv.TRACK_REF_SENZA_NOMINALE
    # candidate references: constant non-None track_ref, no block, no settle_fixed
    assert loc["nom_ratio"] == tv.TRACK_REF_SENZA_NOMINALE and loc["nominal_ok"] is True
    assert loc["nom_settle_s"] is None and loc["nom_end"] is None and len(loc["nom_path"]) == 0
    assert loc["nom_stats"] == {} and loc["nom_it"] is None and loc["nom_quality"] is None
    # frame check on the saved true pose (first baseline point: 0.0195 m)
    nb = loc["nom_base_fields"]
    assert nb["baseline_frame_ok"] is True and nb["baseline_frame_check_cm"] == pytest.approx(0.0)
    assert nb["baseline_frame_check_from"] == "posa_vera_salvata" and loc["frame_verificato"] is True


def test_accesa_frame_sbagliato(tmp_path):
    res, chiamate, history = _esegui_finestra(
        ORCH, tmp_path, _args_finestra(no_nominal_rollout=True), saved=(0.0, 0.5, 0.0))
    # 'return 2' inside the segment: the function returns 2, the run closes
    assert res == 2
    assert chiamate == []
    assert history[0]["status"] == "baseline_frame_mismatch"
    assert history[0]["baseline_frame_check_cm"] == pytest.approx(50.0)
    assert history[0]["nominal_end"] is None


# --- eval_genome with fake objects ---

def _eval_genome_src():
    src = ORCH.read_text().split("\n")
    i0 = next(i for i, r in enumerate(src) if r.strip() == "def eval_genome(g_active):")
    i1 = next(i for i in range(i0, len(src)) if src[i].strip() == "n_warm = 0")
    return textwrap.dedent("\n".join(src[i0:i1]))


def _esegui_candidati(casi, opzione, mode="ombra_g"):
    """casi: list of (status, end, g_check or None, true path). Returns
    objectives, eval_log and rollout calls.
    """
    ns = dict(vars(ao))
    chiamate = []
    node = types.SimpleNamespace(
        last_loc_err_mean=float("nan"), last_loc_err_end=float("nan"), last_loc_err_mean_raw=float("nan"),
        last_track_ratio=0.95, last_perc=None, last_quality_mean=0.9, last_quality_min=0.8,
        last_stop_reason="h", last_tau=0.1, proc_ms_roll=[], last_replay=None, last_trace=[],
        last_twin_fp=None, last_drive_stamps=(float("nan"), float("nan")))

    def rollout(genome, isaac, mola, node_, goal, horizon, tag, **kw):
        st, end, chk, path = casi[len(chiamate)]
        chiamate.append((genome, tag, kw))
        node.last_twin_fp = None if chk is None else {"source": "ombra_g", "g_check": chk}
        node.last_trace = path
        return (0.0, 0.0, 0.0), end, 2.0, st, 1.0 if end else 0.0, 10 if end else 0
    args = types.SimpleNamespace(
        pop=4, horizon=1.0, warmup=0.0, settle_sec=5.0, catchup_sec=0.8, attack_from="spawn",
        stop_on="estimate", untracked="discard", drift_reset_on="genome", trace=False,
        discard_policy="infra", no_nominal_rollout=opzione, g_nominale_senza_posa="trattieni",
        twin_filter="none", damage="mean", dir_weight=0.0, admissibility="none", chamfer_budget=None,
        max_proc_ms=None, f2_q_scale=0.0706, f2_it_scale=0.84, perceptibility="chamfer", f2_budget=None,
        verbose=False, objective="damage")
    ns.update(
        rollout=rollout, TWIN_STATE={"mode": mode}, win_cap=None, t_win=0.0, eval_log=[],
        active_genes=list(range(3)), n_genes=3, expand_genome=lambda g, a, n: np.asarray(g, float),
        isaac=None, mola=types.SimpleNamespace(), node=node, wps=[np.array([1.0, 0.0])], k=2,
        nominal_ok=True, nom_ratio=tv.TRACK_REF_SENZA_NOMINALE if opzione else 0.9, nom_settle_s=None,
        frame_start=None, out=Path("/nonesiste"), nom_end=None, goal=np.array([1.0, 0.0]), nom_path=[],
        baseline_poly=POLY, nom_stats={}, nom_quality=None, nom_it=None,
        nom_assente=({"window": 2, "senza_nominale": True} if opzione else None),
        saved=(0.0, 0.0, 0.0), args=args,
        drift_fields=lambda n: {}, _twin_flag_fields=lambda n, m: {},
        _twin_start_fields=lambda n: {"twin_start": n.last_twin_fp},
        _mola_csv_stats=lambda *a: {}, _gap0_from_trace=lambda t: None,
        _twin_misaligned=lambda n, a, g: (False, None))
    exec(_eval_genome_src(), ns)
    F = [ns["eval_genome"](np.zeros(3)) for _ in casi]
    return F, ns["eval_log"], chiamate


PATH_OK = np.array([[0.0, 0.1], [0.5, 0.2], [1.0, 0.3]])
CASI = [("ok", (1.0, 0.3, 0.0), CHK_OK, PATH_OK),          # valid, correct checks
        (NO_POSE, None, CHK_OK, []),                        # the twin loses the pose
        (NO_POSE, None, CHK_KO, []),                        # twin failure
        ("ok", (1.0, 0.3, 0.0), CHK_KO, PATH_OK)]           # pose but checks failed


def test_candidati_accesa():
    F, log, chiamate = _esegui_candidati(CASI, True)
    # no rollout without genome; candidate references
    assert all(g is not None for g, _, _ in chiamate)
    assert [t for _, t, _ in chiamate] == ["w2_e0", "w2_e1", "w2_e2", "w2_e3"]
    assert all(kw["track_ref"] == 1.0 and kw["settle_fixed"] is None and kw["nominal_ok"] is True
               for _, _, kw in chiamate)
    # damage from the baseline: mean distance from the polyline (0.1, 0.2, 0.3 m)
    assert log[0]["counted"] is True and log[0]["damage"] == pytest.approx(0.2)
    assert F[0][0] == pytest.approx(-0.2) and F[0][1] == pytest.approx(2.0)
    assert log[0]["esito_gemello"] == "controlli_ok" and "dev_mean_window" not in log[0]
    assert log[0]["f2_int"] is None
    # loses the pose: excluded, MOLA outcome, not a failure
    assert log[1]["counted"] is False and log[1]["esito_gemello"] == "gemello_perde_posa"
    assert "guasto_gemello" not in log[1]
    # failure: checks failed, with or without pose
    for e in (log[2], log[3]):
        assert e["counted"] is False and e["esito_gemello"] == "guasto_gemello" and e["guasto_gemello"] is True
    assert F[1][0] == F[2][0] == F[3][0] == np.inf
    assert tv.conta_esiti_gemello(log) == {"controlli_ok": 1, "gemello_perde_posa": 1, "guasto_gemello": 2}


def test_candidati_accesa_gemello_non_g():
    # windows before the shadow starts: twin from scratch, no attribution
    F, log, _ = _esegui_candidati([(NO_POSE, None, None, []), ("ok", (1.0, 0.3, 0.0), None, PATH_OK)],
                                  True, mode="clean")
    assert "esito_gemello" not in log[0] and "guasto_gemello" not in log[0] and log[0]["counted"] is False
    assert "esito_gemello" not in log[1] and log[1]["counted"] is True


def test_candidati_spenta_come_prima():
    F, log, chiamate = _esegui_candidati(CASI, False)
    assert all("esito_gemello" not in e and "guasto_gemello" not in e for e in log)
    # without the option the checks do not exclude: the fourth candidate counts as usual
    assert [e["counted"] for e in log] == [True, False, False, True]
    assert all(kw["track_ref"] == 0.9 for _, _, kw in chiamate)


# --- run_campaign.sh ---

def _riga_rc():
    righe = [r for r in (ROOT / "scripts" / "run_campaign.sh").read_text().split("\n")
             if "--no-nominal-rollout" in r]
    assert len(righe) == 1
    return righe[0]


@pytest.mark.parametrize("serie_b,arm,atteso", [("1", "nsga3", True), ("1", "random", True),
                                                 ("1", "none", False), ("", "nsga3", False),
                                                 ("0", "random", False)])
def test_run_campaign_riga(serie_b, arm, atteso):
    script = f'set -e; SERIE_B="{serie_b}"; ARM="{arm}"; SERIEB_FLAGS=" --x"\n{_riga_rc()}\necho "[$SERIEB_FLAGS]"'
    r = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
    assert r.returncode == 0
    assert r.stdout.strip() == ("[ --x --no-nominal-rollout]" if atteso else "[ --x]")


@pytest.mark.parametrize("gim,atteso", [("", "[ --x --no-nominal-rollout]"), ("2.0", "[ --x --no-nominal-rollout]")])
def test_run_campaign_riga_con_genoma_in_moto(gim, atteso):
    """The --no-nominal-rollout line does not depend on GENOMA_IN_MOTO (D31_FLAGS, RAMPA_NODO)."""
    script = (f'set -e; SERIE_B="1"; ARM="nsga3"; GENOMA_IN_MOTO="{gim}"; SERIEB_FLAGS=" --x"\n{_riga_rc()}\n'
              'echo "[$SERIEB_FLAGS]"')
    r = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
    assert r.returncode == 0 and r.stdout.strip() == atteso
    testo = (ROOT / "scripts" / "run_campaign.sh").read_text()
    assert 'D31_FLAGS="$D31_FLAGS --genoma-in-moto $GENOMA_IN_MOTO"; RAMPA_NODO=" --genoma-in-moto $GENOMA_IN_MOTO"' \
        in testo


# --- analisi_pilota: [4] informative on victim and attacker ---

def test_analisi_pilota_eq_vittima_attaccante(capsys):
    sys.path.insert(0, str(ROOT / "scripts"))
    import analisi_pilota as ap
    righe = []
    for run in ("s1", "s2"):
        for w in range(3):
            for i in range(20):
                righe.append({"_run": run, "_finestra": w, "delta_victim_ms": str(0.5 + 0.01 * i),
                              "delta_attacker_ms": str(0.6 + 0.01 * i)})
    righe.append({"_run": "s1", "_finestra": 0, "delta_main_ms": "0.4"})   # twin: not included
    assert ap.eq_vittima_attaccante(righe) is True
    out = capsys.readouterr().out
    assert "coppie per nuvola 120" in out and "fuori dall'ESITO" in out
    # much higher victim delay: outside the margins, but only informative
    for r in righe:
        if "delta_victim_ms" in r:
            r["delta_victim_ms"] = str(float(r["delta_victim_ms"]) + 3.0)
    assert ap.eq_vittima_attaccante(righe) is False
    assert ap.eq_vittima_attaccante([{"delta_main_ms": "1"}]) is None


def test_analisi_pilota_opzione_spenta_di_default():
    sys.path.insert(0, str(ROOT / "scripts"))
    import analisi_pilota as ap
    src = (ROOT / "scripts" / "analisi_pilota.py").read_text()
    assert 'ap.add_argument("--eq-vittima-attaccante", action="store_true"' in src
    # the informative block touches neither ok[4] nor the outcome
    i = src.index("def eq_vittima_attaccante")
    j = src.index("def main()")
    assert "ok[" not in src[i:j]
    assert "if a.eq_vittima_attaccante:\n        eq_vittima_attaccante(righe_tot)" in src
