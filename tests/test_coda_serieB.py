#!/usr/bin/env python3
"""
Python part of the campaign queue (scripts/coda_serieB_stato.py) and dry run of
scripts/coda_serieB.sh. Synthetic data, no simulation.

    python3 -m pytest tests/test_coda_serieB.py -q
"""

import importlib.util
import json
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("coda", ROOT / "scripts" / "coda_serieB_stato.py")
coda = importlib.util.module_from_spec(spec)
spec.loader.exec_module(coda)


def test_condizioni_D5():
    ids = [c["id"] for c in coda.CONDIZIONI]
    # 20 conditions, including casuale_senzabudget and randpuro_senzabudget (test_genoma_nullo.py)
    assert len(ids) == 20 and ids[0] == "baseline" and ids[-1] == "massimo"
    assert coda.espandi(["tutte"]) == ids


def test_espandi_ordine_canonico_e_modelli():
    assert coda.espandi(["massimo", "nsga3_eps1*", "baseline"]) == ["baseline", "nsga3_eps10", "massimo"]
    try:
        coda.espandi(["nsga4"])
    except ValueError:
        pass
    else:
        raise AssertionError("id sconosciuto accettato")


def test_blocchi_per_seme():
    r = coda.ordina(["1001", "2001"], ["baseline", "massimo"])
    assert r == [("1001", "baseline"), ("1001", "massimo"), ("2001", "baseline"), ("2001", "massimo")]


def test_ripetizioni_e_ripresa(tmp_path):
    st = str(tmp_path / "stato.json")
    runs = [("1001", "nsga3_eps10"), ("1001", "massimo")]
    assert coda.prossima(coda.leggi(st), runs) == ("1001", "nsga3_eps10")
    for i in range(3):
        coda.inizia(st, "nsga3_eps10", "1001", f"c{i}")
        s = coda.chiudi(st, "nsga3_eps10", "1001", {"esito": "guasto_catena", "motivo": "x"})
        assert s == ("da_rifare" if i < 2 else "guasto_definitivo")
    assert coda.prossima(coda.leggi(st), runs) == ("1001", "massimo")
    coda.inizia(st, "massimo", "1001", "m0")
    assert coda.ripara(st) == ["s1001/massimo m0"]   # queue died: the attempt becomes interrupted
    assert coda.prossima(coda.leggi(st), runs) == ("1001", "massimo")
    coda.inizia(st, "massimo", "1001", "m1")
    assert coda.chiudi(st, "massimo", "1001", {"esito": "fatta", "motivo": "completa"}) == "fatta"
    assert coda.prossima(coda.leggi(st), runs) is None


def test_cartella_mai_sovrascritta(tmp_path):
    assert coda.cartella_libera(str(tmp_path), "massimo", "1001").endswith("massimo/s1001")
    (tmp_path / "massimo" / "s1001").mkdir(parents=True)
    (tmp_path / "massimo" / "s1001_rip1").mkdir()
    assert coda.cartella_libera(str(tmp_path), "massimo", "1001").endswith("massimo/s1001_rip2")


def _rep(tmp_path, records, completata=True):
    rep = tmp_path / "rep_01"
    rep.mkdir()
    (rep / "history.json").write_text(json.dumps(records))
    if completata:
        (rep / "completata").write_text("x")
    return rep


def test_storia_run_end(tmp_path):
    rep = _rep(tmp_path, [{"window": 0}, {"window": 1, "run_end": {"reason": "attacco_scoperto"}}])
    s = coda.storia(str(rep))
    assert s["attacco_scoperto"] and not s["guasto_catena"] and s["finestre"] == 2


def test_classifica_ordine():
    base = {"rc": 0, "completata": True, "adattatore_vivo": True, "storia": {}, "ombra": {"esito": "identica"}}
    assert coda.classifica(base)["esito"] == "fatta"
    assert coda.classifica(base)["ritardo"] == "non_controllato"
    # controllo_ritardo_run.py codes: 0 compliant, 3 rit4, 4 configuration, 5 not measurable
    assert coda.classifica({**base, "ritardo_rc": 3})["esito"] == "oltre_ritardo"
    assert coda.classifica({**base, "ritardo_rc": 0})["ritardo"] == "entro_limite"
    assert coda.classifica({**base, "ritardo_rc": 4})["esito"] == "ritardo_configurazione"
    assert coda.classifica({**base, "ritardo_rc": 5})["esito"] == "ritardo_non_misurabile"
    assert coda.classifica({**base, "ritardo_rc": 1})["esito"] == "ritardo_errore"
    assert coda.classifica({**base, "ritardo_rc": 124})["esito"] == "ritardo_errore"
    assert coda.classifica({**base, "ritardo_rc": 3, "baseline": True})["esito"] == "fatta"
    # chain failure takes precedence over attack discovered
    sc = {**base, "storia": {"attacco_scoperto": True, "guasto_catena": True, "run_end": ["guasto_catena"]}}
    assert coda.classifica(sc)["esito"] == "guasto_catena"
    assert coda.classifica({**base, "storia": {"attacco_scoperto": True}})["motivo"] == "attacco scoperto"
    assert coda.classifica({**base, "rc": 1})["esito"] == "guasto_catena"
    assert coda.classifica({**base, "watchdog_interventi": 1})["esito"] == "guasto_catena"
    assert coda.classifica({**base, "tempo_scaduto": True, "segnale": "TERM"})["esito"] == "guasto_catena"
    assert coda.classifica({**base, "segnale": "INT"})["esito"] == "interrotta"
    assert coda.classifica({**base, "motivo_interruzione": "x odom vera oltre 7.0 m (..)"})["esito"] == "fuori_spazio"
    assert coda.classifica({**base, "configurazione": "manca"})["esito"] == "configurazione"
    assert coda.classifica({**base, "baseline": True, "baseline_rc": 1})["esito"] == "guasto_catena"


def test_guasto_catena_codice_4_e_da_rifare(tmp_path):
    base = {"rc": 0, "completata": True, "adattatore_vivo": True, "storia": {}}
    # code 4 of run_campaign.sh, even without history
    e = coda.classifica({**base, "rc": 4, "completata": False})
    assert e["esito"] == "guasto_catena" and "uscito con 4" in e["motivo"]
    # da_rifare.json in the rep: failure even if the history says attack discovered
    rep = _rep(tmp_path, [{"window": 0, "run_end": {"reason": "attacco_scoperto"}}], completata=False)
    (rep / "da_rifare.json").write_text(json.dumps({"reason": "guasto_catena", "motivi": ["nuvole ferme 20 s"]}))
    s = coda.storia(str(rep))
    assert s["guasto_catena"] and s["da_rifare"]["motivi"] == ["nuvole ferme 20 s"]
    e = coda.classifica({**base, "rc": 4, "completata": False, "storia": s})
    assert e["esito"] == "guasto_catena" and "nuvole ferme 20 s" in e["motivo"]


def test_ritardo_rit4_ripetuto_4_e_5_fermano(tmp_path):
    st = str(tmp_path / "stato.json")
    coda.inizia(st, "nsga3_eps10", "1001", "c0")
    assert coda.chiudi(st, "nsga3_eps10", "1001", {"esito": "oltre_ritardo"}) == "da_rifare"
    coda.inizia(st, "nsga3_eps10", "1001", "c1")
    assert coda.chiudi(st, "nsga3_eps10", "1001", {"esito": "guasto_catena"}) == "da_rifare"
    coda.inizia(st, "nsga3_eps10", "1001", "c2")
    assert coda.chiudi(st, "nsga3_eps10", "1001", {"esito": "oltre_ritardo"}) == "guasto_definitivo"
    for es in ("ritardo_configurazione", "ritardo_non_misurabile", "ritardo_errore"):
        assert es in coda.ESITI_FERMANO_CODA and es not in coda.ESITI_RIPETIBILI
    # after the stop the run is still to redo (new folder on relaunch)
    coda.inizia(st, "massimo", "1001", "m0")
    assert coda.chiudi(st, "massimo", "1001", {"esito": "ritardo_non_misurabile"}) == "da_rifare"


def _printf(stamp, buona, cb=0):
    s = f"{stamp:04.1f}"
    return (f"[LidarOdometry] pathStep={int(stamp * 10)} timestamp=1970/01/01,00:00:{s} goodness="
            f"{0.8 if buona else 0.2:.6f} minRequired=0.400000 iters=5 termReason=IterTermReason::Stalled "
            f"isGood={1 if buona else 0} adapt_thres_sigma=0.100000 consecutive_bad={cb}\n")


def _run_sintetica(tmp_path):
    """Fake run: victim printf 10.0-13.0 s, frame log (wall = 1000 + stamp), applied traces
    of w0 (moving, stopped at the last row) and w1 (stopped), history with per-meter damage."""
    run = tmp_path / "s1001"
    rep = run / "campagna" / "straight" / "x" / "rep_01"
    (rep / "logs").mkdir(parents=True)
    (rep / "traces").mkdir()
    cattive = {round(x / 10, 1) for x in list(range(105, 108)) + list(range(115, 129)) + [130]}
    stamps = [round(x / 10, 1) for x in range(100, 131)]
    (rep / "logs" / "mola_victim.log").write_text("".join(_printf(s, s not in cattive) for s in stamps))
    righe = ["t_arr_wall,stamp_victim"] + [f"{1000 + s:.1f},{s}" for s in stamps]
    (run / "frame_log.csv").write_text("\n".join(righe) + "\n")
    w0 = ["wall,true_vx,true_vy"] + [f"{1000 + x / 10:.1f},0.5,0.0" for x in range(100, 110)] + ["1011.0,0.01,0.0"]
    w1 = ["wall,true_vx,true_vy"] + [f"{1000 + x / 10:.1f},0.0,0.0" for x in range(126, 131)]
    (rep / "traces" / "w0_applied.csv").write_text("\n".join(w0) + "\n")
    (rep / "traces" / "w1_applied.csv").write_text("\n".join(w1) + "\n")
    h = [{"window": 0, "chamfer_vincitore_cm": 4.2, "chamfer_applicato_cm": 4.0, "chamfer_budget": 5.0,
          "danno_metro_vittima": {"mean_cm": 3.0, "end_cm": 5.0}, "pause_start_wall": 1011.1, "pause_end_wall": 1011.3},
         {"window": 1, "chamfer_vincitore_cm": 4.9, "chamfer_applicato_cm": 4.7, "chamfer_budget": 5.0,
          "danno_metro_vittima": {"mean_cm": 6.0, "end_cm": 7.0},
          "run_end": {"reason": "finestre_completate", "windows_driven": 2, "windows_held": 0},
          "danno_run": {"victim_end_cm": 7.0, "victim_end_window": 1, "victim_mean_cm": 4.5, "victim_max_cm": 7.5,
                        "winner_mean_cm": 4.0, "n_windows": 2}}]
    (rep / "history.json").write_text(json.dumps(h))
    (rep / "completata").write_text("x")
    (run / "controllo_ritardo.txt").write_text("...\nESITO [4]: conforme (vittima e ombra conformi)\n")
    return run


def test_riepilogo_run_sintetico(tmp_path):
    d = coda.riepilogo_run(str(_run_sintetica(tmp_path)), usa_bag=False)
    assert d["printf"]["n_printf"] == 31
    # 3 rejections in w0 while moving; 14 between w0 and w1 (9 measured stopped, 5 inferred);
    # 1 final in w1
    assert d["scarti"] == {"totale": 18, "in_moto": 3, "fermo": 10, "fermo_dedotto": 5, "non_misurato": 0}
    ep = d["episodi"]
    assert [e["n"] for e in ep] == [3, 14, 1]
    assert ep[0]["finestra"] == "w0" and ep[0]["moto"] == {"in_moto": 3} and not ep[0]["perdita_di_posa"]
    assert ep[1]["finestra"] == "fra w0 e w1" and ep[1]["senza_posa_s"] == 1.5 and ep[1]["perdita_di_posa"]
    assert ep[2]["finestra"] == "w1" and not ep[2]["recuperata"] and ep[2]["perdita_di_posa"]
    assert len(d["perdite_di_posa"]) == 2
    assert d["run_end"][-1]["reason"] == "finestre_completate"
    assert [f["chamfer_vincitore_cm"] for f in d["finestre"]] == [4.2, 4.9]
    assert d["danno_run"]["victim_end_cm"] == 7.0
    t = coda.testo_riepilogo_run(d)
    assert "ESITO [4]: conforme" in t and "perdite di posa" in t and "w1: 4.90 / 4.70" in t


def test_riepilogo_del_giorno_con_dettaglio(tmp_path):
    run = _run_sintetica(tmp_path)
    st = str(tmp_path / "stato.json")
    coda.inizia(st, "nsga3_eps5", "1001", str(run))
    coda.chiudi(st, "nsga3_eps5", "1001", {"esito": "fatta", "motivo": "completa"})
    t = coda.riepilogo(coda.leggi(st), [("1001", "nsga3_eps5")])
    assert "Dettaglio delle run chiuse oggi (1 tentativi)" in t and "scarti della vittima (isGood=0): 18" in t


def test_stima_e_tempo_max():
    ore = {"NSGA-III soglia 5 cm": {5: (6.698, None), 8: (10.711, None)}}
    r = coda.stima_run("nsga3_eps5", 5, 3, 32, 16, 2, 64, 6.72, 1.5, ore)
    assert r["rollout_per_finestra"] == [514, 514]
    assert abs(r["mb"][0] - (1.5 + 6.72 * 514 * 4)) < 0.1
    assert r["ore"] == [6.698, 10.711]
    assert coda.tempo_max_s(10.711) == round(1.5 * 10.711 * 3600)
    assert coda.tempo_max_s(None) == 12 * 3600


def _a_secco(tmp_path, **extra):
    env = {**os.environ, "RADICE": str(tmp_path / "serieB"), "PILOTA_DISCO": str(tmp_path / "manca"),
           "ORE_T6": str(tmp_path / "manca.csv"), "BASELINE_DISCO": str(tmp_path / "manca")}
    env.pop("BASELINE_SERIEB", None)
    env.update(extra)
    return subprocess.run(["bash", str(ROOT / "scripts" / "coda_serieB.sh"), "--a-secco", "--semi", "1001 2001",
                           "baseline", "nsga3_eps10", "massimo"], capture_output=True, text=True, env=env, timeout=300)


def test_a_secco_non_scrive(tmp_path):
    p = _a_secco(tmp_path)
    assert p.returncode == 0, p.stdout + p.stderr
    assert p.stdout.count("######## run") == 6
    assert "scripts/run_campaign.sh straight nsga3 3 1 32 16" in p.stdout
    assert "--fixed-genome=" in p.stdout and "CHAMFER_BUDGET=10" in p.stdout
    assert "MOLA_DEBUG=1 DANNO_METRO=1" in p.stdout
    assert "[--mola-se-ne-accorge] [--controllo-catena] [--danno-metro]" in p.stdout
    assert "controllo_ritardo_run.py" in p.stdout and "riepilogo-run" in p.stdout
    assert "non accende --mola-se-ne-accorge" not in p.stdout
    # damage baseline: mandatory placeholder
    assert "[!] BASELINE_SERIEB non impostata" in p.stdout
    assert "_rip1_rip1" not in p.stdout
    assert not (tmp_path / "serieB").exists()


def test_a_secco_baseline_per_seme(tmp_path):
    (tmp_path / "b_s1001.csv").write_text("x,y\n")
    p = _a_secco(tmp_path, BASELINE_SERIEB=str(tmp_path / "b_s{seme}.csv"))
    assert p.returncode == 0, p.stdout + p.stderr
    assert f"[!] baseline del danno assente: {tmp_path}/b_s2001.csv" in p.stdout
    assert f"ok: baseline del danno {tmp_path}/b_s1001.csv" in p.stdout
    (tmp_path / "b_s2001.csv").write_text("x,y\n")
    p = _a_secco(tmp_path, BASELINE_SERIEB=str(tmp_path / "b_s{seme}.csv"))
    assert "[!] baseline" not in p.stdout and "[!] BASELINE" not in p.stdout
    assert "baseline del danno: " in p.stdout


def test_a_secco_genoma_in_moto(tmp_path):
    """GENOMA_IN_MOTO: passed to run_campaign.sh and expected in the orchestrator and node
    commands; empty or 0 = unchanged."""
    p = _a_secco(tmp_path, GENOMA_IN_MOTO="2.0")
    assert p.returncode == 0, p.stdout + p.stderr
    env = [r for r in p.stdout.splitlines() if "+ env " in r and "scripts/run_campaign.sh" in r]
    assert env and all(" GENOMA_IN_MOTO=2.0 " in r for r in env)
    attese = [r for r in p.stdout.splitlines() if "opzioni attese:" in r]
    nodo = [r for r in p.stdout.splitlines() if "nel comando del perturbation_node:" in r]
    assert attese and all("[--genoma-in-moto 2.0]" in r for r in attese)
    assert nodo and all("[--genoma-in-moto 2.0]" in r for r in nodo)
    for v in ("0", ""):
        p0 = _a_secco(tmp_path, GENOMA_IN_MOTO=v)
        assert p0.returncode == 0 and "GENOMA_IN_MOTO" not in p0.stdout and "--genoma-in-moto" not in p0.stdout


def test_a_secco_chamfer_per_fase(tmp_path):
    """CHAMFER_PER_FASE=1: passed to run_campaign.sh, expected in the orchestrator and node
    commands (also together with GENOMA_IN_MOTO); 0 = unchanged."""
    p = _a_secco(tmp_path, CHAMFER_PER_FASE="1", GENOMA_IN_MOTO="2.0")
    assert p.returncode == 0, p.stdout + p.stderr
    env = [r for r in p.stdout.splitlines() if "+ env " in r and "scripts/run_campaign.sh" in r]
    assert env and all(" CHAMFER_PER_FASE=1 " in r and " GENOMA_IN_MOTO=2.0 " in r for r in env)
    for chiave in ("opzioni attese:", "nel comando del perturbation_node:"):
        righe = [r for r in p.stdout.splitlines() if chiave in r]
        assert righe and all("[--chamfer-per-fase]" in r and "[--genoma-in-moto 2.0]" in r for r in righe)
    p0 = _a_secco(tmp_path, CHAMFER_PER_FASE="0")
    assert p0.returncode == 0 and "CHAMFER_PER_FASE" not in p0.stdout and "--chamfer-per-fase" not in p0.stdout
