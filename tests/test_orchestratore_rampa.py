"""Genome ramp in the orchestrator (--rampa-genoma R): previous genome of the metre and its zones
on the snapshot, /attack/rampa message before the metre, zones and genome of every new genome
(victim and each twin candidate), no message at metre 0, in the nominal run or for the realigned
twin (cancel if the node holds data that would apply to it), start-up checks, history, bench G
and launchers. Off (R = 1): no message, no field. Fake objects, no ROS node, Isaac or MOLA;
tests on a recorded run are skipped if the run is missing.
"""

import json
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from src.optimization import attack_orchestrator as ao  # noqa: E402
from src.optimization import banco_g  # noqa: E402
from src.perturbations import zone_attaccante as za  # noqa: E402

t31 = pytest.importorskip("test_orchestratore_D31")
REP = t31.REP
serve_run = pytest.mark.skipif(not (REP / "history.json").is_file() or not (REP / "zone").is_dir(),
                               reason="run della prova di accettazione assente")
BASE = ["--zone-attaccante", "--metro-al-nodo", "--settle-clouds", "20", "--reopen-clouds", "20",
        "--attack-from", "spawn"]
N = [0.5] * 17
P = [0.2] * 17


@pytest.fixture(autouse=True)
def _rampa_pulita(monkeypatch):
    monkeypatch.setattr(ao, "RAMPA", {"R": 1, "vittima_ultimo": None, "prec": None, "prec_sha": None,
                                      "zone_prec": None, "metro": None, "sospesa": False, "seq": 0,
                                      "finestra": None, "ultimo": {}, "preinviata": {}, "ultimi_dati": None})
    monkeypatch.setattr(ao, "TWIN_STATE", {"window": 4})


# --- arguments ---

def test_argomento_spento_e_controlli():
    a = t31._args_main([])
    assert a.rampa_genoma == 1 and ao.problema_rampa(a) is None
    assert ao.problema_rampa(t31._args_main(t31.FISSO + BASE + ["--rampa-genoma", "9"])) is None
    assert ao.problema_rampa(t31._args_main(t31.FISSO + BASE + ["--rampa-genoma", "12"])) is None
    assert "assestamento" in ao.problema_rampa(t31._args_main(t31.FISSO + BASE + ["--rampa-genoma", "13"]))
    assert "--zone-attaccante" in ao.problema_rampa(t31._args_main(t31.FISSO + ["--rampa-genoma", "9"]))
    a = t31._args_main(t31.FISSO + BASE + ["--rampa-genoma", "9", "--attack-from", "motion"])
    assert "spawn" in ao.problema_rampa(a)
    a = t31._args_main(t31.FISSO + BASE + ["--rampa-genoma", "9", "--reopen-clouds", "15"])
    assert "assestamento 15" in ao.problema_rampa(a)
    assert ">= 1" in ao.problema_rampa(t31._args_main(t31.FISSO + BASE + ["--rampa-genoma", "0"]))


def test_rampa_mismatch():
    n = types.SimpleNamespace(status_t=1.0, node_rampa_on=True, node_rampa={"R": 9}, spin=lambda s: None)
    assert ao.rampa_mismatch(n, 9) is None
    assert "R 9" in ao.rampa_mismatch(n, 5, timeout=0.0)
    assert "e l'orchestratore no" in ao.rampa_mismatch(n, 1)
    n0 = types.SimpleNamespace(status_t=1.0, node_rampa_on=False, node_rampa=None, spin=lambda s: None)
    assert ao.rampa_mismatch(n0, 1) is None and "senza rampa" in ao.rampa_mismatch(n0, 9, timeout=0.0)


# --- messages ---

class _NodoRampa:
    """Fake node: receives /attack/rampa and answers in the status like _rampa_cb."""

    def __init__(self, rifiuta=None):
        self.node_rampa = {"R": 9, "ricevuta": {}}
        self.mandati, self.ordine = [], []
        self.rifiuta = rifiuta
        for nome in ("manda_rampa", "stato_rampa", "set_genome", "set_victim_genome"):
            setattr(self, nome, types.MethodType(getattr(ao.AttackNode, nome), self))

    def _zone_pub(self, topic):
        return topic, types.SimpleNamespace

    def _publish_confirmed(self, pub, msg, confirmed, what, **kw):
        if pub == "/attack/rampa":
            d = json.loads(msg.data)
            self.mandati.append(d)
            if d.get("annulla"):
                # node cancel command: no file read, receipt cancelled
                self.ordine.append(("rampa", d["flusso"], "annulla"))
                self.node_rampa["ricevuta"][d["flusso"]] = {
                    "seq": d["seq"], "stato": "annullata", "genoma_prec_sha": None, "genoma_sha_nuovo": None,
                    "zone_prec_hash": None}
                return confirmed()
            self.ordine.append(("rampa", d["flusso"], d["genoma_sha_nuovo"]))
            stato = self.rifiuta or ("ok" if d["zone_prec"]["path"] else "file_assente")
            self.node_rampa["ricevuta"][d["flusso"]] = {
                "seq": d["seq"], "stato": stato, "genoma_prec_sha": za.genoma_sha(d["genoma_prec"]),
                "genoma_sha_nuovo": d["genoma_sha_nuovo"], "zone_prec_hash": d["zone_prec"]["hash"]}
        else:
            self.ordine.append(("genoma", what))
        return confirmed()

    # set_genome / set_victim_genome
    node_genome_id = victim_genome_id = 0
    genome_pub = _victim_genome_pub = "/attack/genome"
    _F32 = types.SimpleNamespace

    def set_metro(self, flusso):
        self.ordine.append(("metro", flusso))

    def set_zone(self, flusso, genome):
        self.ordine.append(("zone", flusso))


def _arma(k=4):
    ao.RAMPA.update({"R": 9, "prec": list(P), "prec_sha": za.genoma_sha(P), "metro": k,
                     "zone_prec": {"path": "/z/w4_p.npz", "sha256": "a" * 64, "hash": "b" * 64},
                     "finestra": {"messaggi": []}})
    ao.TWIN_STATE["window"] = k


def test_spenta_nessun_messaggio():
    n = _NodoRampa()
    assert n.manda_rampa("gemello", N) is None and n.mandati == []
    assert n.stato_rampa("gemello") is None


def test_metro_0_e_senza_precedente():
    n = _NodoRampa()
    ao.RAMPA["R"] = 9
    ao.TWIN_STATE["window"] = 0
    r = n.manda_rampa("gemello", N)
    assert r["esito"] == "senza_precedente" and n.mandati == []


def test_messaggio_per_ogni_candidato():
    n = _NodoRampa()
    _arma()
    for g in (N, [0.1] * 17, N):
        r = n.manda_rampa("gemello", g)
        assert r["esito"] == "mandata" and r["confermata"] and r["stato_nodo"] == "ok"
    assert [d["seq"] for d in n.mandati] == [1, 2, 3]
    d = n.mandati[0]
    assert d["flusso"] == "gemello" and d["genoma_prec"] == P and d["genoma_sha_nuovo"] == za.genoma_sha(N)
    assert d["zone_prec"] == {"path": "/z/w4_p.npz", "sha256": "a" * 64, "hash": "b" * 64}
    assert len(ao.RAMPA["finestra"]["messaggi"]) == 3 and ao.RAMPA["ultimo"]["gemello"]["seq"] == 3


def test_riallineato_annullo_solo_se_serve():
    n = _NodoRampa()
    _arma()
    n.manda_rampa("gemello", N)                    # last candidate or trial with the winner genome
    ao.RAMPA["sospesa"] = True
    r = n.manda_rampa("gemello", N)                # realigned twin with the same genome: cancel
    assert r["esito"] == "annullata" and n.mandati[-1] == {"seq": 2, "flusso": "gemello", "annulla": True}
    assert n.node_rampa["ricevuta"]["gemello"]["stato"] == "annullata"
    k = len(n.mandati)
    r = n.manda_rampa("gemello", N)                # data already cancelled: no message
    assert r["esito"] == "sospesa" and len(n.mandati) == k


def test_braccio_senza_rampa_annulla_dati_altrui():
    n = _NodoRampa()
    _arma()
    n.manda_rampa("gemello", N)
    ao.RAMPA.update(prec=None, prec_sha=None)
    r = n.manda_rampa("gemello", N)
    assert r["esito"] == "annullata" and r["motivo"] == "senza_precedente"
    assert n.mandati[-1].get("annulla") is True and "genoma_prec" not in n.mandati[-1]
    r = n.manda_rampa("gemello", [0.1] * 17)       # data for another genome: unused by the node
    assert r["esito"] == "senza_precedente"


def test_rifiuto_del_nodo_ferma_la_run():
    n = _NodoRampa(rifiuta="prec_sha256_diverso")
    _arma()
    with pytest.raises(ao.ErroreConfigurazioneZone, match="rampa gemello w4"):
        n.manda_rampa("gemello", N)


def test_preinviata_non_si_ripete():
    n = _NodoRampa()
    _arma()
    n.manda_rampa("gemello", N)
    ao.RAMPA["preinviata"] = {"gemello": za.genoma_sha(N)}
    assert n.manda_rampa("gemello", N)["seq"] == 1 and len(n.mandati) == 1


def test_ordine_rampa_metro_zone_genoma():
    n = _NodoRampa()
    _arma()
    n.node_genome_id = 0

    def conf(pub, msg, confirmed, what, **kw):
        if pub == "/attack/rampa":
            return _NodoRampa._publish_confirmed(n, pub, msg, confirmed, what)
        n.ordine.append(("genoma", what))
        n.node_genome_id += 1
        n.victim_genome_id += 1
        return True
    n._publish_confirmed = conf
    n.set_genome(N)
    n.set_victim_genome(N)
    assert [o[0] for o in n.ordine] == ["rampa", "metro", "zone", "genoma"] * 2
    assert [o[1] for o in n.ordine if o[0] == "rampa"] == ["gemello", "vittima"]
    assert ao.RAMPA["vittima_ultimo"] == N             # P of the next metre


def test_vittima_ultimo_aggiornato_anche_a_rampa_spenta():
    n = _NodoRampa()
    n._publish_confirmed = lambda pub, msg, confirmed, what, **kw: True
    n.set_victim_genome(N)
    assert ao.RAMPA["vittima_ultimo"] == N and n.mandati == []


# --- metre start ---

@serve_run
def test_prepara_rampa_metro_dati_veri(tmp_path, monkeypatch):
    """P = genome applied in w3 (e465); its zones on the w4 snapshot match the run's own."""
    h = json.loads((REP / "history.json").read_text())
    za_w4 = h[4]["zone_attaccante"]
    foto = za_w4["fotografia"]
    dz = tmp_path / "zone"
    dz.mkdir()
    (dz / Path(foto["path"]).name).write_bytes(Path(foto["path"]).read_bytes())
    f = za.carica_fotografia(dz / Path(foto["path"]).name, foto["sha256"], foto["hash"])
    monkeypatch.setitem(ao.ZONE, "foto", f)
    monkeypatch.setitem(ao.ZONE, "metro", 4)
    monkeypatch.setitem(ao.ZONE, "out", str(tmp_path))
    ao.RAMPA.update({"R": 9, "vittima_ultimo": h[3]["genome"]})
    node = types.SimpleNamespace(node_gen_limits=h[4]["node_gen_limits"])
    fin = ao.prepara_rampa_metro(node, 4)
    sha = za.genoma_sha(h[3]["genome"])
    assert sha == "57e2b5a6e61e41ea" and fin["genoma_prec_sha"] == sha
    rif = next(z for z in za_w4["zone"] if z["genoma_sha"] == sha)
    assert fin["zone_prec"]["hash"] == rif["hash"] and Path(fin["zone_prec"]["path"]).name == f"w4_{sha}.npz"
    assert ao.RAMPA["prec"] == h[3]["genome"] and ao.RAMPA["metro"] == 4
    # metre 0: no previous genome
    assert ao.prepara_rampa_metro(node, 0)["motivo"] == "metro 0" and ao.RAMPA["prec"] is None


# --- realignment, history ---

def test_riallineato_sospende_la_rampa(monkeypatch, tmp_path):
    """The realigned twin runs with RAMPA["sospesa"]: set_genome sends no data (or cancels it)."""
    node = t31._NodeRipresa()
    isaac = t31._IsaacRipresa(node)
    mola = types.SimpleNamespace(log_path=None, last_stop=None)
    monkeypatch.setitem(ao.STALL_POLICY, "window", 4)
    monkeypatch.setitem(ao.CONTROLLO_ETA, "finestre", {})
    monkeypatch.setitem(ao.ASSETTO, "registra", False)
    ao.RAMPA["R"] = 9
    node.node_rampa = {"R": 9, "ricevuta": {}}
    node.stato_rampa = types.MethodType(ao.AttackNode.stato_rampa, node)
    visti = []

    def esegui(isa_p, tag, fs):
        visti.append(ao.RAMPA["sospesa"])
        node.est_source, node.last_trace = "twin", []
        return (0, 0, 0), (4.9, 0.4, 0.0), 41.0, "ok", 0.95, 11
    attacker = types.SimpleNamespace(run_origin=(0.0177, 0.0, 0.0), believed_pose=lambda n: (4.0855, 0.519, -0.1189))
    rial = ao.RiallineoVittima(4, esegui, t31._Ombra(None), attacker, tmp_path, mola=mola)
    rial(isaac, node, {"tag": "w4_applied"})
    assert visti == [True] and ao.RAMPA["sospesa"] is False
    assert rial.rec["gemello"]["rampa"]["R"] == 9


def test_twin_start_fields_con_rampa():
    node = types.SimpleNamespace()
    assert "rampa" not in ao._twin_start_fields(node)
    ao.RAMPA["R"] = 9
    node = _NodoRampa()
    node.node_rampa["gemello"] = {"genoma_sha": "x", "genoma_prec_sha": "y", "stato": "finita", "k": 30,
                                  "alpha": 1.0, "seq": 3}
    r = ao._twin_start_fields(node)["rampa"]
    assert r["R"] == 9 and r["nodo"]["stato"] == "finita" and r["messaggio"] is None


def test_segmento_history_rampa():
    src = (ROOT / "src" / "optimization" / "attack_orchestrator.py").read_text()
    assert 'd31_rec["rampa"] = {**(RAMPA.get("finestra") or {}), "vittima": node.stato_rampa("vittima")}' in src
    i = src.index("node.fotografia_metro(k, frame_start)")
    assert "prepara_rampa_metro(node, k)" in src[i:i + 200]


# --- bench G ---

def _rampa_spec(tmp_path):
    """Ramp in bench format with a real zone file of the run (e465 on w4) and genome N = e480."""
    h = json.loads((REP / "history.json").read_text())
    zw = h[4]["zone_attaccante"]
    zp = next(z for z in zw["zone"] if z["genoma_sha"] == za.genoma_sha(h[3]["genome"]))
    zn = next(z for z in zw["zone"] if z["genoma_sha"] == za.genoma_sha(h[4]["genome"]))
    foto = zw["fotografia"]

    def zona(z):
        return {"path": z["file"], "sha256": z["sha256"], "hash": z["hash"], "genoma_sha": z["genoma_sha"],
                "seme": int(z["seme"]), "metro": 4, "fotografia_path": foto["path"],
                "fotografia_sha256": foto["sha256"]}
    return h, {"genoma_prec": h[3]["genome"], "zone_prec": zona(zp), "fonte": "test"}, zona(zn)


@serve_run
def test_banco_specifica_con_rampa(tmp_path):
    h, rp, zn = _rampa_spec(tmp_path)
    s = t31.tg._spec(tmp_path)
    s["casi"] = s["casi"][:1]
    s["casi"][0]["metro"] = 4
    s["casi"][0]["bracci"] = [{"nome": "G", "init": "attacker_est", "genome": h[4]["genome"], "rampa": rp,
                               "zone": zn}]
    p = tmp_path / "s.json"
    p.write_text(json.dumps(s))
    spec = banco_g.load_spec(p)
    assert banco_g.usa_rampa(spec)
    base = dict(zone_attaccante=True, metro_al_nodo=True, settle_clouds=20)
    assert banco_g.problema_estensioni(spec, t31.tg._args(rampa_genoma=9, **base)) is None
    assert "--rampa-genoma R > 1" in banco_g.problema_estensioni(spec, t31.tg._args(**base))
    assert "oltre" in banco_g.problema_estensioni(spec, t31.tg._args(rampa_genoma=13, **base))
    s["casi"][0]["bracci"][0]["rampa"] = {**rp, "genoma_prec": h[2]["genome"]}
    p.write_text(json.dumps(s))
    with pytest.raises(ValueError, match="genoma_sha"):
        banco_g.load_spec(p)


@serve_run
def test_banco_ciclo_rampa_finto(tmp_path, monkeypatch):
    """Trio G, V, Gr with the ramp: message before metre and zones for G and V, cancel for Gr."""
    h, rp, zn = _rampa_spec(tmp_path)
    s = t31.tg._spec(tmp_path)
    s["casi"] = s["casi"][:1]
    s["casi"][0]["metro"] = 4
    g = h[4]["genome"]
    s["casi"][0]["bracci"] = [{"nome": "G", "init": "attacker_est", "genome": g, "rampa": rp},
                              {"nome": "V", "init": "ripristino", "genome": g, "rampa": rp},
                              {"nome": "Gr", "init": "riallineato", "riallinea_da": "V", "genome": g, "rampa": rp}]
    s["ripetizioni"] = 1
    p = tmp_path / "spec.json"
    p.write_text(json.dumps(s))
    out = tmp_path / "rep_01"
    out.mkdir()
    isaac, node = t31._IsaacTrio(), t31._NodeTrio()
    nr = _NodoRampa()
    node.node_rampa, node.node_rampa_on, node.status_t = nr.node_rampa, True, 1.0
    node._zone_pub = nr._zone_pub
    node._publish_confirmed = nr._publish_confirmed
    node.mandati, node.ordine = nr.mandati, nr.ordine
    node.manda_rampa = types.MethodType(ao.AttackNode.manda_rampa, node)
    node.stato_rampa = types.MethodType(ao.AttackNode.stato_rampa, node)
    visti = []

    def _rollout(genome, isa, mola, nd, goal, horizon, tag, **kw):
        isa.pause()
        isa.restore()
        n0 = len(nd.mandati)
        nd.manda_rampa("gemello", genome)              # as set_genome: no extra message
        visti.append((tag, len(nd.mandati) - n0))
        if "dopo_assestamento" in kw:
            kw["dopo_assestamento"](isa, nd, {"tag": tag, "ruolo": "gemello", "mola": mola})
        lp = tmp_path / f"{tag}.log"
        lp.write_text("")
        mola.log_path = str(lp)
        return None, (1.0, 0.0, 0.0), 0.0, "ok", 1.0, 40

    monkeypatch.setattr(ao, "rollout", _rollout)
    monkeypatch.setattr(ao, "SHADOW_QUIET_S", 0.0)
    args = types.SimpleNamespace(g_banco=str(p), g_overlay="/ovl", g_resave="ogni", horizon=1.0, settle_sec=2.0,
                                 catchup_sec=0.8, attack_from="spawn", stop_on="estimate", drift_reset_on="keep",
                                 trace=False, start_pose="-5.99,-1.0,0", damage_ref=None, baseline_path=None,
                                 settle_clouds=20, rampa_genoma=9)
    assert ao._g_banco_run(args, isaac, types.SimpleNamespace(alive=lambda: True), node, out, [(0.0, 0.0)]) == 0
    assert [v[1] for v in visti] == [0, 0, 0]
    sha = za.genoma_sha(g)
    assert [m.get("genoma_sha_nuovo", "annulla") for m in node.mandati] == [sha, sha, "annulla"]
    assert all(m["genoma_prec"] == h[3]["genome"] and Path(m["zone_prec"]["path"]).parent == out / "zone"
               for m in node.mandati[:2])
    d = json.loads((out / "banco_G.json").read_text())
    es = {e["braccio"]: e for e in d["esiti"]}
    assert es["G"]["rampa"]["messaggio"]["esito"] == "mandata"
    assert es["Gr"]["rampa"]["messaggio"]["esito"] == "annullata"


# --- spec builder and launchers ---

@serve_run
def test_spec_configurazione_2_dalla_run(tmp_path):
    out = tmp_path / "spec.json"
    r = subprocess.run([sys.executable, "-B", str(ROOT / "scripts" / "banco_G_d_spec.py"), "--out", str(out),
                        "--piano", "configurazioni", "--rep", str(REP), "--metro", "4", "--configurazioni", "2",
                        "--rampa", "9", "--ripetizioni", "1"], capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stdout + r.stderr
    s = json.loads(out.read_text())
    h = json.loads((REP / "history.json").read_text())
    rp = {b["nome"]: b.get("rampa") for b in s["casi"][0]["bracci"]}
    assert rp["G"]["genoma_prec"] == h[3]["genome"] and rp["G"] == rp["V"] == rp["Gr"]
    assert rp["G"]["zone_prec"]["genoma_sha"] == "57e2b5a6e61e41ea"
    rif = next(z for z in h[4]["zone_attaccante"]["zone"] if z["genoma_sha"] == "57e2b5a6e61e41ea")
    assert rp["G"]["zone_prec"]["hash"] == rif["hash"]
    r = subprocess.run([sys.executable, "-B", str(ROOT / "scripts" / "banco_G_d_spec.py"), "--out",
                        str(tmp_path / "x.json"), "--piano", "configurazioni", "--rep", str(REP), "--metro", "4",
                        "--configurazioni", "2", "--rampa", "13"], capture_output=True, text=True, cwd=ROOT)
    assert r.returncode != 0 and "--rampa 13" in r.stdout + r.stderr


def test_run_campaign_rampa():
    assert t31._blocco_d31({}) == (0, "[]")
    assert t31._blocco_d31({"RAMPA": "1"}) == (0, "[]")
    assert t31._blocco_d31({"RAMPA": "9", "ORCH_EXTRA": "--x"}) == (0, "[--rampa-genoma 9 --x]")
    assert t31._blocco_d31({"RAMPA": "nove"})[0] != 0
    src = (ROOT / "scripts" / "run_campaign.sh").read_text()
    assert '[[ -n "$RAMPA_NODO" ]] && extra="$extra$RAMPA_NODO"' in src


def test_coda_rampa():
    src = (ROOT / "scripts" / "coda_serieB.sh").read_text()
    i0 = src.index("D31_ENV=(); D31_ATTESE=(); D31_DESC=\"\"")
    i1 = src.index("\n# --- Arguments ---")
    codice = ("set -uo pipefail\n" + src[i0:i1]
              + '\necho "${#D31_ENV[@]}|${D31_ATTESE[*]-}|${D31_NODO_ATTESE[*]-}|${D31_DESC}|"\n')
    r = subprocess.run(["bash", "-c", codice], capture_output=True, text=True, env={"PATH": "/usr/bin:/bin"})
    assert r.stdout.strip() == "0||||"
    r = subprocess.run(["bash", "-c", codice], capture_output=True, text=True,
                       env={"PATH": "/usr/bin:/bin", "RAMPA": "9"})
    assert r.stdout.strip() == "1|--rampa-genoma 9|--rampa-genoma 9| RAMPA=9|"
    assert '(( ${#D31_NODO_ATTESE[@]} )) && NODO_ATTESE+=("${D31_NODO_ATTESE[@]}")' in src
