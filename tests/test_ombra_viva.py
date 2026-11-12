"""--controllo-ombra-viva: if the shadow MOLA process dies during the run, the run stops at once
as a chain failure (da_rifare.json) and the queue repeats it.
"""
import importlib.util
import json
import re
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.optimization import attack_orchestrator as ao  # noqa: E402

spec = importlib.util.spec_from_file_location("coda", ROOT / "scripts" / "coda_serieB_stato.py")
coda = importlib.util.module_from_spec(spec)
spec.loader.exec_module(coda)


class Proc:
    def __init__(self, rc):
        self.rc = rc

    def poll(self):
        return self.rc


def nodo(ombra=None, watch=object()):
    n = types.SimpleNamespace(ombra=ombra, watch=watch, _t_check=0.0)
    n.check_ombra = types.MethodType(ao.AttackNode.check_ombra, n)
    return n


def test_ombra_morta_solleva_ombra_persa():
    n = nodo(types.SimpleNamespace(proc=Proc(-11), log_path="logs/mola_shadow.log"))
    with pytest.raises(ao.OmbraPersa, match="codice -11"):
        n.check_ombra()


def test_ombra_persa_e_uno_stallo_per_la_gestione_esistente():
    assert issubclass(ao.OmbraPersa, ao.IsaacStall)
    assert not issubclass(ao.OmbraPersa, ao.GuastoEtaNuvola)


@pytest.mark.parametrize("ombra", [None, types.SimpleNamespace(proc=None), types.SimpleNamespace(proc=Proc(None))])
def test_niente_eccezione_se_spenta_non_avviata_o_viva(ombra):
    nodo(ombra).check_ombra()


def test_check_stall_non_controlla_senza_watch(monkeypatch):
    # end of run: watch set to None before stopping the shadow, so no false alarm
    n = nodo(types.SimpleNamespace(proc=Proc(0), log_path="x"), watch=None)
    ao.AttackNode.check_stall(n)


def test_check_stall_controlla_l_ombra_prima_di_isaac(monkeypatch, tmp_path):
    n = nodo(types.SimpleNamespace(proc=Proc(-11), log_path="x"))
    monkeypatch.setattr(ao, "POSE_FILE", str(tmp_path / "manca.json"))
    with pytest.raises(ao.OmbraPersa):
        ao.AttackNode.check_stall(n)


def test_da_rifare_e_classificazione_della_coda(tmp_path):
    rec = ao.scrivi_da_rifare_ombra(tmp_path, 1, "ombra persa: test", 1001)
    d = json.loads((tmp_path / "da_rifare.json").read_text())
    assert d == rec and d["reason"] == "guasto_catena" and d["causa"] == "ombra_persa" and d["window"] == 1
    (tmp_path / "history.json").write_text(json.dumps([{"window": 1, "status": "ombra_persa", "causa": "ombra_persa"}]))
    s = coda.storia(str(tmp_path))
    assert s["guasto_catena"] is True and s["da_rifare"]["causa"] == "ombra_persa"
    assert "guasto_catena" in coda.ESITI_RIPETIBILI


def test_opzione_spenta_di_default_e_documentata():
    src = (ROOT / "src/optimization/attack_orchestrator.py").read_text()
    m = re.search(r'ap\.add_argument\("--controllo-ombra-viva", action="store_true"', src)
    assert m, "opzione assente o non store_true"
    # hook-up: node.ombra = shadow only with the option on and a shadow present
    assert re.search(r'if getattr\(args, "controllo_ombra_viva", False\):\n\s+if shadow is not None:\n\s+node\.ombra = shadow\n', src), \
        "aggancio dell'ombra al nodo assente o cambiato"
    assert src.count("node.ombra = shadow") == 1


def test_coda_passa_l_opzione_solo_se_accesa():
    sh = (ROOT / "scripts/coda_serieB.sh").read_text()
    assert 'CONTROLLO_OMBRA_VIVA="${CONTROLLO_OMBRA_VIVA:-0}"' in sh
    assert '"$CONTROLLO_OMBRA_VIVA" == "1" ]] && ORCH_X="$ORCH_X --controllo-ombra-viva"' in sh
    assert 'ATTESE+=("--controllo-ombra-viva")' in sh


def test_la_coda_classifica_guasto_catena_con_motivo(tmp_path):
    ao.scrivi_da_rifare_ombra(tmp_path, 1, "ombra persa: il processo MOLA dell'ombra e' terminato (codice -11)", 1001)
    (tmp_path / "history.json").write_text(json.dumps([{"window": 1, "status": "ombra_persa", "causa": "ombra_persa"}]))
    e = coda.classifica({"storia": coda.storia(str(tmp_path)), "rc": 4, "completata": False})
    assert e["esito"] == "guasto_catena" and "ombra persa" in e["motivo"]


def test_gli_except_larghi_lasciano_passare_ombra_persa():
    # matched_base, sync_wait_publisher, deferred start, wait_pose, replay wait_ready:
    # OmbraPersa must not turn into a failed rollout
    src = (ROOT / "src/optimization/attack_orchestrator.py").read_text()
    blocchi = re.findall(r"except OmbraPersa:\n\s+raise\n\s+except (RuntimeError|Exception) as e:", src)
    assert len(blocchi) == 5, blocchi
