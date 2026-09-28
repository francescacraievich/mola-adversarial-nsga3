"""
Watcher di play/pause per Isaac Sim.

Gira nello Script Editor: con la scena caricata (in pausa), incollare questo
script, premere Run, poi lanciare l'orchestratore (src/optimization/) o
src/baseline/run_baseline_experiment.py.

Ad ogni frame controlla due file trigger e comanda la timeline:
    /tmp/isaac_play.trigger   -> play()  (creato da run_baseline_experiment.py
                                          quando MOLA e' pronto)
    /tmp/isaac_pause.trigger  -> pause() (creato dall'orchestratore dell'attacco)
    /tmp/isaac_state.txt      <- "playing" oppure "paused", scritto ad ogni frame

L'orchestratore congela la simulazione mentre NSGA-III valuta i candidati:
l'ottimizzazione richiede minuti di wall-clock e senza pausa il genoma vincente
sarebbe obsoleto al momento del caricamento. Il sim-time resta continuo e la
fisica riprende da dove si era fermata. A simulazione ferma il LiDAR non produce
nuvole, quindi la pausa fa anche da cancello per il nodo di perturbazione.
"""

import os
import omni
import omni.kit.app
import omni.timeline

_timeline = omni.timeline.get_timeline_interface()
_PLAY = "/tmp/isaac_play.trigger"
_PAUSE = "/tmp/isaac_pause.trigger"
_STATE = "/tmp/isaac_state.txt"


def _check_trigger(_event):
    if os.path.exists(_PLAY):
        os.remove(_PLAY)
        _timeline.play()
        print("[isaac_ctrl] Play")

    if os.path.exists(_PAUSE):
        os.remove(_PAUSE)
        _timeline.pause()
        print("[isaac_ctrl] Pause")

    # Il trigger dice che la pausa e' stata chiesta, questo file conferma che e'
    # avvenuta: l'orchestratore lo legge prima di usare le nuvole.
    try:
        with open(_STATE, "w") as f:
            f.write("playing" if _timeline.is_playing() else "paused")
    except OSError:
        pass


# Trigger rimasti da esecuzioni precedenti.
for _f in (_PLAY, _PAUSE):
    if os.path.exists(_f):
        os.remove(_f)

if "isaac_ctrl_sub" in dir():
    try:
        isaac_ctrl_sub.unsubscribe()
    except Exception:
        pass

isaac_ctrl_sub = omni.kit.app.get_app().get_update_event_stream().create_subscription_to_pop(
    _check_trigger,
    name="isaac_play_pause_trigger_watcher",
)

print(f"[isaac_ctrl] Watcher attivo")
print(f"[isaac_ctrl]   play  : {_PLAY}")
print(f"[isaac_ctrl]   pause : {_PAUSE}")
print(f"[isaac_ctrl]   stato : {_STATE}")
