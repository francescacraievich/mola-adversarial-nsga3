#!/usr/bin/env python3
"""
Waypoint follower in anello chiuso sulla stima di MOLA.

Il robot naviga solo su /lidar_odometry/pose: se un attacco sposta la stima,
il robot sterza di conseguenza e finisce fisicamente altrove. La posa vera
(/chassis/odom) viene solo registrata e non entra mai nel controllo.

MOLA parte da (0,0,0) nella posa iniziale del robot, quindi i waypoint sono
espressi in un frame comune a stima e verita'. La metrica finale e' il divario
stima-verita' a fine percorso, confrontato con la soglia di rumore della
baseline pulita. I comandi vanno su /cmd_vel; con --log il tracciato
(stima, verita', divario, stato) e' salvato in CSV.

Uso:
    python3 src/nodes/waypoint_follower_node.py --waypoints "4,0; 4,2.5; 0,2.5; 0,0" \
        --warmup-poses 60 --log data/attack/<file>.csv
"""

import argparse
import csv
import math
import sys
import time

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node


def yaw_from_quat(q) -> float:
    """Yaw da quaternione, assumendo moto planare."""
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                      1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def angle_diff(a: float, b: float) -> float:
    """Differenza angolare a-b normalizzata in (-pi, pi]."""
    return (a - b + math.pi) % (2 * math.pi) - math.pi


def parse_waypoints(s: str):
    """'4,0; 4,2.5; 0,2.5; 0,0' -> [(4.0,0.0), (4.0,2.5), ...]"""
    pts = []
    for chunk in s.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        x, y = chunk.split(",")
        pts.append((float(x), float(y)))
    if not pts:
        raise ValueError("nessun waypoint valido")
    return pts


class WaypointFollower(Node):

    def __init__(self, args):
        super().__init__("waypoint_follower")

        self.waypoints = parse_waypoints(args.waypoints)
        self.tol = args.tolerance
        self.max_v = args.max_speed
        self.max_w = args.max_turn
        self.align_deg = args.align_threshold
        self.k_lin = args.k_lin
        self.k_ang = args.k_ang
        self.wp_timeout = args.waypoint_timeout
        self.pose_timeout = args.pose_timeout
        self.stop_at_waypoints = not args.no_stop

        # Posa di controllo: la stima di MOLA, l'unica che guida.
        self.est = None              # (x, y, yaw)
        self.est_stamp = 0.0         # wall-clock dell'ultimo messaggio
        self.n_poses = 0
        self.warmup_poses = args.warmup_poses
        self.create_subscription(Odometry, args.pose_topic, self._est_cb, 50)

        # Posa vera: solo registrata, mai usata nel controllo.
        self.truth = None
        self.create_subscription(Odometry, args.truth_topic, self._truth_cb, 50)

        self.cmd_pub = self.create_publisher(Twist, args.cmd_topic, 10)

        self.wp_idx = 0
        self.state = "WAIT_POSE"     # WAIT_POSE | ALIGN | DRIVE | SETTLE | DONE
        self.wp_start_time = None
        self.settle_until = 0.0
        self.t0 = time.time()

        self.log_rows = []
        self.log_path = args.log

        # Controllo a cadenza fissa e non un comando per messaggio: se l'attacco
        # rallenta o blocca MOLA, un controller agganciato ai messaggi fermerebbe
        # il robot da solo mascherando l'effetto. Qui il watchdog e' esplicito.
        self.timer = self.create_timer(1.0 / args.rate, self._tick)

        self.get_logger().info("=" * 64)
        self.get_logger().info("  WAYPOINT FOLLOWER \u2014 closed loop su stima MOLA")
        self.get_logger().info("=" * 64)
        self.get_logger().info(f"  controllo su : {args.pose_topic}   (stima)")
        self.get_logger().info(f"  registra     : {args.truth_topic}  (verita', non usata)")
        self.get_logger().info(f"  waypoint     : {self.waypoints}")
        self.get_logger().info(f"  tolleranza   : {self.tol:.2f} m")
        self.get_logger().info(f"  In attesa di {self.warmup_poses} pose da MOLA...")

    # ------------------------------------------------------------------
    # Callback
    # ------------------------------------------------------------------

    def _est_cb(self, msg: Odometry):
        p = msg.pose.pose
        self.est = (p.position.x, p.position.y, yaw_from_quat(p.orientation))
        self.est_stamp = time.time()
        self.n_poses += 1

        if self.state == "WAIT_POSE":
            # La prima posa e' (0,0,0) subito dopo la relocalizzazione, con il
            # modello di velocita' non ancora assestato: si attende che la stima
            # sia affidabile prima di chiudere l'anello su di essa.
            if self.n_poses < self.warmup_poses:
                if self.n_poses == 1:
                    self.get_logger().info(
                        f"  Prima posa da MOLA. Attendo {self.warmup_poses} pose "
                        f"di assestamento prima di muovere..."
                    )
                return
            self.state = "ALIGN"
            self.wp_start_time = time.time()
            self.get_logger().info(
                f"  Assestamento completato ({self.n_poses} pose). "
                f"Posa iniziale: ({self.est[0]:.2f}, {self.est[1]:.2f}, "
                f"{math.degrees(self.est[2]):.1f}\u00b0) \u2014 parto."
            )

    def _truth_cb(self, msg: Odometry):
        p = msg.pose.pose
        self.truth = (p.position.x, p.position.y, yaw_from_quat(p.orientation))

    # ------------------------------------------------------------------
    # Loop di controllo
    # ------------------------------------------------------------------

    def _publish(self, v: float, w: float):
        m = Twist()
        m.linear.x = float(v)
        m.angular.z = float(w)
        self.cmd_pub.publish(m)

    def _tick(self):
        if self.state in ("WAIT_POSE", "DONE"):
            return

        # Watchdog: senza pose da MOLA (es. ICP che non converge) il robot si
        # ferma.
        age = time.time() - self.est_stamp
        if age > self.pose_timeout:
            self._publish(0.0, 0.0)
            self.get_logger().warn(
                f"  Nessuna posa da MOLA da {age:.1f}s \u2014 fermo il robot. "
                "Sotto attacco puo' essere il sintomo di ICP che non converge."
            )
            return

        if self.state == "SETTLE":
            if time.time() < self.settle_until:
                self._publish(0.0, 0.0)
                return
            self.state = "ALIGN"
            self.wp_start_time = time.time()

        x, y, yaw = self.est
        tx, ty = self.waypoints[self.wp_idx]
        dx, dy = tx - x, ty - y
        dist = math.hypot(dx, dy)
        bearing_err = angle_diff(math.atan2(dy, dx), yaw)

        self._log(dist, bearing_err)

        # Arrivo.
        if dist < self.tol:
            self._publish(0.0, 0.0)
            self._report_waypoint(dist)
            self.wp_idx += 1
            if self.wp_idx >= len(self.waypoints):
                self._finish()
                return
            if self.stop_at_waypoints:
                # Fermata prima di riorientarsi, cosi' la velocita' angolare
                # residua non si integra nel tratto successivo.
                self.state = "SETTLE"
                self.settle_until = time.time() + 1.0
            else:
                self.state = "ALIGN"
                self.wp_start_time = time.time()
            return

        # Timeout sul waypoint.
        if time.time() - self.wp_start_time > self.wp_timeout:
            self._publish(0.0, 0.0)
            self.get_logger().error(
                f"  Timeout sul waypoint {self.wp_idx + 1} dopo {self.wp_timeout:.0f}s "
                f"(distanza residua {dist:.2f} m). Interrompo."
            )
            self._finish()
            return

        # Legge di controllo.
        if abs(bearing_err) > math.radians(self.align_deg):
            # Errore di puntamento grande: rotazione sul posto. Avanzare mal
            # orientati allunga il percorso e in curva stretta puo' dare un ciclo
            # limite intorno al waypoint.
            self.state = "ALIGN"
            w = max(-self.max_w, min(self.max_w, self.k_ang * bearing_err))
            # Velocita' minima per vincere l'attrito statico.
            if 0 < abs(w) < 0.05:
                w = math.copysign(0.05, w)
            self._publish(0.0, w)
        else:
            self.state = "DRIVE"
            v = min(self.max_v, self.k_lin * dist)
            w = max(-self.max_w, min(self.max_w, self.k_ang * bearing_err))
            self._publish(v, w)

    # ------------------------------------------------------------------
    # Registrazione e resoconto
    # ------------------------------------------------------------------

    def _log(self, dist, bearing_err):
        if self.log_path is None:
            return
        ex, ey, eyaw = self.est
        tx, ty, tyaw = self.truth if self.truth else (float("nan"),) * 3
        self.log_rows.append({
            "t": round(time.time() - self.t0, 3),
            "wp": self.wp_idx,
            "est_x": round(ex, 4), "est_y": round(ey, 4),
            "est_yaw": round(math.degrees(eyaw), 2),
            "true_x": round(tx, 4), "true_y": round(ty, 4),
            "true_yaw": round(math.degrees(tyaw), 2),
            # Divario stima-verita': misura diretta dell'effetto dell'attacco.
            "est_true_gap": round(math.hypot(ex - tx, ey - ty), 4)
            if self.truth else float("nan"),
            "dist_to_wp": round(dist, 4),
            "bearing_err_deg": round(math.degrees(bearing_err), 2),
            "state": self.state,
        })

    def _report_waypoint(self, dist):
        tx, ty = self.waypoints[self.wp_idx]
        msg = (f"  WP {self.wp_idx + 1}/{len(self.waypoints)} ({tx:.2f}, {ty:.2f}) "
               f"raggiunto secondo MOLA  [residuo {dist:.3f} m]")
        if self.truth:
            # Errore vero sul waypoint: spostamento non rilevato dal sistema.
            real = math.hypot(self.truth[0] - tx, self.truth[1] - ty)
            gap = math.hypot(self.est[0] - self.truth[0], self.est[1] - self.truth[1])
            msg += f"\n      posizione vera: ({self.truth[0]:.2f}, {self.truth[1]:.2f})"
            msg += f"   errore reale sul waypoint: {real*100:.1f} cm"
            msg += f"   divario stima-verita': {gap*100:.1f} cm"
        self.get_logger().info(msg)

    def _dump_log(self):
        if not self.log_path or not self.log_rows:
            return
        import os
        os.makedirs(os.path.dirname(self.log_path) or ".", exist_ok=True)
        with open(self.log_path, "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(self.log_rows[0].keys()))
            wr.writeheader()
            wr.writerows(self.log_rows)
        self.get_logger().info(f"  Log: {self.log_path}  ({len(self.log_rows)} righe)")

        # Riepilogo, utile anche a percorso non completato.
        gaps = [r["est_true_gap"] for r in self.log_rows
                if isinstance(r["est_true_gap"], float) and r["est_true_gap"] == r["est_true_gap"]]
        if gaps:
            self.get_logger().info(
                f"  Divario stima-verita': primo {gaps[0]*100:.1f} cm   "
                f"massimo {max(gaps)*100:.1f} cm   ultimo {gaps[-1]*100:.1f} cm"
            )

    def _finish(self):
        self._publish(0.0, 0.0)
        self.state = "DONE"

        self.get_logger().info("\n" + "=" * 64)
        self.get_logger().info("  PERCORSO TERMINATO")
        self.get_logger().info("=" * 64)

        if self.est and self.truth:
            gx, gy = self.waypoints[-1]
            gap = math.hypot(self.est[0] - self.truth[0], self.est[1] - self.truth[1])
            real = math.hypot(self.truth[0] - gx, self.truth[1] - gy)
            self.get_logger().info(
                f"  MOLA crede di essere in : ({self.est[0]:6.2f}, {self.est[1]:6.2f})\n"
                f"  Il robot e' davvero in  : ({self.truth[0]:6.2f}, {self.truth[1]:6.2f})\n"
                f"  Ultimo waypoint         : ({gx:6.2f}, {gy:6.2f})\n"
                f"\n"
                f"  DIVARIO STIMA-VERITA'   : {gap*100:.1f} cm\n"
                f"  ERRORE SUL BERSAGLIO    : {real*100:.1f} cm"
            )
            # Soglia = media + 3 sigma dell'ATE della baseline pulita sul
            # rettangolo (src/baseline/loop_baseline_stats.py).
            if gap * 1000 > 90.4:
                self.get_logger().info(
                    "\n  Il divario supera la soglia di baseline (90.4 mm):\n"
                    "  la deviazione non e' spiegabile come rumore."
                )
            else:
                self.get_logger().info(
                    "\n  Divario entro la banda di rumore della baseline (90.4 mm)."
                )

        if self.log_path and self.log_rows:
            self._dump_log()


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--waypoints", type=str, default="4,0; 4,2.5; 0,2.5; 0,0",
                    help="lista 'x,y; x,y; ...' nel frame di MOLA "
                         "(origine = posa di partenza del robot)")
    ap.add_argument("--pose-topic", type=str, default="/lidar_odometry/pose",
                    help="posa di CONTROLLO: la stima di MOLA")
    ap.add_argument("--truth-topic", type=str, default="/chassis/odom",
                    help="posa VERA, solo per registrazione")
    ap.add_argument("--cmd-topic", type=str, default="/cmd_vel")
    ap.add_argument("--tolerance", type=float, default=0.25,
                    help="raggio di arrivo in metri")
    ap.add_argument("--max-speed", type=float, default=0.8)
    ap.add_argument("--max-turn", type=float, default=0.5)
    ap.add_argument("--align-threshold", type=float, default=25.0,
                    help="sopra questo errore di puntamento (gradi) ruota sul posto")
    ap.add_argument("--k-lin", type=float, default=0.8)
    ap.add_argument("--k-ang", type=float, default=1.2)
    ap.add_argument("--rate", type=float, default=20.0)
    ap.add_argument("--waypoint-timeout", type=float, default=120.0)
    ap.add_argument("--pose-timeout", type=float, default=2.0,
                    help="secondi senza posa da MOLA prima di fermare il robot")
    ap.add_argument("--no-stop", action="store_true",
                    help="non fermarsi ai waypoint (percorso piu' fluido, "
                         "ma la velocita' angolare residua si integra)")
    ap.add_argument("--warmup-poses", type=int, default=20,
                    help="pose da attendere prima di muovere il robot: la prima "
                         "posa di MOLA e' (0,0,0) subito dopo la relocalizzazione, "
                         "quando il modello di velocita' non si e' assestato")
    ap.add_argument("--log", type=str, default=None)
    cli, ros_args = ap.parse_known_args(sys.argv[1:])

    rclpy.init(args=ros_args)
    node = WaypointFollower(cli)
    try:
        while rclpy.ok() and node.state != "DONE":
            rclpy.spin_once(node, timeout_sec=0.05)
    except KeyboardInterrupt:
        node._publish(0.0, 0.0)
        node.get_logger().info("Interrotto — salvo comunque il log.")
        # Il log va scritto anche su interruzione: le run che non arrivano in
        # fondo sono quelle da analizzare.
        node._dump_log()
    finally:
        node._publish(0.0, 0.0)
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
