#!/usr/bin/env python3
"""
ROS 2 node that adapts the Isaac Sim RTX LiDAR cloud to MOLA's input.

Subscribes to PointCloud2 (x, y, z) on /front_3d_lidar/lidar_points, adds the
intensity field MOLA requires, rewrites frame_id and republishes on
/carter/lidar_with_intensity. The x, y, z offsets are read from the message
fields, not assumed.

Isaac Sim publishes one LiDAR revolution as several partial messages, one per
angular sector; the node merges them into one cloud per revolution, either by a
fixed count (--sub-scans-per-cycle N) or by a time window. A timestamp
monotonicity filter drops messages from a stale Isaac session still running after
world.reset() and detects the backward jump of a reset. The per-point 't' field is
not available (simultaneous GPU raycasting) and is not synthesised. With
--count-file the number of published scans is written to a file for the
orchestrator; statistics go to the [DIAG] log lines.

Usage:
    python3 src/nodes/add_intensity_node.py --input-topic /front_3d_lidar/lidar_points \\
        --sub-scans-per-cycle 1 --output-frame base_link
"""

import argparse
import sys
from collections import Counter

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs_py import point_cloud2
import numpy as np
import struct

MAX_FORWARD_JUMP = 30.0   # [s] forward jump beyond this: stale Isaac session
MAX_BACKWARD_RESET = 10.0 # [s] backward jump beyond this: Isaac Sim reset

# Sub-scan aggregation window, s of sim time. One RTX LiDAR revolution takes
# ~100 ms; the margin covers the delay of the last sub-scan.
AGGREGATION_WINDOW_SEC = 0.120

# Fixed-count aggregation, alternative to the time window. With the window, the
# number of sub-scans per cloud depends on arrival times: a sub-scan past the edge
# drops a sector, so MOLA gets different input in two identical runs, and the
# recursive filters of the pipeline (adaptive_threshold) carry that difference for
# the whole run. A fixed count gives every cloud the same number of sectors. Use the
# mode of the [DIAG] SUB-SCAN histogram printed in window mode. 0 = time window.
SUB_SCANS_PER_CYCLE = 0

DEFAULT_INPUT_TOPIC = '/carter/lidar_fix'
DEFAULT_OUTPUT_TOPIC = '/carter/lidar_with_intensity'

# Output frame_id: must be the frame in which the LiDAR pose (LIDAR_POSE_*) passed to
# MOLA by the launch script is expressed.
DEFAULT_OUTPUT_FRAME = 'chassis_link'

# PointField.datatype -> (numpy dtype, size in bytes)
_PFTYPE_TO_NPTYPE = {
    PointField.INT8:    (np.int8,    1),
    PointField.UINT8:   (np.uint8,   1),
    PointField.INT16:   (np.int16,   2),
    PointField.UINT16:  (np.uint16,  2),
    PointField.INT32:   (np.int32,   4),
    PointField.UINT32:  (np.uint32,  4),
    PointField.FLOAT32: (np.float32, 4),
    PointField.FLOAT64: (np.float64, 8),
}


def _get_field(msg, name):
    """Find a PointField by name; returns (offset, numpy dtype, bytes) or None."""
    for f in msg.fields:
        if f.name == name:
            entry = _PFTYPE_TO_NPTYPE.get(f.datatype)
            if entry is None:
                return None
            return (f.offset, entry[0], entry[1])
    return None


class AddIntensityNode(Node):
    def __init__(self, sub_scans_per_cycle: int = SUB_SCANS_PER_CYCLE,
                 input_topic: str = DEFAULT_INPUT_TOPIC,
                 output_topic: str = DEFAULT_OUTPUT_TOPIC,
                 output_frame: str = DEFAULT_OUTPUT_FRAME,
                 count_file: str = ""):
        super().__init__('add_intensity_node')

        self._count_file = count_file or None
        self._input_topic = input_topic
        self._output_topic = output_topic
        self._output_frame = output_frame
        self._sub_scans_per_cycle = sub_scans_per_cycle
        self._cycle_hist = Counter()   # sub-scans per published cycle
        self._last_accepted_stamp: float | None = None
        self._recv_count = 0
        self._pub_count = 0
        self._drop_forward = 0
        self._drop_empty = 0
        self._sub_scan_count = 0
        self._fields_logged = False

        # Field layout, read from the first message.
        self._x_offset = None
        self._y_offset = None
        self._z_offset = None
        self._field_dtype = None  # common dtype of x, y, z
        self._field_size = None   # bytes per component

        # Aggregation buffer of the current cycle.
        self._agg_points = []          # one np.array(N,3) per sub-scan
        self._agg_window_start = None  # sim time of the first sub-scan
        self._agg_header = None        # header of the first sub-scan, reused on output

        self.subscription = self.create_subscription(
            PointCloud2,
            self._input_topic,
            self.lidar_callback,
            10
        )
        self.publisher = self.create_publisher(
            PointCloud2,
            self._output_topic,
            10
        )

        # Periodic [DIAG] statistics.
        self.create_timer(10.0, self._log_stats)

        self.get_logger().info('Add Intensity Node started (with scan aggregation)')
        if self._sub_scans_per_cycle > 0:
            self.get_logger().info(
                f'Aggregation mode: FIXED COUNT ({self._sub_scans_per_cycle} sub-scans/cycle)'
            )
        else:
            self.get_logger().info(
                f'Aggregation mode: TIME WINDOW ({AGGREGATION_WINDOW_SEC*1000:.0f}ms) '
                f'- non deterministica ai bordi, usa --sub-scans-per-cycle'
            )
        self.get_logger().info(f'Subscribing to: {self._input_topic}')
        self.get_logger().info(f'Publishing to: {self._output_topic}')
        self.get_logger().info(f'Output frame_id: {self._output_frame}')

    def _log_stats(self):
        self.get_logger().info(
            f'[DIAG] recv={self._recv_count} pub={self._pub_count} '
            f'drop_fwd={self._drop_forward} drop_empty={self._drop_empty} '
            f'sub_scans_in_buf={len(self._agg_points)} '
            f'last_t={self._last_accepted_stamp}'
        )
        if self._cycle_hist:
            # A spread distribution means the time window groups sectors
            # inconsistently: the mode is the value for --sub-scans-per-cycle.
            items = sorted(self._cycle_hist.items())
            total = sum(self._cycle_hist.values())
            dist = '  '.join(f'{k}:{v} ({100.0*v/total:.0f}%)' for k, v in items)
            modal = self._cycle_hist.most_common(1)[0][0]
            self.get_logger().info(
                f'[DIAG] SUB-SCAN per ciclo -> {dist}   | modale={modal}'
            )

    def _cache_field_layout(self, msg):
        """Read and cache the x, y, z offsets from the message."""
        fx = _get_field(msg, 'x')
        fy = _get_field(msg, 'y')
        fz = _get_field(msg, 'z')

        if fx is None or fy is None or fz is None:
            self.get_logger().error(
                f'PointCloud2 missing x/y/z fields! '
                f'Available fields: {[f.name for f in msg.fields]}'
            )
            return False

        if not (fx[1] == fy[1] == fz[1]):
            self.get_logger().error(
                f'x/y/z fields have different types: '
                f'x={fx[1]}, y={fy[1]}, z={fz[1]}'
            )
            return False

        self._x_offset = fx[0]
        self._y_offset = fy[0]
        self._z_offset = fz[0]
        self._field_dtype = fx[1]
        self._field_size = fx[2]

        self.get_logger().info(
            f'[DIAG] Field layout cached: '
            f'x@{self._x_offset} y@{self._y_offset} z@{self._z_offset} '
            f'dtype={self._field_dtype.__name__} size={self._field_size}B '
            f'point_step={msg.point_step}'
        )
        return True

    def _extract_valid_points(self, msg):
        """Extract x, y, z, dropping NaN and origin points; np.array(N,3) or None."""
        raw = np.frombuffer(msg.data, dtype=np.uint8).reshape(-1, msg.point_step)
        if raw.shape[0] == 0:
            return None

        sz = self._field_size
        x_vals = raw[:, self._x_offset:self._x_offset + sz].view(self._field_dtype).ravel()
        y_vals = raw[:, self._y_offset:self._y_offset + sz].view(self._field_dtype).ravel()
        z_vals = raw[:, self._z_offset:self._z_offset + sz].view(self._field_dtype).ravel()

        xyz = np.column_stack([x_vals, y_vals, z_vals]).astype(np.float32)

        # Points at the origin are Isaac Sim padding.
        valid = np.isfinite(xyz).all(axis=1) & (np.abs(xyz).sum(axis=1) > 1e-6)
        points = xyz[valid]

        return points if len(points) > 0 else None

    def _publish_aggregated(self):
        """Publish the aggregated buffer as a single PointCloud2."""
        if not self._agg_points:
            return

        all_points = np.vstack(self._agg_points)
        n_total = len(all_points)
        n_sub = len(self._agg_points)
        self._cycle_hist[n_sub] += 1

        if n_total == 0:
            self._agg_points.clear()
            self._agg_window_start = None
            self._agg_header = None
            return

        # Synthetic intensity, decreasing with range.
        distances = np.linalg.norm(all_points, axis=1)
        intensities = 100.0 / (1.0 + distances)

        centroid = all_points.mean(axis=0)
        if self._pub_count < 10 or self._pub_count % 50 == 0:
            zmin, zmax = all_points[:, 2].min(), all_points[:, 2].max()
            self.get_logger().info(
                f'[DIAG] agg_scan#{self._pub_count}: '
                f't={self._agg_window_start:.3f} '
                f'sub_scans={n_sub} total_pts={n_total} '
                f'z=[{zmin:.2f},{zmax:.2f}] '
                f'dist=[{distances.min():.2f},{distances.max():.2f}] '
                f'centroid=({centroid[0]:.3f},{centroid[1]:.3f},{centroid[2]:.3f})'
            )

        self._pub_count += 1

        new_points = np.column_stack([all_points, intensities]).tolist()

        fields = [
            PointField(name='x',         offset=0,  datatype=PointField.FLOAT32, count=1),
            PointField(name='y',         offset=4,  datatype=PointField.FLOAT32, count=1),
            PointField(name='z',         offset=8,  datatype=PointField.FLOAT32, count=1),
            PointField(name='intensity', offset=12, datatype=PointField.FLOAT32, count=1),
        ]

        new_msg = point_cloud2.create_cloud(self._agg_header, fields, new_points)
        new_msg.header.frame_id = self._output_frame
        self.publisher.publish(new_msg)

        self._agg_points.clear()
        self._agg_window_start = None
        self._agg_header = None

        if self._count_file is not None:
            # The node starts before MOLA and stops after it, so the raw total vs
            # MOLA poses would give a false drop rate. The orchestrator reads the
            # file at the start and end of the MOLA window and uses the delta.
            try:
                with open(self._count_file, 'w') as f:
                    f.write(str(self._pub_count))
            except OSError:
                pass

    def lidar_callback(self, msg):
        self._recv_count += 1
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        # Log the PointCloud2 layout once.
        if not self._fields_logged:
            self._fields_logged = True
            field_info = [(f.name, f.offset, f.datatype, f.count) for f in msg.fields]
            self.get_logger().info(
                f'[DIAG] First msg: point_step={msg.point_step} '
                f'width={msg.width} height={msg.height} '
                f'is_dense={msg.is_dense} frame={msg.header.frame_id} '
                f'stamp={stamp:.3f}s fields={field_info}'
            )

        if self._x_offset is None:
            if not self._cache_field_layout(msg):
                return

        # Timestamp monotonicity filter.
        if self._last_accepted_stamp is not None:
            dt = stamp - self._last_accepted_stamp

            if dt > MAX_FORWARD_JUMP:
                self._drop_forward += 1
                if self._drop_forward <= 5:
                    self.get_logger().warn(
                        f'[DIAG] DROP fwd jump: dt={dt:.2f}s '
                        f'last={self._last_accepted_stamp:.2f} cur={stamp:.2f}'
                    )
                return

            if dt < -MAX_BACKWARD_RESET:
                self.get_logger().info(
                    f'Session reset: t {self._last_accepted_stamp:.2f}s → {stamp:.2f}s'
                )
                # Buffered points belong to the previous session.
                self._agg_points.clear()
                self._agg_window_start = None
                self._agg_header = None

        self._last_accepted_stamp = stamp

        points = self._extract_valid_points(msg)

        if points is None:
            self._drop_empty += 1
            if self._drop_empty <= 5:
                n_raw = msg.width * msg.height
                self.get_logger().warn(
                    f'[DIAG] DROP empty: n_raw={n_raw} all NaN/zero t={stamp:.2f}'
                )
            return

        if self._sub_scans_per_cycle > 0:
            # Fixed count: publish as soon as N sub-scans are in, not when the
            # first sub-scan of the next cycle arrives, so no window delay.
            if self._agg_window_start is None:
                self._agg_window_start = stamp
                self._agg_header = msg.header
            self._agg_points.append(points)

            if len(self._agg_points) >= self._sub_scans_per_cycle:
                self._publish_aggregated()
            return

        # Time window.
        if self._agg_window_start is None:
            self._agg_window_start = stamp
            self._agg_header = msg.header
            self._agg_points.append(points)
        elif stamp - self._agg_window_start > AGGREGATION_WINDOW_SEC:
            # Window expired: publish and restart from the current sub-scan.
            self._publish_aggregated()

            self._agg_window_start = stamp
            self._agg_header = msg.header
            self._agg_points.append(points)
        else:
            self._agg_points.append(points)

    def destroy_node(self):
        """Publish the remaining buffer before shutdown."""
        self._publish_aggregated()
        super().destroy_node()


def main(args=None):
    parser = argparse.ArgumentParser(
        description="Aggrega i sub-scan del RTX LiDAR di Isaac Sim e aggiunge intensity."
    )
    parser.add_argument(
        "--sub-scans-per-cycle",
        type=int,
        default=SUB_SCANS_PER_CYCLE,
        help=(
            "Se > 0, aggrega esattamente N sub-scan per nuvola invece di usare la "
            "finestra temporale da %dms. Elimina la variabilita' di raggruppamento "
            "ai bordi della finestra. Per scoprire N: lancia il nodo senza questo "
            "flag e leggi l'istogramma '[DIAG] SUB-SCAN per ciclo'."
            % int(AGGREGATION_WINDOW_SEC * 1000)
        ),
    )
    parser.add_argument(
        "--input-topic",
        type=str,
        default=DEFAULT_INPUT_TOPIC,
        help="Topic PointCloud2 di ingresso.",
    )
    parser.add_argument(
        "--output-topic",
        type=str,
        default=DEFAULT_OUTPUT_TOPIC,
        help="Topic PointCloud2 di uscita (quello letto da MOLA).",
    )
    parser.add_argument(
        "--output-frame",
        type=str,
        default=DEFAULT_OUTPUT_FRAME,
        help="frame_id applicato alla nuvola in uscita.",
    )
    parser.add_argument(
        "--count-file",
        type=str,
        default="",
        help=(
            "Se indicato, scrive il numero di scan pubblicati in questo file ad "
            "ogni pubblicazione. Permette all'orchestratore di misurare quanti "
            "scan sono stati emessi esattamente nella finestra in cui MOLA era "
            "attivo."
        ),
    )
    cli_args, ros_args = parser.parse_known_args(sys.argv[1:] if args is None else args)

    rclpy.init(args=ros_args)
    node = AddIntensityNode(
        sub_scans_per_cycle=cli_args.sub_scans_per_cycle,
        input_topic=cli_args.input_topic,
        output_topic=cli_args.output_topic,
        output_frame=cli_args.output_frame,
        count_file=cli_args.count_file,
    )
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
