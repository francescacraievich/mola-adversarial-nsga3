#!/usr/bin/env python3
"""
Extract point cloud frames AND tf transforms from a ROS2 bag file.

This extracts both LiDAR frames and associated transforms for use in NSGA-III optimization.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore


def extract_pointcloud2_to_numpy(msg):
    """Convert PointCloud2 message to numpy array."""
    import struct

    fields = {f.name: (f.offset, f.datatype) for f in msg.fields}
    point_step = msg.point_step
    data = bytes(msg.data)
    n_points = msg.width * msg.height
    points = []

    for i in range(n_points):
        offset = i * point_step
        x = struct.unpack_from("f", data, offset + fields["x"][0])[0]
        y = struct.unpack_from("f", data, offset + fields["y"][0])[0]
        z = struct.unpack_from("f", data, offset + fields["z"][0])[0]

        if "intensity" in fields:
            intensity = struct.unpack_from("f", data, offset + fields["intensity"][0])[0]
        else:
            intensity = 1.0

        if np.isfinite(x) and np.isfinite(y) and np.isfinite(z):
            points.append([x, y, z, intensity])

    return np.array(points, dtype=np.float32)


def extract_tf_to_dict(msg, timestamp):
    """Extract transforms from TFMessage to dictionary format."""
    transforms = []
    for t in msg.transforms:
        tf_data = {
            "timestamp": timestamp,
            "frame_id": t.header.frame_id,
            "child_frame_id": t.child_frame_id,
            "translation": [
                t.transform.translation.x,
                t.transform.translation.y,
                t.transform.translation.z,
            ],
            "rotation": [
                t.transform.rotation.x,
                t.transform.rotation.y,
                t.transform.rotation.z,
                t.transform.rotation.w,
            ],
        }
        transforms.append(tf_data)
    return transforms


def _extract_lidar_frames(reader, lidar_topic):
    """Extract LiDAR frames from bag."""
    frames = []
    frame_timestamps = []
    lidar_connections = [c for c in reader.connections if c.topic == lidar_topic]
    if lidar_connections:
        print(f"\nExtracting LiDAR frames from {lidar_topic}...")
        for connection, timestamp, rawdata in reader.messages(connections=lidar_connections):
            msg = reader.deserialize(rawdata, connection.msgtype)
            cloud = extract_pointcloud2_to_numpy(msg)
            if len(cloud) > 0:
                frames.append(cloud)
                frame_timestamps.append(timestamp)
                if len(frames) % 20 == 0:
                    print(f"  Extracted {len(frames)} frames...")
        print(f"  Total: {len(frames)} frames")
    return frames, frame_timestamps


def _extract_tf_messages(reader, topic_name, is_static=False):
    """Extract TF messages from bag."""
    tf_messages = []
    connections = [c for c in reader.connections if c.topic == topic_name]
    if connections:
        tf_type = "static transforms" if is_static else "transforms"
        print(f"\nExtracting {topic_name} {tf_type}...")
        for connection, timestamp, rawdata in reader.messages(connections=connections):
            msg = reader.deserialize(rawdata, connection.msgtype)
            transforms = extract_tf_to_dict(msg, timestamp)
            tf_messages.extend(transforms)
        print(f"  Total: {len(tf_messages)} {tf_type}")
    return tf_messages


def _save_outputs(frames, frame_timestamps, tf_messages, tf_static_messages, args):
    """Save extracted data to files."""
    output_frames = Path(args.output_frames)
    output_frames.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_frames, np.array(frames, dtype=object), allow_pickle=True)
    print(f"\nSaved frames to: {output_frames}")

    timestamps_path = output_frames.with_suffix(".timestamps.npy")
    np.save(timestamps_path, np.array(frame_timestamps))
    print(f"Saved timestamps to: {timestamps_path}")

    output_tf = Path(args.output_tf)
    np.save(output_tf, np.array(tf_messages, dtype=object), allow_pickle=True)
    print(f"Saved tf to: {output_tf}")

    output_tf_static = Path(args.output_tf_static)
    np.save(output_tf_static, np.array(tf_static_messages, dtype=object), allow_pickle=True)
    print(f"Saved tf_static to: {output_tf_static}")


def main():
    """Main function to extract point clouds and TF from ROS2 bag."""
    parser = argparse.ArgumentParser(description="Extract point cloud frames and TF from ROS2 bag")
    parser.add_argument("--bag", type=str, required=True, help="Path to bag directory")
    parser.add_argument("--lidar-topic", type=str, default="/carter/lidar_with_intensity")
    parser.add_argument("--output-frames", type=str, default="data/frame_sequence.npy")
    parser.add_argument("--output-tf", type=str, default="data/tf_sequence.npy")
    parser.add_argument("--output-tf-static", type=str, default="data/tf_static.npy")

    args = parser.parse_args()

    print("\n" + "=" * 60)
    print(" EXTRACT POINT CLOUDS AND TF FROM BAG")
    print("=" * 60 + "\n")

    bag_path = Path(args.bag)
    if not bag_path.exists():
        print(f"Error: Bag not found: {bag_path}")
        return 1

    typestore = get_typestore(Stores.ROS2_JAZZY)

    with AnyReader([bag_path], default_typestore=typestore) as reader:
        print("Topics in bag:")
        for c in reader.connections:
            print(f"  - {c.topic} ({c.msgtype})")

        frames, frame_timestamps = _extract_lidar_frames(reader, args.lidar_topic)
        tf_messages = _extract_tf_messages(reader, "/tf", is_static=False)
        tf_static_messages = _extract_tf_messages(reader, "/tf_static", is_static=True)

    _save_outputs(frames, frame_timestamps, tf_messages, tf_static_messages, args)

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
