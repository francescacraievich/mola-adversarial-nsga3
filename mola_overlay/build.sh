#!/usr/bin/env bash
# Builds the patched mola_lidar_odometry 3.2.0 and mola_state_estimation_simple 2.4.2 into
# $MOLA_WS/install_G1, with the compiler flags of the Debian packages, so that the only
# difference from the apt binaries is the patch.
set -eo pipefail
WS="${MOLA_WS:-$HOME/mola_ws}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source /opt/ros/jazzy/setup.bash
mkdir -p "$WS/src_G1" "$WS/upstream"
cd "$WS/upstream"
[[ -d mola_lidar_odometry ]] || git clone -q --depth 1 --branch 3.2.0 https://github.com/MOLAorg/mola_lidar_odometry.git
[[ -d mola_state_estimation ]] || git clone -q --depth 1 --branch 2.4.2 https://github.com/MOLAorg/mola_state_estimation.git
cd "$WS/src_G1"
if [[ ! -d mola_lidar_odometry ]]; then
  rsync -a --exclude=.git "$WS/upstream/mola_lidar_odometry/" mola_lidar_odometry/
  rsync -a "$WS/upstream/mola_state_estimation/mola_state_estimation_simple/" mola_state_estimation_simple/
  patch -p1 < "$HERE/mola_lidar_odometry-3.2.0-full-state.patch"
  patch -p1 < "$HERE/mola_state_estimation_simple-2.4.2-full-state.patch"
fi
cd "$WS"
# flags of the bloom/debhelper build of the apt packages (file-prefix-map keeps paths out of the binary)
MAP="-ffile-prefix-map=$WS/src_G1/mola_lidar_odometry=. -ffile-prefix-map=$WS/src_G1/mola_state_estimation_simple=."
BASE="-g -O2 -fno-omit-frame-pointer -mno-omit-leaf-frame-pointer -flto=auto -ffat-lto-objects -fstack-protector-strong -fstack-clash-protection -Wformat -Werror=format-security -fcf-protection"
export CXXFLAGS="$BASE $MAP $(dpkg-buildflags --get CPPFLAGS) -DNDEBUG"
export CFLAGS="$BASE $MAP $(dpkg-buildflags --get CPPFLAGS)"
export LDFLAGS=
colcon --log-base log_G1 build --base-paths src_G1 --build-base build_G1 --install-base install_G1 \
  --executor sequential --cmake-args -DCMAKE_BUILD_TYPE=None -DCMAKE_INSTALL_LIBDIR=lib/x86_64-linux-gnu \
  -DCMAKE_EXPORT_NO_PACKAGE_REGISTRY=ON -DCMAKE_FIND_PACKAGE_NO_PACKAGE_REGISTRY=ON -DBUILD_TESTING=OFF
