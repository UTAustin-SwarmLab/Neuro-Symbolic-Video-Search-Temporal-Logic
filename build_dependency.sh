#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(pwd)"
VENDORS_DIR="$REPO_DIR/vendors"
INSTALL_PREFIX="$VENDORS_DIR/install"

mkdir -p "$VENDORS_DIR"
cd "$VENDORS_DIR"

# carl-storm
cd "$VENDORS_DIR"
if [ ! -d "carl-storm" ]; then
  git clone https://github.com/moves-rwth/carl-storm
fi
cmake -S carl-storm -B carl-storm/build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$INSTALL_PREFIX"
cmake --build carl-storm/build -j"$(nproc)" --target lib_carl
cmake --build carl-storm/build --target install

# storm-stable
if [ ! -d "storm-stable" ]; then
  git clone --branch stable --depth 1 --recursive https://github.com/moves-rwth/storm.git storm-stable
fi
cmake -S storm-stable -B storm-stable/build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$INSTALL_PREFIX" \
  -DSTORM_DEVELOPER=OFF \
  -DSTORM_LOG_DISABLE_DEBUG=ON \
  -DSTORM_PORTABLE=ON \
  -DSTORM_USE_SPOT_SHIPPED=ON
cmake --build storm-stable/build -j"$(nproc)"
cmake --build storm-stable/build --target install

export CMAKE_ARGS="-DCMAKE_POLICY_VERSION_MINIMUM=3.5"
export STORM_DIR_HINT="$INSTALL_PREFIX"
export CARL_DIR_HINT="$INSTALL_PREFIX"
unset CMAKE_ARGS || true

