#!/bin/bash
# Builds the PyroWave client and deploys it to the Apple Vision Pro, like deploy_test.sh did for
# JPEG XS. Run it in a GUI Terminal window: code signing needs the unlocked login keychain, which a
# plain SSH session cannot reach.
#
#   bash deploy_pyrowave.sh                 # clone/update to ~/dev/alvr-visionos-pyrowave
#   SKIP_CORE=1 bash deploy_pyrowave.sh     # reuse the client core framework from the last run
#
# Overridable: REPO_DIR, BRANCH, DEVICE_UDID, BUNDLE_ID.
set -euo pipefail

REPO_URL=https://github.com/TobiH-GE/alvr-visionos-pyrowave.git
REPO_DIR=${REPO_DIR:-$HOME/dev/alvr-visionos-pyrowave}
BRANCH=${BRANCH:-main}
: "${DEVICE_UDID:?set DEVICE_UDID, see xcrun devicectl list devices}"
BUNDLE_ID=${BUNDLE_ID:-alvr.client}
# A fixed DerivedData path, so the app is found without guessing Xcode's hashed folder name.
DERIVED_DATA="$REPO_DIR/build/DerivedData"
APP_PATH="$DERIVED_DATA/Build/Products/Release-xros/ALVRClient.app"

echo "=== [1/5] Repository ($BRANCH) ==="
if [ -d "$REPO_DIR/.git" ]; then
    git -C "$REPO_DIR" fetch origin "$BRANCH"
    git -C "$REPO_DIR" checkout "$BRANCH"
    git -C "$REPO_DIR" pull --ff-only origin "$BRANCH"
else
    git clone --branch "$BRANCH" "$REPO_URL" "$REPO_DIR"
fi
cd "$REPO_DIR"
# The ALVR submodule is alvr-client-core-pyrowave-private.
git submodule sync --recursive
git submodule update --init --recursive
echo "app $(git rev-parse --short HEAD), client core $(git -C ALVR rev-parse --short HEAD)"

echo "=== [2/5] Client core framework (Rust, cbindgen) ==="
if [ "${SKIP_CORE:-0}" = "1" ] && [ -d ALVRClient/ALVRClientCore.xcframework ]; then
    echo "skipped (SKIP_CORE=1)"
else
    bash build_and_repack.sh
fi
grep -q ALVR_CODEC_PYRO_WAVE ALVRClient/ALVRClientCore.xcframework/*/ALVRClientCore.framework/Headers/alvr_client_core.h \
    || { echo "the client core header has no ALVR_CODEC_PYRO_WAVE: wrong submodule commit?"; exit 1; }

echo "=== [3/5] Building ALVRClient (Release, visionOS device) ==="
xcodebuild -project ALVRClient.xcodeproj -scheme ALVRClient -configuration Release \
    -destination "generic/platform=visionOS" -derivedDataPath "$DERIVED_DATA" build

echo "=== [4/5] Installing on device ==="
xcrun devicectl device install app --device "$DEVICE_UDID" "$APP_PATH"

echo "=== [5/5] Launching ==="
xcrun devicectl device process launch --device "$DEVICE_UDID" "$BUNDLE_ID"

echo "=== DEPLOY DONE ==="
echo "Log lines starting with 'PyroWave:' come from the decoder. To read the app's stdout live:"
echo "  xcrun devicectl device process launch --console --device $DEVICE_UDID $BUNDLE_ID"
