#!/bin/sh
# Builds the oxinfer C API static library for iOS devices and the Simulator (Apple silicon) and
# packages both as ios/build/Oxinfer.xcframework (with include/oxinfer.h as module `Oxinfer`).
#
# Usage: ios/build-rust.sh
# Uses rustup's toolchain (Homebrew's rustc has no iOS targets): rustup target add
# aarch64-apple-ios aarch64-apple-ios-sim
set -eu
cd "$(dirname "$0")/.."
ROOT=$(pwd)
OUT="$ROOT/ios/build"

# Prefer rustup's cargo/rustc over any other installation on PATH.
if command -v rustup >/dev/null 2>&1; then
	RUSTC=$(rustup which rustc)
	CARGO=$(rustup which cargo)
	PATH="$(dirname "$RUSTC"):$PATH"
	export RUSTC PATH
else
	CARGO=cargo
fi

# Match the app's deployment target so the linker does not warn about newer objects.
export IPHONEOS_DEPLOYMENT_TARGET=17.0

TARGETS="aarch64-apple-ios aarch64-apple-ios-sim"
for t in $TARGETS; do
	echo "== building liboxinfer.a for $t"
	"$CARGO" build-capi --target "$t"
done

rm -rf "$OUT/Oxinfer.xcframework" "$OUT/headers"
mkdir -p "$OUT/headers"
cp "$ROOT/include/oxinfer.h" "$OUT/headers/"
cat >"$OUT/headers/module.modulemap" <<'EOF'
module Oxinfer {
    header "oxinfer.h"
    export *
}
EOF
xcodebuild -create-xcframework \
	-library "$ROOT/target/aarch64-apple-ios/release/liboxinfer.a" -headers "$OUT/headers" \
	-library "$ROOT/target/aarch64-apple-ios-sim/release/liboxinfer.a" -headers "$OUT/headers" \
	-output "$OUT/Oxinfer.xcframework"
echo "== $OUT/Oxinfer.xcframework"
