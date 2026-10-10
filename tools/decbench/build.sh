#!/bin/bash
# Builds the PyroWave decoder bench on the Mac (Apple silicon). Run from this directory.
set -e
PW=../../ALVRClient/PyroWave
mkdir -p build
for f in pyrowave_common.mm pyrowave_decoder.mm pyrowave_encoder.mm pyrowave_bitstream.cpp; do
    clang++ -std=c++17 -O2 -fobjc-arc -I$PW -c $PW/$f -o build/${f%.*}.o
done
clang++ -std=c++17 -O2 -fobjc-arc -I$PW decbench.mm build/*.o -framework Metal -framework Foundation -framework IOSurface -framework CoreVideo -o build/decbench
echo built build/decbench
