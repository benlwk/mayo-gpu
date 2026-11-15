#!/usr/bin/env bash
# bash build.sh [round2|round3] [1|2|3|5]      e.g. bash build.sh round3 2
set -e
ROUND=${1:-round3}
VARIANT=${2:-2}
cd "$(dirname "$0")/$ROUND"

cmake -S mayo-c -B mayo-c/build -DCMAKE_BUILD_TYPE=Release -DENABLE_TESTS=OFF -DENABLE_PARAMS_DYNAMIC=ON >/dev/null
cmake --build mayo-c/build -j

make -C coarse clean >/dev/null && make -C coarse VARIANT=$VARIANT -j
make -C fine   clean >/dev/null && make -C fine   VARIANT=$VARIANT -j
echo "built $ROUND/coarse/mayo-coarse and $ROUND/fine/mayo-fine (MAYO_$VARIANT)"
