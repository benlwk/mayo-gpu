# MAYO on GPU
This is the code for the paper "Fast and Constant-Time MAYO Signatures on GPUs with Timing Leakage and Parallel Granularity Analysis"
CUDA signing and verification for the MAYO post-quantum signature scheme, all four parameter sets.

- `coarse/`: one thread per signature, bit-sliced GF(16).
- `fine/`: one thread per output vector, warp-ballot Gaussian elimination.
- `round3/fine/` also holds the Tensor-Core path (see below).

`round2/` follows the NIST Round 2 parameters, `round3/` the Round 3 ones. All signers run in constant time.

## Requirements

Linux or WSL2, CUDA 12+, CMake 3.5+, gcc. The Tensor-Core path needs an sm_80 or newer GPU.

## Build

```sh
bash build.sh round3 2       # round (default round3), parameter set (1, 2, 3 or 5)
```

Or step by step:

```sh
cd round3
cmake -S mayo-c -B mayo-c/build -DCMAKE_BUILD_TYPE=Release -DENABLE_TESTS=OFF -DENABLE_PARAMS_DYNAMIC=ON
cmake --build mayo-c/build -j
make -C coarse VARIANT=2                 # DESIGN=nibble for the nibble-sliced version
make -C fine   VARIANT=2 BATCH=4096      # BATCH = signatures per launch
```

`ARCH` defaults to `native`. Set it (e.g. `ARCH=sm_80`) when building for another GPU.

## Run

```sh
cd coarse && ./mayo-coarse                      # checks against MAYO-C, then benchmarks
MAYO_BENCH_BATCHES=1024,16384 ./mayo-coarse     # pick the batch sizes
cd fine && ./mayo-fine                          # sign a batch, check it on the CPU, benchmark
cd fine && ./mayo-fine --verify                 # sign a batch, check it, benchmark verify
```

Options (environment variables):

- `MAYO_DISTINCT_MSGS=1`: a different message in every slot (default: one message for all).
- `MAYO_BENCH_STREAM=1` (coarse): failed signatures join the next batch instead of extra launches.
- `MAYO_BENCH_INCLUDE_COPIES=1`: time the host-device copies too.

## Tensor-Core path (round3/fine)

The Tensor-Core path reimplements the design of Choi et al. ([ePrint 2026/2577](https://eprint.iacr.org/2026/2577)) and adds our improvements. The fastest settings:

```sh
cd round3 && make -C fine clean && make -C fine VARIANT=2 BATCH=65536
cd fine
SIGN="MAYO_CHOI_SIGN=3 MAYO_CHOI_SOLVE=5 MAYO_LAMBDA_BSR=1 MAYO_SHAKE_T=1 MAYO_BUILD_S2=1 \
      MAYO_LAMBDA_FOLD2=3 MAYO_RHS_W=1 MAYO_SIGN_FUSERHS=1"
VERIFY="MAYO_CHOI_VERIFY=1 MAYO_LAMBDA_BSR=1 MAYO_SHAKE_T=1 MAYO_RHS_W=1 \
        MAYO_VERIFY_FUSEFOLD=1 MAYO_VERIFY_FUSERHS=1"
env MAYO_DISTINCT_MSGS=1 $SIGN ./mayo-fine
env MAYO_DISTINCT_MSGS=1 $VERIFY ./mayo-fine --verify 10
```

Add `MAYO_TIME_KEYPREP=1` to time the key preparation in every batch, as Choi et al. do.

Start of the ablation (our reimplementation of Choi et al.):

```sh
env MAYO_DISTINCT_MSGS=1 MAYO_CHOI_SIGN=1 MAYO_CHOI_SOLVE=1 MAYO_LAMBDA_BSR=1 ./mayo-fine
env MAYO_DISTINCT_MSGS=1 MAYO_CHOI_VERIFY=1 MAYO_LAMBDA_BSR=1 ./mayo-fine --verify 10
```

Then add the switches one at a time:

| Switch | Effect |
|---|---|
| `MAYO_SHAKE_T=1` | one thread per SHAKE instance, state in registers |
| `MAYO_BUILD_S2=1` | s = v + Ox from bit-sliced dot products |
| `MAYO_LAMBDA_FOLD2=3` | bit-sliced lambda fold in signing |
| `MAYO_CHOI_SIGN=3` | V P1 V^T fused into the Tensor-Core kernel |
| `MAYO_CHOI_SOLVE=3` | the block solver builds A and the right-hand side itself |
| `MAYO_CHOI_SOLVE=5` | warp solver: one warp per signature |
| `MAYO_RHS_W=1` | right-hand side y with one warp per signature |
| `MAYO_SIGN_FUSERHS=1` | the public map in signing accumulates y directly |
| `MAYO_VERIFY_FUSEFOLD=1` | verification: lambda in bit planes, folded inside the fused kernel |
| `MAYO_VERIFY_FUSERHS=1` | verification: the fused kernel also computes y |

Most switches also take a compare value (usually 2) that runs both versions once and prints whether they match.

`mayo-c/` is the reference [MAYO-C](https://github.com/PQCMayo/MAYO-C) library (Apache-2.0). It is used for key generation and CPU checks.
