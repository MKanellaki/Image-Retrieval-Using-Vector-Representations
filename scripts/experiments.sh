#!/usr/bin/env bash
# Every run of the experimental study, one phase at a time (uses scripts/tune.sh).
#
# usage: scripts/experiments.sh <phase>
#   exact | lsh | hypercube | ivfflat | ivfpq   tuning grids on the validation set, D = 25000
#   repeat                                      chosen values 3 times on validation (stable times)
#   test                                        chosen values on the test set, D = 1000 5000 10000 25000
#
# Results: results/tune_<method>.csv, results/repeat.csv, results/test.csv
# Per-run output files and stderr logs (illumination / viewpoint summaries): results/runs/
set -euo pipefail
cd "$(dirname "$0")/.."

# ---- Chosen values: fill in after each tuning phase, before `repeat` and `test` ----
LSH_BEST="-k 8 -L 20 -w 0.1"
HYPERCUBE_BEST="-kproj 14 -w 0.1 -M 5000 -probes 1000"
IVFFLAT_BEST="-kclusters 1000 -nprobe 20"
IVFPQ_COARSE="-kclusters 1000 -nprobe 20"   # copy kclusters / nprobe from IVFFLAT_BEST before the ivfpq phase
IVFPQ_BEST="$IVFPQ_COARSE -M 128 -nbits 6"

tune() {   # tune <csv> <method> "<params>"...
    local csv=$1
    shift
    OUT="results/$csv" scripts/tune.sh "$@"
}

case "${1:-}" in
exact)
    tune tune_exact.csv exact ""
    ;;

lsh)   # 4 x 4 x 4 = 64 runs, under a second each
    for w in 0.05 0.1 0.2 0.3; do
        for k in 2 4 6 8; do
            for L in 5 10 15 20; do
                tune tune_lsh.csv lsh "-k $k -L $L -w $w"
            done
        done
    done
    ;;

hypercube)   # 4 x 4 x 5 = 80 runs, about a second each
    # Needs the shuffled database (shuffle_database in main): with the rows in their original
    # order, M cut off the buckets after the HPatches rows and the results were inflated.
    for kproj in 10 12 14 16; do
        for w in 0.03 0.05 0.1 0.15; do
            # M and probes grow together, otherwise the smaller one alone stops the search
            for mp in "1000 100" "2000 300" "5000 1000" "10000 2000" "20000 5000"; do
                set -- $mp
                tune tune_hypercube.csv hypercube "-kproj $kproj -w $w -M $1 -probes $2"
            done
        done
    done
    ;;

ivfflat)   # 5 x 6 = 30 runs, ~8 s each
    for kc in 50 100 159 300 500; do
        for np in 1 2 5 10 20 40; do
            tune tune_ivfflat.csv ivfflat "-kclusters $kc -nprobe $np"
        done
    done
    ;;

ivfpq)   # 4 x 3 = 12 runs, 20 s to a few minutes each (nbits 10 is the slowest)
    for M in 8 16 32 64; do
        for nbits in 6 8 10; do
            tune tune_ivfpq.csv ivfpq "$IVFPQ_COARSE -M $M -nbits $nbits"
        done
    done
    ;;

repeat)   # run the chosen values 3 times; report the average time
    for i in 1 2 3; do
        tune repeat.csv exact ""
        tune repeat.csv lsh "$LSH_BEST"
        tune repeat.csv hypercube "$HYPERCUBE_BEST"
        tune repeat.csv ivfflat "$IVFFLAT_BEST"
        tune repeat.csv ivfpq "$IVFPQ_BEST"
    done
    ;;

test)   # final numbers, run once
    for D in 1000 5000 10000 25000; do
        export SET=test D
        tune test.csv exact ""
        tune test.csv lsh "$LSH_BEST"
        tune test.csv hypercube "$HYPERCUBE_BEST"
        tune test.csv ivfflat "$IVFFLAT_BEST"
        tune test.csv ivfpq "$IVFPQ_BEST"
    done
    ;;

*)
    sed -n '2,10p' "$0" | sed 's/^# \{0,1\}//'
    exit 1
    ;;
esac
