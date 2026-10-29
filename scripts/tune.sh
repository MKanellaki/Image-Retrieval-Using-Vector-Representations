#!/usr/bin/env bash
# Runs ./search once per parameter set and appends one CSV row per run.
#
# usage: scripts/tune.sh <method> "<params>" ["<params>" ...]
#   e.g. scripts/tune.sh lsh "-k 4 -L 5 -w 0.05" "-k 4 -L 10 -w 0.1"
#
# Environment overrides: SET (validation), D (25000), VOCAB (256), OUT (results/tuning.csv)
set -euo pipefail
export LC_ALL=C   # "." as the decimal point, whatever the system locale

if [ $# -lt 2 ]; then
    echo "usage: $0 <exact|lsh|hypercube|ivfflat|ivfpq> \"<params>\" [\"<params>\" ...]" >&2
    exit 1
fi

method=$1
shift
SET=${SET:-validation}
D=${D:-25000}
VOCAB=${VOCAB:-256}
OUT=${OUT:-results/tuning.csv}

mkdir -p "$(dirname "$OUT")" results/runs
if [ ! -f "$OUT" ]; then
    echo "method,params,set,D,R@5,R@10,mAP@10,avg_ms,lt10,wall_s" > "$OUT"
fi

for params in "$@"; do
    tag=$(echo "$params" | tr -d ' -' | tr '.' 'p')
    name="${method}_${tag:-defaults}"
    name="${name}_${SET}_D${D}"
    log="results/runs/${name}.log"
    start=$(date +%s.%N)
    # shellcheck disable=SC2086  # params must split into separate arguments
    ./search -hp hpatches-sequences-release -mir mirflickr -split split.txt \
        -vocab "$VOCAB" -D "$D" -set "$SET" -o "results/runs/${name}.txt" \
        "-$method" $params 2> "$log"
    wall=$(awk -v a="$start" -v b="$(date +%s.%N)" 'BEGIN { printf "%.1f", b - a }')

    # stderr line: "  all  23 queries  R@5 x  R@10 x  mAP@10 x  avg x ms  (<10 results: n)"
    read -r r5 r10 map avg lt10 < <(grep -E '^\[search\] +all ' "$log" |
        sed -E 's/.*R@5 ([0-9.]+) +R@10 ([0-9.]+) +mAP@10 ([0-9.]+) +avg ([0-9.]+) ms +\(<10 results: ([0-9]+)\).*/\1 \2 \3 \4 \5/')

    printf '%s,"%s",%s,%s,%s,%s,%s,%s,%s,%s\n' \
        "$method" "$params" "$SET" "$D" "$r5" "$r10" "$map" "$avg" "$lt10" "$wall" >> "$OUT"
    printf '%-10s %-28s mAP@10 %s  R@10 %s  avg %s ms  (%ss)\n' \
        "$method" "$params" "$map" "$r10" "$avg" "$wall"
done
