#!/bin/sh
# Compact, readable launcher for synthetic-dataset generation + CSV export.

# ---------- config ----------
WINDOW=30
SCRIPT_GEN="python ./src/data/synthesize_data.py"
SCRIPT_CSV="python ./src/data/utils/generate_csv.py ./all_data/synthetic"
# ----------------------------

gen(){
  noisy=""
  case "${1:-}" in
    noisy) noisy="--add_noise"; shift ;;
  esac
  pattern=$1; split=$2; seed=$3
  dir="all_data/synthetic/${pattern}/${split}"
  printf '▶ %s/%s  seed=%s  %s\n' "$pattern" "$split" "$seed" "${noisy#--}"
  # build the argument list
  set -- \
     --generate \
     ${noisy} \
     --pattern-type "$pattern" \
     --split "$split" \
     --data_dir "$dir" \
     --synthetic_func "synthetic_${pattern#noisy-}_anomalies" \
     --seed "$seed" \
     --window "$WINDOW"
  # run
  $SCRIPT_GEN "$@"
}

# ---------- clean data ----------
for p in range point freq trend ; do
  gen "$p" eval  42
  gen "$p" train 3047
done

# ---------- noisy data ----------
for p in noisy-point noisy-freq noisy-trend ; do
  gen noisy "$p" eval  42
  gen noisy "$p" train 3047
done

# ---------- CSV export ----------
$SCRIPT_CSV

echo "Done."
