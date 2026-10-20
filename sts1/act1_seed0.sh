#!/bin/bash
# The seed-0 act-1 experiment on TC2 in one command, from the repo root:
#   git pull && bash sts1/act1_seed0.sh           # build, config, collect + check, format check, queue jobs
#   bash sts1/act1_seed0.sh results               # print the results of the queued jobs
# Jobs: train, then (after it succeeds) one job with the accuracy evaluation and act-1 play.
set -e
cd "$(dirname "$0")/.."
PY=~/.conda/envs/gpu-env/bin/python
export HF_HUB_OFFLINE=1 TRANSFORMERS_VERBOSITY=error PYTHONUNBUFFERED=1   # unbuffered: progress shows through tee
JOBS_FILE=runs/act1_seed0.jobs
mkdir -p runs

if [ "$1" = results ]; then
  [ -f "$JOBS_FILE" ] || { echo "no jobs queued yet: run  bash sts1/act1_seed0.sh  first"; exit 1; }
  read -r TRAIN AFTER < "$JOBS_FILE"
  squeue -u "$USER" -h -o "%i %T" | grep -E "^($TRAIN|$AFTER) " && echo "(still queued or running)" || true
  echo "== train ($TRAIN)"
  grep -o "{'loss'[^}]*}" "runs/sts1-sft-$TRAIN.out" | sed -n '1p;$p' || true
  grep -o "{'eval_loss'[^}]*}\|{'train_runtime'[^}]*}" "runs/sts1-sft-$TRAIN.out" || true
  grep -m3 -E "Error|Traceback" "runs/sts1-sft-$TRAIN.out" || true
  LOG="runs/sts1-sft-$AFTER.out"
  [ -f "$LOG" ] || { echo "== eval + play ($AFTER): no log yet"; exit 0; }
  echo "== accuracy on val ($AFTER)"
  grep -E "^(group|all|not forced)" "$LOG" || true
  echo "== act 1 play ($AFTER), teacher MCTS+rules won 7/20 (rule seeds 5 6 8 11 | 16 17 19)"
  grep -E "^seed|^summary" "$LOG" || true
  grep -m3 -E "Error|Traceback|Fatal" "$LOG" || true
  exit 0
fi

# QoS: at most 2 submitted jobs per user (queued + running), so eval and play share one job
squeue -u "$USER" -h | grep -q . && { echo "STOP: you already have jobs in the queue (squeue -u $USER), need 2 free slots"; exit 1; }

echo "== build"
PY=$PY JOBS=16 ./build.sh > runs/build.log 2>&1 || { tail -20 runs/build.log; exit 1; }

echo "== config"
$PY - <<'EOF'
import json
p = "sts1/config.json"; c = json.load(open(p))
c["collect"].update(mode="act1", run_seed=0, games=20, seed_start=0, split=[6, 2, 2], out_dir="data")
c["play"].update(mode="act1", run_seed=0, games=20)
c["fight"].update(seed=0, ascension=0)
c["llm"].update(base_url="", model="Qwen/Qwen3-4B-Instruct-2507",
                adapter="runs/sft-act1_seed0/final", temperature=0.0, max_tokens=512)
json.dump(c, open(p, "w"), indent=2)
EOF

echo "== collect"
$PY sts1/collect.py | tee runs/collect_act1_seed0.log
# the same numbers as on Windows, else the Linux simulator behaves differently: stop
for expected in "1344 records" "384 records" "366 records"; do
  grep -q "$expected" runs/collect_act1_seed0.log || { echo "STOP: expected '$expected'"; exit 1; }
done

echo "== format check"
$PY sts1/sft.py show | tail -4

TRAIN=$(sbatch --parsable sts1/sft.sbatch)
AFTER=$(sbatch --parsable --dependency=afterok:$TRAIN sts1/sft.sbatch evalplay runs/sft-act1_seed0/final)
echo "$TRAIN $AFTER" > "$JOBS_FILE"
echo "== queued: train $TRAIN, then eval + play $AFTER"
echo "check later with: bash sts1/act1_seed0.sh results"
