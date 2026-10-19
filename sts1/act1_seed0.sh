#!/bin/bash
# The seed-0 act-1 experiment on TC2 in one command, from the repo root:
#   git pull && bash sts1/act1_seed0.sh           # build, config, collect + check, format check, queue jobs
#   bash sts1/act1_seed0.sh results               # print the results of the queued jobs
# Jobs: train, then (after it succeeds) the accuracy evaluation and act-1 play.
set -e
cd "$(dirname "$0")/.."
PY=~/.conda/envs/gpu-env/bin/python
export HF_HUB_OFFLINE=1 TRANSFORMERS_VERBOSITY=error
JOBS_FILE=runs/act1_seed0.jobs
mkdir -p runs

if [ "$1" = results ]; then
  read -r TRAIN EVAL PLAY < "$JOBS_FILE"
  squeue -u "$USER" -h -o "%i %T" | grep -E "^($TRAIN|$EVAL|$PLAY) " && echo "(still queued or running)" || true
  echo "== train ($TRAIN)"
  grep -o "{'loss'[^}]*}" "runs/sts1-sft-$TRAIN.out" | sed -n '1p;$p' || true
  grep -o "{'eval_loss'[^}]*}\|{'train_runtime'[^}]*}" "runs/sts1-sft-$TRAIN.out" || true
  grep -m3 -E "Error|Traceback" "runs/sts1-sft-$TRAIN.out" || true
  echo "== accuracy on val ($EVAL)"
  tail -5 "runs/sts1-sft-$EVAL.out" 2>/dev/null || echo "no log yet"
  echo "== act 1 play ($PLAY), teacher MCTS+rules won 7/20 (rule seeds 5 6 8 11 | 16 17 19)"
  grep -E "^seed|^summary" "runs/sts1-sft-$PLAY.out" 2>/dev/null || echo "no log yet"
  exit 0
fi

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
EVAL=$(sbatch --parsable --dependency=afterok:$TRAIN sts1/sft.sbatch eval runs/sft-act1_seed0/final 0)
PLAY=$(sbatch --parsable --dependency=afterok:$TRAIN sts1/sft.sbatch play)
echo "$TRAIN $EVAL $PLAY" > "$JOBS_FILE"
echo "== queued: train $TRAIN, then eval $EVAL and play $PLAY"
echo "check later with: bash sts1/act1_seed0.sh results"
