#!/bin/bash
# The act-1 experiment on TC2 in one command, from the repo root:
#   git pull && bash sts1/act1.sh        # build, config, collect, format check, queue the job chain
#   bash sts1/act1.sh results            # print what the chain has produced so far
# Data: act 1 of runs 1-1000 (run s with rule seed s: a new map every game), split 6:2:2 by run,
# so train 1-600, val 601-800, test 801-1000. The chain (sft.sbatch chain) trains 1 epoch,
# resubmitting itself past the 6 h limit, then evaluates (all of val + 2000 train records) and
# plays the 200 test runs.
# Before (git history): act1_seed0.sh, the same on the 20 rule seeds of run 0.
set -e
cd "$(dirname "$0")/.."
PY=~/.conda/envs/gpu-env/bin/python
export HF_HUB_OFFLINE=1 TRANSFORMERS_VERBOSITY=error PYTHONUNBUFFERED=1   # unbuffered: progress shows through tee
DATA=act1_1000runs
JOBS_FILE=runs/$DATA.jobs
mkdir -p runs

if [ "$1" = results ]; then
  [ -f "$JOBS_FILE" ] || { echo "nothing queued yet: run  bash sts1/act1.sh  first"; exit 1; }
  squeue -u "$USER" -h -o "%i %T %M" | sed 's/^/queue: /'
  LOGS=$(find runs -name 'sts1-sft-*.out' -newer "$JOBS_FILE" | sort)
  echo "== teacher (MCTS + rules) on the same runs, from collect"
  grep -E "^(train|val|test) " runs/collect_$DATA.log || true
  echo "== training logs: $(echo $LOGS)"
  for f in $LOGS; do grep -q "{'loss'" "$f" || continue
    echo "-- $f: first/last loss, every validation loss"
    grep -o "{'loss'[^}]*}" "$f" | sed -n '1p;$p'
    grep -o "{'eval_loss'[^}]*}\|{'train_runtime'[^}]*}" "$f" || true
    grep -m3 -E "Error|Traceback" "$f" || true
  done
  for f in $LOGS; do grep -q "validation records" "$f" || continue
    echo "-- $f: accuracy (val, then train), act 1 play on test runs 801-1000"
    grep -E "records$|^(group|all|not forced)" "$f" || true
    grep -E "^summary" "$f" || echo "(play: $(grep -c '^seed' "$f") of 200 runs done)"
    grep -m3 -E "Error|Traceback|Fatal" "$f" || true
  done
  exit 0
fi

# QoS: at most 2 submitted jobs per user (queued + running); the chain needs both
squeue -u "$USER" -h | grep -q . && { echo "STOP: you already have jobs in the queue (squeue -u $USER), need 2 free slots"; exit 1; }
[ -d runs/sft-$DATA ] && { echo "STOP: runs/sft-$DATA exists (an earlier run); delete it to train again"; exit 1; }

echo "== build"
PY=$PY JOBS=16 ./build.sh > runs/build.log 2>&1 || { tail -20 runs/build.log; exit 1; }

echo "== config"
$PY - <<'EOF'
import json
p = "sts1/config.json"; c = json.load(open(p))
c["collect"].update(mode="act1", run_seed=None, games=1000, seed_start=1, split=[6, 2, 2], out_dir="data")
c["play"].update(mode="act1", run_seed=None, games=200)   # the test runs 801-1000
c["fight"].update(seed=801, ascension=0)
c["llm"].update(base_url="", model="Qwen/Qwen3-4B-Instruct-2507",
                adapter="runs/sft-act1_1000runs/final", temperature=0.0, max_tokens=512)
json.dump(c, open(p, "w"), indent=2)
EOF

echo "== collect (1000 runs, a few minutes)"
nice $PY sts1/collect.py | tee runs/collect_$DATA.log

echo "== format check"
$PY sts1/sft.py show | tail -4 | tee runs/show_$DATA.log
grep -c "over MAX_LENGTH: 0" runs/show_$DATA.log | grep -qx 3 || { echo "STOP: examples longer than MAX_LENGTH"; exit 1; }

FIRST=$(sbatch --parsable sts1/sft.sbatch chain runs/sft-$DATA/final)
echo "$FIRST" > "$JOBS_FILE"
echo "== queued: job chain from $FIRST (train ~12 h over 2-3 jobs, then eval + play ~4 h)"
echo "check any time with: bash sts1/act1.sh results"
