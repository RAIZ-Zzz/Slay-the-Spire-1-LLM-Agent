#!/bin/bash
# TC2 GPU job for sts1/sft.py (A40 46 GB; settings copied from AI6130's run_tc2.sbatch).
# From the repo root on TC2:
#   mkdir -p runs && SMOKE=1 sbatch sts1/sft.sbatch   # trial: 20 steps, 200 validation examples -> runs/sft-smoke
#   mkdir -p runs && sbatch sts1/sft.sbatch           # full run -> runs/sft; resubmit to resume after the 6 h limit
#   sbatch sts1/sft.sbatch eval runs/sft-smoke/final 500   # action accuracy, LoRA adapter, first 500 val records
#   sbatch sts1/sft.sbatch eval none 500                   # same without LoRA (zero-shot Qwen)
#   sbatch sts1/sft.sbatch play                            # sts1/llm.py: the model in sts1/config.json plays
#   sbatch sts1/sft.sbatch evalplay runs/sft/final         # eval (all of val), then play, in one job
#   sbatch sts1/sft.sbatch chain runs/sft-X/final          # train (resubmitting past 6 h), then evalplay
# Log: runs/sts1-sft-<jobid>.out   Queue: squeue -u $USER   Cancel: scancel <jobid>
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=30G
#SBATCH --time=06:00:00
#SBATCH --job-name=sts1-sft
#SBATCH --output=runs/%x-%j.out

cd ~/my-project/LLM-STS/sts1-agent
export HF_HOME=~/.cache/huggingface HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 TRANSFORMERS_VERBOSITY=error
PY=~/.conda/envs/gpu-env/bin/python
nvidia-smi --query-gpu=name,memory.total --format=csv

if [ "$1" = eval ]; then
  $PY sts1/sft.py "$@"
elif [ "$1" = play ]; then
  $PY -X faulthandler sts1/llm.py   # faulthandler: on a segfault, print the Python line it died in
elif [ "$1" = evalplay ]; then      # eval ADAPTER on all of val and 2000 train records, then play: one job
  $PY sts1/sft.py eval "$2" 0
  $PY sts1/sft.py eval-train "$2" 2000
  $PY -X faulthandler sts1/llm.py
elif [ "$1" = chain ]; then         # chain ADAPTER [N]: train past the 6 h limit, then evalplay ADAPTER
  # First queue a successor that resumes from the last checkpoint if this job hits the time limit
  # (the QoS allows 2 submitted jobs: this one + the successor). If training ends inside this job,
  # the successor is replaced by evalplay. A failing run is not detected here (on the time limit
  # python may die before bash does, so its exit status cannot tell timeout from crash); instead
  # the chain stops after 4 jobs, i.e. 24 h.
  N=${3:-1}
  [ "$N" -lt 4 ] && NEXT=$(sbatch --parsable --dependency=afterany:$SLURM_JOB_ID sts1/sft.sbatch chain "$2" $((N + 1)))
  if $PY sts1/sft.py train; then
    [ -n "$NEXT" ] && scancel $NEXT
    sbatch sts1/sft.sbatch evalplay "$2"
  fi
elif [ -n "$SMOKE" ]; then
  # same train() with a few settings overridden, so sft.py itself stays the real configuration
  $PY - <<'EOF'
import sys, trl
sys.path.insert(0, "sts1")
import sft
full_config = trl.SFTConfig
sft.OUTPUT_DIR = sft.ROOT / "runs" / "sft-smoke"
trl.SFTConfig = lambda **kw: full_config(**{**kw, "output_dir": str(sft.OUTPUT_DIR), "max_steps": 20,
                                            "logging_steps": 1, "eval_steps": 10, "save_steps": 20})
dataset = sft.build_dataset()
dataset["validation"] = dataset["validation"].select(range(200))
sft.train(dataset)
EOF
else
  $PY sts1/sft.py train
fi
