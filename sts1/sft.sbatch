#!/bin/bash
# TC2 GPU job for sts1/sft.py (A40 46 GB; settings copied from AI6130's run_tc2.sbatch).
# From the repo root on TC2:
#   mkdir -p runs && SMOKE=1 sbatch sts1/sft.sbatch   # trial: 20 steps, 200 validation examples -> runs/sft-smoke
#   mkdir -p runs && sbatch sts1/sft.sbatch           # full run -> runs/sft; resubmit to resume after the 6 h limit
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

if [ -n "$SMOKE" ]; then
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
