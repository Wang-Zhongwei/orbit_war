#!/bin/bash
# Smoke-test single-file Kaggle submissions: import (compiles JAX at import
# time, like Kaggle does) then run a full episode with env.run([mod.agent, ...])
# and report per-step latency. Usage:
#   sbatch jobs/templates/sub_smoke.slurm submission/sub_v8.py [more.py ...]
#SBATCH --job-name=sub_smoke
#SBATCH --partition=cpuqs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:40:00
#SBATCH --output=jobs/logs/%x-%j.out
#SBATCH --error=jobs/logs/%x-%j.err

set -euo pipefail
REPO_ROOT="/home/018270239/orbit_war"
cd "${REPO_ROOT}"
export PYTHONUNBUFFERED=1

for sub in "$@"; do
  echo "=== smoke: ${sub} ==="
  uv run python - "${sub}" <<'EOF'
import importlib.util, sys, time, statistics

path = sys.argv[1]
t0 = time.time()
spec = importlib.util.spec_from_file_location("sub_smoke_mod", path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
print(f"import OK in {time.time()-t0:.1f}s", flush=True)

from kaggle_environments import make
env = make("orbit_wars", debug=True)

times = []
def timed(obs, cfg=None):
    t = time.time()
    a = mod.agent(obs)
    times.append(time.time() - t)
    return a

env.run([timed, lambda obs, cfg=None: []])  # idle opponent; we only test our agent
statuses = [s["status"] for s in env.state]
print(f"episode done: steps={len(env.steps)} statuses={statuses}", flush=True)
print(f"per-step: mean={statistics.mean(times)*1e3:.0f}ms "
      f"max={max(times)*1e3:.0f}ms p95={sorted(times)[int(len(times)*0.95)]*1e3:.0f}ms", flush=True)
assert "ERROR" not in statuses and "TIMEOUT" not in statuses, "agent errored"
print("SMOKE PASS", flush=True)
EOF
done
echo "ALL PASS"
