"""GPU job queue for a shared server: each job starts on the GPU with the most free memory (at least --need MiB), at most
--parallel jobs run at once, and a job that dies of CUDA out-of-memory is put back in the queue (up to --retries times).

  python scripts/run_queue.py jobs.txt --parallel 8 --need 7000 --env THREADS=6
Each line of jobs.txt: <log name> <command ...>   (the log goes to logs/queue/<log name>.log; CUDA_VISIBLE_DEVICES is set)
"""
from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def free_mem() -> dict[int, int]:
    out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.total,memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
    return {int(i): int(t) - int(u) for i, t, u in (l.split(", ") for l in out.strip().splitlines())}


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("jobs"); ap.add_argument("--parallel", type=int, default=8); ap.add_argument("--need", type=int, default=7000)
    ap.add_argument("--retries", type=int, default=4); ap.add_argument("--threads", type=int, default=6); a = ap.parse_args()
    q = [(l.split(None, 1)[0], l.split(None, 1)[1], 0) for l in Path(a.jobs).read_text().splitlines() if l.strip()]
    logd = ROOT / "logs/queue"; logd.mkdir(parents=True, exist_ok=True); running, reserved = [], {}
    env0 = {**os.environ, "OMP_NUM_THREADS": str(a.threads), "MKL_NUM_THREADS": str(a.threads), "OPENBLAS_NUM_THREADS": str(a.threads), "NUMBA_NUM_THREADS": str(a.threads)}
    while q or running:
        for p, name, cmd, tries, g in list(running):
            if p.poll() is None:
                continue
            running.remove((p, name, cmd, tries, g)); reserved[g] = reserved.get(g, 0) - a.need
            log = (logd / f"{name}.log").read_text(errors="ignore")
            if p.returncode != 0 and ("OutOfMemoryError" in log or "CUDA out of memory" in log) and tries < a.retries:
                q.append((name, cmd, tries + 1)); print(f"retry {name} ({tries + 1})", flush=True)
            else:
                print(f"{'done' if p.returncode == 0 else 'FAILED'} {name} (exit {p.returncode})", flush=True)
        while q and len(running) < a.parallel:
            fm = {g: m - reserved.get(g, 0) for g, m in free_mem().items()}; g = max(fm, key=fm.get)
            if fm[g] < a.need:
                break
            name, cmd, tries = q.pop(0); reserved[g] = reserved.get(g, 0) + a.need
            p = subprocess.Popen(["bash", "-c", f"cd {shlex.quote(str(ROOT))} && source \"$(conda info --base)/etc/profile.d/conda.sh\" && conda activate project018 && {cmd}"],
                                 env={**env0, "CUDA_VISIBLE_DEVICES": str(g)}, stdout=open(logd / f"{name}.log", "w"), stderr=subprocess.STDOUT)
            running.append((p, name, cmd, tries, g)); print(f"start {name} on GPU {g} (free {fm[g]} MiB)", flush=True); time.sleep(20)
        time.sleep(30)
    print("QUEUE EMPTY", flush=True)


if __name__ == "__main__":
    main()
