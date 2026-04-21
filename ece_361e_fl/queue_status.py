from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


PROJECT_DIR = Path(__file__).resolve().parent
CLOUD_CFG_RE = re.compile(r"cloud_cfg_exp(\d+)_run(\d+)\.json$")


@dataclass(frozen=True)
class Job:
    exp: int
    run: int
    cloud_cfg: Path
    dev_cfg: Path


def load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def discover_jobs(configs_dir: Path, start_exp: int, end_exp: Optional[int]) -> List[Job]:
    jobs: List[Job] = []

    for cloud_cfg in sorted(configs_dir.glob("cloud_cfg_exp*_run*.json")):
        match = CLOUD_CFG_RE.fullmatch(cloud_cfg.name)
        if not match:
            continue

        exp = int(match.group(1))
        run = int(match.group(2))
        if exp < start_exp:
            continue
        if end_exp is not None and exp > end_exp:
            continue

        dev_cfg = configs_dir / f"dev_cfg_exp{exp}_run{run}.json"
        if not dev_cfg.exists():
            continue

        jobs.append(Job(exp=exp, run=run, cloud_cfg=cloud_cfg, dev_cfg=dev_cfg))

    jobs.sort(key=lambda job: (job.exp, job.run))
    return jobs


def job_key(job: Job) -> tuple[int, int]:
    return job.exp, job.run


def job_is_after(last_success: Optional[Dict], exp: int, run: int) -> bool:
    if not last_success:
        return True
    return (exp, run) > (int(last_success.get("exp", -1)), int(last_success.get("run", -1)))


def read_last_metric_row(log_csv: Path) -> Optional[Dict[str, str]]:
    if not log_csv.exists():
        return None

    last_row: Optional[Dict[str, str]] = None
    with log_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            last_row = row
    return last_row


def read_last_nonempty_line(path: Path) -> Optional[str]:
    if not path.exists():
        return None

    last: Optional[str] = None
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                last = stripped
    return last


def infer_active_job(pending_jobs: List[Job], logs_dir: Path) -> Optional[Job]:
    active_candidates: List[tuple[float, Job]] = []

    for job in pending_jobs:
        cloud_log = logs_dir / f"cloud_manager_cloud_exp{job.exp}_run{job.run}.log"
        metrics_log = logs_dir / f"log_exp{job.exp}_run{job.run}.csv"

        mtimes = []
        if cloud_log.exists():
            mtimes.append(cloud_log.stat().st_mtime)
        if metrics_log.exists():
            mtimes.append(metrics_log.stat().st_mtime)

        if mtimes:
            active_candidates.append((max(mtimes), job))

    if not active_candidates:
        return None

    active_candidates.sort(key=lambda item: item[0], reverse=True)
    return active_candidates[0][1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Show queue progress for the FL experiment sweep.")
    parser.add_argument("--configs_dir", type=str, default="configs", help="Directory containing cloud/dev configs")
    parser.add_argument("--state_file", type=str, default="logs/job_queue_state.json", help="Queue state file")
    parser.add_argument("--logs_dir", type=str, default="logs", help="Directory containing queue logs")
    parser.add_argument("--start_exp", type=int, default=100, help="First experiment id to include")
    parser.add_argument("--end_exp", type=int, default=None, help="Last experiment id to include")
    parser.add_argument("--max_pending", type=int, default=5, help="How many upcoming jobs to print")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configs_dir = (PROJECT_DIR / args.configs_dir).resolve()
    state_file = (PROJECT_DIR / args.state_file).resolve()
    logs_dir = (PROJECT_DIR / args.logs_dir).resolve()

    jobs = discover_jobs(configs_dir=configs_dir, start_exp=args.start_exp, end_exp=args.end_exp)
    if not jobs:
        print("No jobs discovered for the requested range.")
        return

    state = load_json(state_file) if state_file.exists() else {"last_successful": None, "history": []}
    last_success = state.get("last_successful")

    completed_jobs = [job for job in jobs if not job_is_after(last_success, job.exp, job.run)]
    pending_jobs = [job for job in jobs if job_is_after(last_success, job.exp, job.run)]
    active_job = infer_active_job(pending_jobs=pending_jobs, logs_dir=logs_dir)

    print("Queue Status")
    print("------------")
    print(f"Discovered jobs: {len(jobs)}")
    print(f"Completed jobs: {len(completed_jobs)}")
    print(f"Pending jobs: {len(pending_jobs)}")

    if last_success:
        print(
            "Last successful: "
            f"exp{last_success.get('exp')} run{last_success.get('run')}"
        )
    else:
        print("Last successful: none")

    if active_job:
        print(f"Inferred active job: exp{active_job.exp} run{active_job.run}")
        metrics_log = logs_dir / f"log_exp{active_job.exp}_run{active_job.run}.csv"
        metric_row = read_last_metric_row(metrics_log)
        if metric_row:
            print(
                "Latest metrics: "
                f"round={metric_row.get('CommRound')} | "
                f"acc={metric_row.get('Acc')} | "
                f"loss={metric_row.get('Loss')} | "
                f"time={metric_row.get('Time')}"
            )

        cloud_log = logs_dir / f"cloud_manager_cloud_exp{active_job.exp}_run{active_job.run}.log"
        last_line = read_last_nonempty_line(cloud_log)
        if last_line:
            print(f"Latest log line: {last_line}")
    else:
        print("Inferred active job: none")

    if pending_jobs:
        print("")
        print("Next jobs:")
        shown = 0
        for job in pending_jobs:
            if active_job and job_key(job) == job_key(active_job):
                continue
            print(f"  exp{job.exp} run{job.run}")
            shown += 1
            if shown >= args.max_pending:
                break


if __name__ == "__main__":
    main()
