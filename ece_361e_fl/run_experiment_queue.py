from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

PROJECT_DIR = Path(__file__).resolve().parent
CLOUD_CFG_RE = re.compile(r"cloud_cfg_exp(\d+)_run(\d+)\.json$")


@dataclass(frozen=True)
class Job:
    exp: int
    run: int
    cloud_cfg: Path
    dev_cfg: Path


@dataclass(frozen=True)
class DeviceTarget:
    hw_type: str
    host: str
    manager_host: str
    manager_port: int


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
            print(f"[!] Skipping exp{exp} run{run}: missing {dev_cfg.name}")
            continue

        jobs.append(Job(exp=exp, run=run, cloud_cfg=cloud_cfg, dev_cfg=dev_cfg))

    return jobs


def iter_device_targets(jobs: Iterable[Job], manager_port_offset: int) -> List[DeviceTarget]:
    seen = set()
    targets: List[DeviceTarget] = []

    for job in jobs:
        dev_cfg = load_json(job.dev_cfg)
        for index in range(1, int(dev_cfg.get("num_devices", 0)) + 1):
            dev = dev_cfg.get(f"dev{index}", {})
            host = str(dev.get("host", "")).strip()
            hw_type = str(dev.get("hw_type", "")).strip().lower()
            if not host or not hw_type:
                continue

            manager_host = str(dev.get("manager_host", host)).strip()
            manager_port = int(dev.get("manager_port", int(dev["port"]) + manager_port_offset))
            key = (hw_type, host, manager_host, manager_port)
            if key in seen:
                continue
            seen.add(key)
            targets.append(
                DeviceTarget(
                    hw_type=hw_type,
                    host=host,
                    manager_host=manager_host,
                    manager_port=manager_port,
                )
            )

    return targets


def manager_request(host: str, port: int, payload: Dict, timeout_s: int = 10) -> Dict:
    raw = (json.dumps(payload) + "\n").encode("utf-8")
    try:
        with socket.create_connection((host, port), timeout=timeout_s) as sock:
            sock.sendall(raw)
            data = b""
            while b"\n" not in data:
                chunk = sock.recv(4096)
                if not chunk:
                    break
                data += chunk
    except OSError as exc:
        return {"ok": False, "error": str(exc)}

    if not data:
        return {"ok": False, "error": "No response from manager"}

    try:
        return json.loads(data.decode("utf-8").strip())
    except json.JSONDecodeError as exc:
        return {"ok": False, "error": f"Invalid JSON from manager: {exc}"}


def manager_is_reachable(host: str, port: int, timeout_s: int = 5) -> bool:
    response = manager_request(host, port, {"cmd": "PING"}, timeout_s=timeout_s)
    return bool(response.get("ok"))


def start_remote_manager(
    target: DeviceTarget,
    remote_project_dir: str,
    remote_python: str,
    ssh_port: int,
    timeout_s: int,
    verbose: bool,
) -> None:
    try:
        import paramiko
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "paramiko is required to start remote device managers automatically. "
            "Install it locally or rerun with --skip_manager_start if the managers are already running."
        ) from exc

    try:
        from utils.general_utils import get_hw_info
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "The full project Python dependencies are required to resolve edge-device credentials. "
            "Activate the experiment environment before starting remote device managers."
        ) from exc

    password, username, _ = get_hw_info(target.hw_type)
    log_name = f"device_manager_{target.hw_type}_{target.manager_port}.log"
    launch_args = [
        remote_python,
        "-u",
        "device_manager.py",
        "--listen_host",
        "0.0.0.0",
        "--listen_port",
        str(target.manager_port),
        "--project_dir",
        remote_project_dir,
    ]
    if verbose:
        launch_args.append("--verbose")

    pgrep_pattern = f"device_manager.py .*--listen_port {target.manager_port}"
    quoted_project_dir = shlex.quote(remote_project_dir)
    quoted_log_path = shlex.quote(f"{remote_project_dir}/logs/{log_name}")
    quoted_launch = " ".join(shlex.quote(arg) for arg in launch_args)
    remote_cmd = (
        f"cd {quoted_project_dir} && mkdir -p logs && "
        f"if pgrep -f {shlex.quote(pgrep_pattern)} >/dev/null 2>&1; then "
        f"echo '__ALREADY_RUNNING__'; "
        f"else nohup {quoted_launch} > {quoted_log_path} 2>&1 < /dev/null & "
        f"echo '__STARTED__'; fi"
    )

    print(f"[+] Ensuring device manager is running on {target.host}:{target.manager_port}")
    with paramiko.SSHClient() as ssh_client:
        ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        connect_kwargs = {
            "hostname": target.host,
            "username": username,
            "port": ssh_port,
            "timeout": timeout_s,
            "banner_timeout": timeout_s,
            "auth_timeout": timeout_s,
        }
        if password:
            connect_kwargs["password"] = password
        ssh_client.connect(**connect_kwargs)
        _, stdout, stderr = ssh_client.exec_command(remote_cmd, timeout=timeout_s)
        stdout_text = stdout.read().decode("utf-8", errors="replace").strip()
        stderr_text = stderr.read().decode("utf-8", errors="replace").strip()
        if stderr_text:
            print(f"[!] {target.host} stderr: {stderr_text}")
        if stdout_text:
            print(f"[+] {target.host}: {stdout_text}")


def wait_for_manager(target: DeviceTarget, timeout_s: int, poll_interval_s: float = 2.0) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if manager_is_reachable(target.manager_host, target.manager_port, timeout_s=5):
            print(f"[+] Manager reachable at {target.manager_host}:{target.manager_port}")
            return
        time.sleep(poll_interval_s)

    raise TimeoutError(
        f"Timed out waiting for device manager at {target.manager_host}:{target.manager_port} "
        f"after {timeout_s} seconds."
    )


def run_command(cmd: List[str], cwd: Path, env: Optional[Dict[str, str]] = None) -> None:
    print(f"[+] Running: {' '.join(shlex.quote(part) for part in cmd)}")
    subprocess.run(cmd, cwd=str(cwd), check=True, env=env)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate configs, ensure remote device managers are running, and launch the cloud queue manager.",
    )
    parser.add_argument(
        "--config_script",
        type=str,
        default="config_100.bash",
        help="Bash script that generates cloud/dev configs before launching the queue",
    )
    parser.add_argument("--skip_generate", action="store_true", help="Skip running the config-generation bash script")
    parser.add_argument("--skip_manager_start", action="store_true", help="Assume device managers are already running")
    parser.add_argument("--sync_models", action="store_true", help="Sync local models/ to the edge devices first")
    parser.add_argument("--configs_dir", type=str, default="configs", help="Directory containing generated configs")
    parser.add_argument("--start_exp", type=int, default=100, help="First experiment id to consider")
    parser.add_argument("--end_exp", type=int, default=None, help="Last experiment id to consider")
    parser.add_argument(
        "--manager_port_offset",
        type=int,
        default=1000,
        help="Manager port = device port + offset unless manager_port is present in the device config",
    )
    parser.add_argument("--manager_timeout_s", type=int, default=120, help="Timeout waiting for device readiness")
    parser.add_argument("--manager_boot_timeout_s", type=int, default=60, help="Timeout waiting for manager PING")
    parser.add_argument("--ssh_port", type=int, default=22, help="SSH port used to start remote device managers")
    parser.add_argument(
        "--remote_project_dir",
        type=str,
        default="/home/student/ece_361e_fl",
        help="Project directory on the edge devices",
    )
    parser.add_argument("--remote_python", type=str, default="python3", help="Python executable on the edge devices")
    parser.add_argument("--state_file", type=str, default="logs/job_queue_state.json", help="Queue state file")
    parser.add_argument(
        "--force_comm_rounds",
        type=int,
        default=30,
        help="If >0, overwrite comm_rounds before execution (same behavior as cloud_manager.py)",
    )
    parser.add_argument("--dry_run", action="store_true", help="Show the discovered jobs and targets, then stop")
    parser.add_argument("--verbose", action="store_true", help="Verbose device-manager logging")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configs_dir = (PROJECT_DIR / args.configs_dir).resolve()

    if not args.skip_generate:
        generate_env = dict(os.environ)
        generate_env.setdefault("PYTHON_BIN", sys.executable)
        run_command(["bash", args.config_script], cwd=PROJECT_DIR, env=generate_env)

    jobs = discover_jobs(configs_dir=configs_dir, start_exp=args.start_exp, end_exp=args.end_exp)
    if not jobs:
        raise RuntimeError(
            f"No jobs found in {configs_dir} for exp >= {args.start_exp}"
            + ("" if args.end_exp is None else f" and exp <= {args.end_exp}")
        )

    print("[+] Discovered jobs:")
    for job in jobs:
        print(f"    - exp{job.exp} run{job.run}")

    targets = iter_device_targets(jobs=jobs, manager_port_offset=args.manager_port_offset)
    if not targets:
        raise RuntimeError("No device targets discovered from the generated configs.")

    print("[+] Device manager targets:")
    for target in targets:
        print(f"    - {target.hw_type}: {target.host} -> {target.manager_host}:{target.manager_port}")

    if args.dry_run:
        print("[+] Dry run complete.")
        return

    if args.sync_models:
        run_command(
            [
                sys.executable,
                "sync_models_to_edges.py",
                "--dev_cfg",
                str(jobs[0].dev_cfg.relative_to(PROJECT_DIR)),
                "--devices",
                "rpi",
                "mc1",
                "--verbose",
            ],
            cwd=PROJECT_DIR,
        )

    if not args.skip_manager_start:
        for target in targets:
            if manager_is_reachable(target.manager_host, target.manager_port, timeout_s=5):
                print(f"[+] Reusing running manager at {target.manager_host}:{target.manager_port}")
                continue

            start_remote_manager(
                target=target,
                remote_project_dir=args.remote_project_dir,
                remote_python=args.remote_python,
                ssh_port=args.ssh_port,
                timeout_s=args.manager_boot_timeout_s,
                verbose=args.verbose,
            )
            wait_for_manager(target=target, timeout_s=args.manager_boot_timeout_s)

    cloud_manager_cmd = [
        sys.executable,
        "cloud_manager.py",
        "--configs_dir",
        args.configs_dir,
        "--start_exp",
        str(args.start_exp),
        "--manager_port_offset",
        str(args.manager_port_offset),
        "--manager_timeout_s",
        str(args.manager_timeout_s),
        "--state_file",
        args.state_file,
        "--force_comm_rounds",
        str(args.force_comm_rounds),
    ]

    if args.dry_run:
        cloud_manager_cmd.append("--dry_run")

    run_command(cloud_manager_cmd, cwd=PROJECT_DIR)


if __name__ == "__main__":
    main()
