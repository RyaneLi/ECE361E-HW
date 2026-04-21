from __future__ import annotations

import argparse
import shlex
from typing import Iterable

import paramiko

from utils.general_utils import get_hw_info


DEVICE_HOSTS = {
    "rpi": "sld-rpi-09.ece.utexas.edu",
    "mc1": "sld-mc1-09.ece.utexas.edu",
}


def start_manager(hw_type: str, host: str, port: int, remote_project_dir: str) -> None:
    password, username, _ = get_hw_info(hw_type)
    session = f"device_manager_{hw_type}_{port}"
    launch = (
        f"cd {shlex.quote(remote_project_dir)} && "
        f"python -u device_manager.py --listen_host 0.0.0.0 --listen_port {port} "
        f"--project_dir {shlex.quote(remote_project_dir)} >> "
        f"logs/{session}.log 2>&1"
    )
    inner = (
        f"tmux kill-session -t {shlex.quote(session)} >/dev/null 2>&1 || true; "
        f"tmux new-session -d -s {shlex.quote(session)} {shlex.quote(launch)}; "
        "sleep 2; tmux ls | grep device_manager || true"
    )
    cmd = f"bash -lc {shlex.quote(inner)}"

    with paramiko.SSHClient() as client:
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        connect_kwargs = dict(
            hostname=host,
            username=username,
            port=22,
            timeout=20,
            banner_timeout=20,
            auth_timeout=20,
        )
        if password:
            connect_kwargs["password"] = password
        client.connect(**connect_kwargs)
        _, stdout, stderr = client.exec_command(cmd, timeout=30)
        out = stdout.read().decode("utf-8", errors="replace").strip()
        err = stderr.read().decode("utf-8", errors="replace").strip()
        print(f"=== {hw_type} ({host}) ===")
        if out:
            print(out)
        if err:
            print(f"STDERR: {err}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Start tmux-backed device_manager.py on the edge devices.")
    parser.add_argument("--devices", nargs="+", default=["rpi", "mc1"], help="Device types to start")
    parser.add_argument("--manager_port", type=int, default=10090, help="Device manager port")
    parser.add_argument(
        "--remote_project_dir",
        type=str,
        default="/home/student/ece_361e_fl",
        help="Project directory on the edge devices",
    )
    return parser.parse_args()


def normalize_devices(devices: Iterable[str]) -> Iterable[str]:
    for device in devices:
        hw_type = device.strip().lower()
        if hw_type not in DEVICE_HOSTS:
            raise ValueError(f"Unsupported device type: {device}")
        yield hw_type


def main() -> None:
    args = parse_args()
    for hw_type in normalize_devices(args.devices):
        start_manager(
            hw_type=hw_type,
            host=DEVICE_HOSTS[hw_type],
            port=args.manager_port,
            remote_project_dir=args.remote_project_dir,
        )


if __name__ == "__main__":
    main()
