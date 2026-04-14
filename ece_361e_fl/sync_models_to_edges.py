"""Sync the local models directory to the edge devices via SCP.

This script copies the contents of the local `models/` directory into
`/home/student/ece_361e_fl/models/` on the configured edge devices.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List, Tuple

import paramiko
from scp import SCPClient

from utils.general_utils import get_hw_info


PROJECT_DIR = Path(__file__).resolve().parent
LOCAL_MODELS_DIR = PROJECT_DIR / "models"
REMOTE_MODELS_DIR = "/home/student/ece_361e_fl/models/"


def load_device_config(dev_cfg_path: Path) -> dict:
    with dev_cfg_path.open("r", encoding="utf-8") as file_handle:
        return json.load(file_handle)


def iter_device_targets(dev_cfg: dict, requested_devices: Iterable[str]) -> List[Tuple[str, str]]:
    requested = {device.lower() for device in requested_devices}
    targets: List[Tuple[str, str]] = []

    for index in range(1, int(dev_cfg.get("num_devices", 0)) + 1):
        dev = dev_cfg.get(f"dev{index}", {})
        hw_type = str(dev.get("hw_type", "")).lower()
        host = str(dev.get("host", "")).strip()

        if hw_type in requested and host:
            targets.append((hw_type, host))

    return targets


def ensure_remote_dir(ssh_client: paramiko.SSHClient, remote_dir: str) -> None:
    ssh_client.exec_command(f"mkdir -p {remote_dir}")


def sync_models_to_host(host: str, hw_type: str, dry_run: bool = False, verbose: bool = False) -> None:
    password, username, _ = get_hw_info(hw_type)

    if verbose or dry_run:
        print(f"[+] {hw_type}@{host}: syncing {LOCAL_MODELS_DIR} -> {REMOTE_MODELS_DIR}")

    if dry_run:
        for source_path in sorted(LOCAL_MODELS_DIR.rglob("*")):
            if source_path.is_dir() or "__pycache__" in source_path.parts or source_path.suffix == ".pyc":
                continue
            relative_path = source_path.relative_to(LOCAL_MODELS_DIR)
            print(f"    {relative_path}")
        return

    with paramiko.SSHClient() as ssh_client:
        ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh_client.connect(
            hostname=host,
            username=username,
            password=password,
            port=22,
            timeout=30,
            banner_timeout=30,
            auth_timeout=30,
        )

        ensure_remote_dir(ssh_client, REMOTE_MODELS_DIR)

        with SCPClient(ssh_client.get_transport()) as scp_client:
            for source_path in sorted(LOCAL_MODELS_DIR.rglob("*")):
                if source_path.is_dir() or "__pycache__" in source_path.parts or source_path.suffix == ".pyc":
                    continue

                relative_path = source_path.relative_to(LOCAL_MODELS_DIR)
                remote_parent = str((Path(REMOTE_MODELS_DIR) / relative_path.parent).as_posix())
                ssh_client.exec_command(f"mkdir -p {remote_parent}")
                scp_client.put(str(source_path), remote_path=remote_parent)

        if verbose:
            print(f"[+] {hw_type}@{host}: sync complete")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sync local models/ to edge devices via SCP")
    parser.add_argument(
        "--dev_cfg",
        type=str,
        default="configs/dev_cfg_exp100_run1.json",
        help="Device config file containing edge hostnames",
    )
    parser.add_argument(
        "--devices",
        nargs="+",
        default=["rpi", "mc1"],
        help="Device hw types to sync (e.g. rpi mc1)",
    )
    parser.add_argument("--dry_run", action="store_true", help="Show what would be copied without transferring")
    parser.add_argument("--verbose", action="store_true", help="Print progress messages")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dev_cfg_path = (PROJECT_DIR / args.dev_cfg).resolve()

    if not LOCAL_MODELS_DIR.exists():
        raise FileNotFoundError(f"Local models directory not found: {LOCAL_MODELS_DIR}")

    dev_cfg = load_device_config(dev_cfg_path)
    targets = iter_device_targets(dev_cfg=dev_cfg, requested_devices=args.devices)

    if not targets:
        print("[!] No matching devices found in the device config.")
        return

    for hw_type, host in targets:
        sync_models_to_host(host=host, hw_type=hw_type, dry_run=args.dry_run, verbose=args.verbose)


if __name__ == "__main__":
    main()