import argparse
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from urllib import error, request


# Local-only convenience Discord config.
# Fill the webhook URL directly if you do not want to export env vars in the shell.
# Keep secrets out of version control if this file is ever shared.
LOCAL_DISCORD_ENV = {
    "FL_DISCORD_WEBHOOK_URL": "https://discordapp.com/api/webhooks/1493440076418519080/CPNAG8HhgbjvAB8c4fwPmtuIOZ7fe_vw2LkBHAFO-jRbOVR6qdjyHRRSYe4LZQrxia4n",
}

for _k, _v in LOCAL_DISCORD_ENV.items():
    if _v:
        os.environ[_k] = str(_v)


CLOUD_CFG_RE = re.compile(r"cloud_cfg_exp(\d+)_run(\d+)\.json$")
TIME_TO_BEAT_S = 850.0
ENERGY_TO_BEAT_J = 786.0
DISCORD_SEPARATOR = "==================================================================="


@dataclass
class Job:
    exp: int
    run: int
    cloud_cfg: Path
    dev_cfg: Path


def discover_jobs(configs_dir: Path, start_exp: int) -> List[Job]:
    jobs: List[Job] = []

    for p in configs_dir.glob("cloud_cfg_exp*_run*.json"):
        m = CLOUD_CFG_RE.search(p.name)
        if not m:
            continue

        exp = int(m.group(1))
        run = int(m.group(2))
        if exp < start_exp:
            continue

        dev_cfg = configs_dir / f"dev_cfg_exp{exp}_run{run}.json"
        if not dev_cfg.exists():
            print(f"[!] Skipping exp{exp} run{run}: missing {dev_cfg.name}")
            continue

        jobs.append(Job(exp=exp, run=run, cloud_cfg=p, dev_cfg=dev_cfg))

    jobs.sort(key=lambda j: (j.exp, j.run))
    return jobs


def load_json(file_path: Path) -> Dict:
    with file_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(file_path: Path, payload: Dict) -> None:
    with file_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def archive_job_checkpoints(project_dir: Path, job: Job, cloud_cfg: Dict) -> Optional[Path]:
    try:
        from utils.general_utils import get_hw_info
    except ModuleNotFoundError as exc:
        print(f"[!] Could not import get_hw_info for checkpoint archiving: {exc}")
        return None

    laptop_number = str(cloud_cfg.get("laptop_number", "laptop_1"))
    _, _, cloud_path = get_hw_info(hw_type=laptop_number)
    source_dir = Path(cloud_path)
    if not source_dir.exists():
        print(f"[!] Checkpoint source directory does not exist: {source_dir}")
        return None

    archive_dir = project_dir / "artifacts" / "checkpoints" / f"exp{job.exp}_run{job.run}"
    archive_dir.mkdir(parents=True, exist_ok=True)

    copied_any = False
    for filename in ("global_weights.pth", "dev_0.pth", "dev_1.pth"):
        src = source_dir / filename
        if src.exists():
            shutil.copy2(src, archive_dir / filename)
            copied_any = True

    shutil.copy2(job.cloud_cfg, archive_dir / job.cloud_cfg.name)
    shutil.copy2(job.dev_cfg, archive_dir / job.dev_cfg.name)

    if not copied_any:
        print(f"[!] No checkpoint files were found to archive in {source_dir}")
        return None

    print(f"[+] Archived checkpoints to {archive_dir}")
    return archive_dir


def parse_numeric_value(v) -> int:
    if isinstance(v, int):
        return v
    if isinstance(v, str):
        m = re.search(r"(\d+)", v)
        if m:
            return int(m.group(1))
    raise ValueError(f"Cannot parse numeric value from {v}")


def manager_request(host: str, port: int, request: Dict, timeout_s: int = 15) -> Dict:
    raw = (json.dumps(request) + "\n").encode("utf-8")

    try:
        with socket.create_connection((host, port), timeout=timeout_s) as sock:
            sock.sendall(raw)

            data = b""
            while b"\n" not in data:
                chunk = sock.recv(4096)
                if not chunk:
                    break
                data += chunk
    except (OSError, ConnectionError) as exc:
        return {"ok": False, "error": f"Manager connection failed ({host}:{port}): {exc}"}

    if not data:
        return {"ok": False, "error": "No response from manager"}

    try:
        return json.loads(data.decode("utf-8").strip())
    except json.JSONDecodeError:
        return {"ok": False, "error": "Invalid JSON response", "raw": data.decode("utf-8", errors="replace")}


def run_process(cmd: List[str], cwd: Path, log_file: Optional[Path] = None) -> Tuple[int, str]:
    print(f"[+] Running: {' '.join(cmd)}")

    output_lines: List[str] = []
    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    with (log_file.open("w", encoding="utf-8") if log_file else open(os.devnull, "w", encoding="utf-8")) as lf:
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            output_lines.append(line)
            if log_file:
                lf.write(line)

    returncode = proc.wait()
    return returncode, "".join(output_lines)


def _send_discord_message(webhook_url: str, content: str) -> None:
    payload = {"content": content}
    raw = json.dumps(payload).encode("utf-8")
    webhook_url = webhook_url.replace("https://discordapp.com/", "https://discord.com/")
    req = request.Request(
        webhook_url,
        data=raw,
        headers={
            "Content-Type": "application/json",
            "User-Agent": "ece361e-fl-queue-manager/1.0",
        },
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=20):
            pass
    except error.HTTPError as exc:
        body = ""
        try:
            body = exc.read().decode("utf-8", errors="replace")
        except Exception:
            body = ""
        raise RuntimeError(f"Discord webhook HTTP {exc.code}: {exc.reason}. Response: {body}") from exc


def _chunk_message(msg: str, max_len: int = 1900) -> List[str]:
    if len(msg) <= max_len:
        return [msg]

    chunks: List[str] = []
    current = ""
    for line in msg.splitlines(keepends=True):
        if len(current) + len(line) > max_len:
            if current:
                chunks.append(current)
                current = ""
            if len(line) > max_len:
                start = 0
                while start < len(line):
                    chunks.append(line[start:start + max_len])
                    start += max_len
            else:
                current = line
        else:
            current += line

    if current:
        chunks.append(current)

    return chunks


def send_discord_report(message: str) -> None:
    webhook_url = os.environ.get("FL_DISCORD_WEBHOOK_URL") or os.environ.get("DISCORD_WEBHOOK_URL")
    if not webhook_url:
        print("[!] Discord notification skipped: missing FL_DISCORD_WEBHOOK_URL")
        return

    chunks = _chunk_message(message, max_len=1900)
    for idx, chunk in enumerate(chunks, start=1):
        prefix = f"(part {idx}/{len(chunks)})\n" if len(chunks) > 1 else ""
        _send_discord_message(webhook_url, prefix + chunk)

    print("[+] Discord notification sent.")


def _parse_float(text_value: str) -> Optional[float]:
    try:
        return float(text_value.replace(",", "").strip())
    except (TypeError, ValueError):
        return None


def parse_figs_metrics(figs_output: str) -> Dict[str, Optional[float]]:
    def extract(pattern: str) -> Optional[float]:
        m = re.search(pattern, figs_output)
        if not m:
            return None
        return _parse_float(m.group(1))

    metrics: Dict[str, Optional[float]] = {
        "global_acc": extract(r"Global test accuracy \[%\]:\s*([0-9.,]+)\s*%"),
        "time_to_90": extract(r"Total wall clock time to reach 90\.00% \[s\]:\s*([0-9.,]+)\s*seconds"),
        "avg_time_per_round": extract(r"Average time per communication round \[s\]:\s*([0-9.,]+)\s*seconds"),
        "total_wall_time": extract(r"Total wall clock time \[s\]:\s*([0-9.,]+)\s*seconds"),
        "rpi_energy": extract(r"RPi avg\. energy consumption per round \[J\]:\s*([0-9.,]+)\s*Joules"),
        "mc1_energy": extract(r"MC1 avg\. energy consumption per round \[J\]:\s*([0-9.,]+)\s*Joules"),
        "total_energy": extract(r"Total avg\. energy consumption per communication round \[J\]:\s*([0-9.,]+)\s*Joules"),
    }

    convergence_match = re.search(r"Convergence time \[#round\]:\s*([0-9]+)", figs_output)
    metrics["convergence_round"] = float(convergence_match.group(1)) if convergence_match else None
    metrics["qualifies"] = 1.0 if metrics["time_to_90"] is not None else 0.0
    return metrics


def parse_device_avg_times(cloud_output: str) -> Dict[str, float]:
    device_times: Dict[str, float] = {}
    in_section = False

    for line in cloud_output.splitlines():
        stripped = line.strip()
        if stripped == "Device average training time per communication round [s]:":
            in_section = True
            continue

        if not in_section:
            continue

        if not stripped:
            break

        match = re.match(r"^(.+?):\s*([0-9.,]+)\s*seconds$", stripped)
        if not match:
            continue

        device_name = match.group(1).strip()
        value = _parse_float(match.group(2))
        if value is not None:
            device_times[device_name] = value

    return device_times


def parse_early_stop_info(cloud_output: str) -> Dict[str, Optional[float]]:
    match = re.search(
        r"EARLY_STOP:\s*round=(\d+);\s*best_acc=([0-9.,]+);\s*current_acc=([0-9.,]+);"
        r"\s*threshold=([0-9.,]+);\s*patience=(\d+);\s*reason=([A-Za-z0-9_\-]+)",
        cloud_output,
    )
    if not match:
        return {"early_stop_triggered": 0.0, "early_stop_round": None}

    return {
        "early_stop_triggered": 1.0,
        "early_stop_round": float(match.group(1)),
        "early_stop_best_acc": _parse_float(match.group(2)),
        "early_stop_current_acc": _parse_float(match.group(3)),
        "early_stop_threshold": _parse_float(match.group(4)),
        "early_stop_patience": float(match.group(5)),
        "early_stop_reason": match.group(6),
    }


def _fmt(value: Optional[float], suffix: str = "") -> str:
    if value is None:
        return "n/a"
    return f"{value:,.2f}{suffix}"


def build_benchmark_excerpt(metrics: Dict[str, Optional[float]]) -> List[str]:
    qualifies = bool(metrics.get("qualifies"))
    if not qualifies:
        return [
            "Qualification summary:",
            "This experiment does not qualify (did not reach and sustain >= 90% global accuracy).",
        ]

    lines: List[str] = ["Qualification summary:"]

    time_to_90 = metrics.get("time_to_90")
    if time_to_90 is not None:
        time_margin = TIME_TO_BEAT_S - time_to_90
        lines.append(
            f"**Time to beat: {TIME_TO_BEAT_S:,.2f} seconds** | "
            f"Current (time to reach 90%): {_fmt(time_to_90, ' seconds')}"
        )
        if time_margin >= 0:
            lines.append(f"Beating time target by {time_margin:,.2f} seconds")
        else:
            lines.append(f"Missing time target by {abs(time_margin):,.2f} seconds")

    total_energy = metrics.get("total_energy")
    if total_energy is not None:
        energy_margin = ENERGY_TO_BEAT_J - total_energy
        lines.append(f"**Energy to beat: {ENERGY_TO_BEAT_J:,.2f} J** | Current: {_fmt(total_energy, ' J')}")
        if energy_margin >= 0:
            lines.append(f"Beating energy target by {energy_margin:,.2f} J")
        else:
            lines.append(f"Missing energy target by {abs(energy_margin):,.2f} J")

    return lines


def update_benchmark_tracker(tracker_path: Path, job: Job, metrics: Dict[str, Optional[float]]) -> None:
    tracker = {"entries": []}
    if tracker_path.exists():
        try:
            existing = load_json(tracker_path)
            if isinstance(existing, dict) and isinstance(existing.get("entries"), list):
                tracker = existing
        except Exception:
            tracker = {"entries": []}

    qualifies = bool(metrics.get("qualifies"))
    time_to_90 = metrics.get("time_to_90")
    total_energy = metrics.get("total_energy")

    beats_time = bool(qualifies and time_to_90 is not None and time_to_90 <= TIME_TO_BEAT_S)
    beats_energy = bool(qualifies and total_energy is not None and total_energy <= ENERGY_TO_BEAT_J)

    entry = {
        "exp": job.exp,
        "run": job.run,
        "timestamp": time.time(),
        "qualifies": qualifies,
        "global_acc": metrics.get("global_acc"),
        "time_to_90_s": time_to_90,
        "total_energy_per_round_j": total_energy,
        "beats_time": beats_time,
        "beats_energy": beats_energy,
        "time_margin_s": (TIME_TO_BEAT_S - time_to_90) if time_to_90 is not None else None,
        "energy_margin_j": (ENERGY_TO_BEAT_J - total_energy) if total_energy is not None else None,
        "categories_beaten": [
            label for label, ok in (("time", beats_time), ("energy", beats_energy)) if ok
        ],
    }

    tracker["entries"].append(entry)
    tracker_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(tracker_path, tracker)


def is_leaderboard_entry_included(entry: Dict) -> bool:
    return not bool(entry.get("leaderboard_excluded"))


def build_goal_beaters_summary(tracker_path: Path) -> List[str]:
    def _format_item(entry: Dict, metric_key: str, unit: str, bold: bool = False) -> Optional[str]:
        value = entry.get(metric_key)
        if value is None:
            return None
        exp = entry.get("exp")
        run = entry.get("run")
        item = f"exp{exp} run{run} ({float(value):,.2f}{unit})"
        return f"**{item}**" if bold else item

    lines: List[str] = ["", "Goal-beating experiments so far:"]
    time_items: List[str] = []
    energy_items: List[str] = []

    if tracker_path.exists():
        try:
            tracker = load_json(tracker_path)
            entries = tracker.get("entries", []) if isinstance(tracker, dict) else []
            if isinstance(entries, list):
                time_entries: List[Dict] = []
                energy_entries: List[Dict] = []
                for entry in entries:
                    if not isinstance(entry, dict):
                        continue
                    if not is_leaderboard_entry_included(entry):
                        continue
                    if entry.get("beats_time"):
                        if entry.get("time_to_90_s") is not None:
                            time_entries.append(entry)
                    if entry.get("beats_energy"):
                        if entry.get("total_energy_per_round_j") is not None:
                            energy_entries.append(entry)

                time_entries.sort(key=lambda item: float(item["time_to_90_s"]))
                energy_entries.sort(key=lambda item: float(item["total_energy_per_round_j"]))

                for idx, entry in enumerate(time_entries[:10]):
                    item = _format_item(entry, "time_to_90_s", "s", bold=(idx == 0))
                    if item:
                        time_items.append(item)

                for idx, entry in enumerate(energy_entries[:10]):
                    item = _format_item(entry, "total_energy_per_round_j", "J", bold=(idx == 0))
                    if item:
                        energy_items.append(item)
        except Exception:
            pass

    if time_items:
        lines.append("Beat target Wallclock: " + ", ".join(time_items))
    else:
        lines.append("Beat target Wallclock: none yet")

    if energy_items:
        lines.append("Beat target Energy: " + ", ".join(energy_items))
    else:
        lines.append("Beat target Energy: none yet")

    return lines


def format_job_discord_message(
    job: Job,
    cloud_cfg: Dict,
    dev_cfg: Dict,
    metrics: Dict[str, Optional[float]],
    benchmark_tracker_path: Path,
) -> str:
    rpi_epochs = dev_cfg.get("dev1", {}).get("local_epochs", "?")
    mc1_epochs = dev_cfg.get("dev2", {}).get("local_epochs", "?")
    lines = [
        f"Job: exp{job.exp}, run{job.run}",
        f"Config summary: model={cloud_cfg.get('model_name')} | lr={cloud_cfg.get('learning_rate')} | "
        f"loss={cloud_cfg.get('loss_type')} | mu={cloud_cfg.get('mu')} | beta={cloud_cfg.get('beta')} | "
        f"rounds={cloud_cfg.get('comm_rounds')} | rpi_epochs={rpi_epochs} | mc1_epochs={mc1_epochs}",
        "",
        f"Experiment {job.exp}:",
        f"    Global test accuracy [%]: {_fmt(metrics.get('global_acc'), ' %')}",
    ]

    if metrics.get("convergence_round") is not None:
        lines.append(f"    Convergence time [#round]: {int(metrics['convergence_round'])} communication rounds")
    else:
        lines.append("    No round found where all subsequent rounds have accuracy >= 90.00%.")

    if metrics.get("early_stop_triggered"):
        threshold = metrics.get("early_stop_threshold")
        threshold_text = f"{threshold:.2f}%" if isinstance(threshold, (float, int)) else "90.00%"
        lines.append(
            "    Early stopping: stopped at communication round "
            f"{int(metrics['early_stop_round'])} because accuracy converged below {threshold_text}."
        )
    else:
        lines.append("    Early stopping: not triggered.")

    lines.append(f"    **Time to beat: {TIME_TO_BEAT_S:,.2f} seconds**")
    lines.append(f"    Total wall clock time to reach 90.00% [s]: {_fmt(metrics.get('time_to_90'), ' seconds')}")
    lines.append(f"    Average time per communication round [s]: {_fmt(metrics.get('avg_time_per_round'), ' seconds')}")
    device_avg_times = metrics.get("device_avg_train_times") or {}

    def lookup_device_time(prefix: str) -> Optional[float]:
        for device_name, avg_time in device_avg_times.items():
            if device_name.lower().startswith(prefix.lower()):
                return avg_time
        return None

    lines.append(f"    RPI training time per round [s]: {_fmt(lookup_device_time('rpi'), ' seconds')}")
    lines.append(f"    MC1 training time per round [s]: {_fmt(lookup_device_time('mc1'), ' seconds')}")
    lines.append(f"    Total wall clock time [s]: {_fmt(metrics.get('total_wall_time'), ' seconds')}")
    lines.append("")

    lines.extend([
        f"Experiment {job.exp}:",
        f"    RPi avg. energy consumption per round [J]: {_fmt(metrics.get('rpi_energy'), ' Joules')}",
        f"    MC1 avg. energy consumption per round [J]: {_fmt(metrics.get('mc1_energy'), ' Joules')}",
        f"    Total avg. energy consumption per communication round [J]: {_fmt(metrics.get('total_energy'), ' Joules')}",
        f"    **Energy to beat: {ENERGY_TO_BEAT_J:,.2f} J**",
        "",
    ])

    lines.extend(build_benchmark_excerpt(metrics))
    lines.extend(build_goal_beaters_summary(benchmark_tracker_path))

    lines.extend(["", DISCORD_SEPARATOR])
    return "\n".join(lines)


def trim_figs_output(figs_output: str) -> str:
    filtered_lines = []
    skip_prefixes = (
        "Connecting to ",
        "Successfully fetched ",
    )

    for line in figs_output.splitlines():
        if line.startswith(skip_prefixes):
            continue
        filtered_lines.append(line)

    trimmed = "\n".join(filtered_lines).strip()
    return trimmed if trimmed else "[no summary output]"


def load_state(state_path: Path) -> Dict:
    if not state_path.exists():
        return {"last_successful": None, "history": []}
    return load_json(state_path)


def save_state(state_path: Path, state: Dict) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(state_path, state)


def job_is_after(last_success: Optional[Dict], exp: int, run: int) -> bool:
    if not last_success:
        return True
    last_exp = int(last_success.get("exp", -1))
    last_run = int(last_success.get("run", -1))
    return (exp, run) > (last_exp, last_run)


def prepare_devices(
    job: Job,
    cloud_cfg: Dict,
    dev_cfg: Dict,
    manager_port_offset: int,
    manager_timeout_s: int,
) -> Tuple[bool, List[Tuple[str, int]]]:
    num_devices = int(dev_cfg["num_devices"])
    started_managers: List[Tuple[str, int]] = []

    seed = parse_numeric_value(cloud_cfg.get("seed", 2))
    verbose = bool(cloud_cfg.get("verbose", False))

    for device_num in range(1, num_devices + 1):
        dev = dev_cfg[f"dev{device_num}"]
        manager_host = dev.get("manager_host", dev["host"])
        manager_port = int(dev.get("manager_port", int(dev["port"]) + manager_port_offset))

        req = {
            "cmd": "PREPARE",
            "timeout_s": manager_timeout_s,
            "payload": {
                "host": dev["host"],
                "port": int(dev["port"]),
                "device_type": dev["hw_type"],
                "dev_idx": device_num - 1,
                "exp": job.exp,
                "run": job.run,
                "seed": seed,
                "verbose": verbose,
            },
        }

        print(f"[+] Preparing device manager at {manager_host}:{manager_port} for dev_idx={device_num - 1}")
        res = manager_request(manager_host, manager_port, req, timeout_s=manager_timeout_s + 10)
        if not res.get("ok", False):
            print(f"[!] Device manager prepare failed for {manager_host}:{manager_port}: {res}")
            return False, started_managers

        print(f"[+] Device manager ready: {manager_host}:{manager_port}")
        started_managers.append((manager_host, manager_port))

    return True, started_managers


def shutdown_started_managers(started_managers: List[Tuple[str, int]]) -> None:
    for host, port in started_managers:
        try:
            res = manager_request(host, port, {"cmd": "SHUTDOWN"}, timeout_s=8)
            print(f"[+] Shutdown request to {host}:{port}: {res}")
        except Exception as exc:
            print(f"[!] Failed shutdown request to {host}:{port}: {exc}")


def collect_manager_endpoints(
    jobs: List[Job],
    manager_port_offset: int,
) -> List[Tuple[str, int]]:
    endpoints: List[Tuple[str, int]] = []
    seen: Set[Tuple[str, int]] = set()

    for job in jobs:
        try:
            dev_cfg = load_json(job.dev_cfg)
            num_devices = int(dev_cfg.get("num_devices", 0))
        except Exception as exc:
            print(f"[!] Could not load device config for exp{job.exp} run{job.run}: {exc}")
            continue

        for device_num in range(1, num_devices + 1):
            dev = dev_cfg.get(f"dev{device_num}", {})
            host = dev.get("manager_host", dev.get("host"))
            port_value = dev.get("manager_port")

            if host is None:
                continue

            try:
                if port_value is None:
                    port = int(dev["port"]) + manager_port_offset
                else:
                    port = int(port_value)
            except (KeyError, TypeError, ValueError):
                continue

            endpoint = (str(host), int(port))
            if endpoint not in seen:
                seen.add(endpoint)
                endpoints.append(endpoint)

    return endpoints


def stop_remote_managers(endpoints: List[Tuple[str, int]], timeout_s: int = 10) -> None:
    if not endpoints:
        print("[!] No device manager endpoints discovered for STOP_SERVER notification.")
        return

    print("[+] Queue complete; notifying device managers to stop scripts...")
    for host, port in endpoints:
        try:
            res = manager_request(host, port, {"cmd": "STOP_SERVER"}, timeout_s=timeout_s)
            if res.get("ok", False):
                print(f"[+] STOP_SERVER sent to {host}:{port}")
            else:
                print(f"[!] STOP_SERVER failed for {host}:{port}: {res}")
        except Exception as exc:
            print(f"[!] STOP_SERVER exception for {host}:{port}: {exc}")


def clear_figs_directory(figs_dir: Path) -> None:
    """Delete all files/subfolders inside figs_dir while keeping figs_dir itself."""
    if not figs_dir.exists():
        return

    for item in figs_dir.iterdir():
        try:
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()
        except Exception as exc:
            print(f"[!] Failed to delete {item}: {exc}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Cloud queue manager for FL jobs")
    parser.add_argument("--configs_dir", type=str, default="configs", help="Directory containing cloud/dev cfg files")
    parser.add_argument("--start_exp", type=int, default=100, help="Start executing from this experiment id")
    parser.add_argument("--manager_port_offset", type=int, default=1000, help="Manager port = device port + offset unless manager_port exists in dev cfg")
    parser.add_argument("--manager_timeout_s", type=int, default=120, help="Timeout waiting for each device manager readiness")
    parser.add_argument("--state_file", type=str, default="logs/job_queue_state.json", help="JSON file tracking last successful job")
    parser.add_argument("--force_comm_rounds", type=int, default=30, help="If >0, overwrite cloud cfg comm_rounds before execution")
    parser.add_argument("--dry_run", action="store_true", help="List jobs and exit")
    parser.add_argument("--discord_test_only", action="store_true", help="Send a Discord test message and exit")
    parser.add_argument(
        "--discord_test_message",
        type=str,
        default="FL queue manager test notification",
        help="Message body to use with --discord_test_only",
    )
    args = parser.parse_args()

    project_dir = Path(__file__).resolve().parent
    configs_dir = (project_dir / args.configs_dir).resolve()
    state_file = (project_dir / args.state_file).resolve()
    benchmark_tracker_file = (project_dir / "logs" / "benchmark_tracking.json").resolve()

    if args.discord_test_only:
        stamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        test_msg = f"{args.discord_test_message}\nTime: {stamp}\nHost: {socket.gethostname()}" + f"\n\n===================================================================\n"
        send_discord_report(test_msg)
        print("[+] Discord test mode complete.")
        return

    jobs = discover_jobs(configs_dir=configs_dir, start_exp=args.start_exp)
    if not jobs:
        print(f"[!] No jobs found in {configs_dir} for exp >= {args.start_exp}")
        return

    all_manager_endpoints = collect_manager_endpoints(
        jobs=jobs,
        manager_port_offset=args.manager_port_offset,
    )

    state = load_state(state_file)
    last_success = state.get("last_successful")

    runnable_jobs = [j for j in jobs if job_is_after(last_success, j.exp, j.run)]
    if not runnable_jobs:
        print("[+] No pending jobs (all discovered jobs are <= last successful job).")
        stop_remote_managers(all_manager_endpoints)
        try:
            send_discord_report("Done, queue is empty.")
        except Exception as exc:
            print(f"[!] Discord send failed for queue-empty notification: {exc}")
        return

    print("[+] Pending jobs:")
    for j in runnable_jobs:
        print(f"    - exp{j.exp} run{j.run}")

    if args.dry_run:
        print("[+] Dry run complete.")
        return

    queue_completed = True

    for job in runnable_jobs:
        print(f"\n===== Starting job exp{job.exp} run{job.run} =====")

        cloud_cfg = load_json(job.cloud_cfg)
        dev_cfg = load_json(job.dev_cfg)

        if args.force_comm_rounds > 0:
            cloud_cfg["comm_rounds"] = int(args.force_comm_rounds)
            write_json(job.cloud_cfg, cloud_cfg)
            print(f"[+] Set comm_rounds={args.force_comm_rounds} in {job.cloud_cfg.name}")

        ok, started_managers = prepare_devices(
            job=job,
            cloud_cfg=cloud_cfg,
            dev_cfg=dev_cfg,
            manager_port_offset=args.manager_port_offset,
            manager_timeout_s=args.manager_timeout_s,
        )
        if not ok:
            print("[!] Job aborted: at least one device manager failed to prepare.")
            shutdown_started_managers(started_managers)
            queue_completed = False
            break

        cloud_cmd = [
            sys.executable,
            "cloud.py",
            "--cloud_cfg",
            str(job.cloud_cfg.relative_to(project_dir)),
            "--dev_cfg",
            str(job.dev_cfg.relative_to(project_dir)),
        ]
        cloud_log = project_dir / "logs" / f"cloud_manager_cloud_exp{job.exp}_run{job.run}.log"
        cloud_log.parent.mkdir(parents=True, exist_ok=True)
        rc, cloud_output = run_process(cloud_cmd, cwd=project_dir, log_file=cloud_log)
        if rc != 0:
            print(f"[!] cloud.py failed for exp{job.exp} run{job.run} with exit code {rc}")
            clear_figs_directory(project_dir / "figs")
            shutdown_started_managers(started_managers)
            queue_completed = False
            break

        archive_job_checkpoints(project_dir=project_dir, job=job, cloud_cfg=cloud_cfg)

        device_avg_times = parse_device_avg_times(cloud_output)
        early_stop_info = parse_early_stop_info(cloud_output)

        figs_cmd = [
            sys.executable,
            "generate_figs.py",
            "--exps",
            str(job.exp),
            "--runs",
            "1",
            "--run_ids",
            str(job.run),
            "--exp_labels",
            f"exp{job.exp}",
            "--plot_title",
            f"Exp{job.exp}",
            "--time",
            "--accuracy",
            "--power",
            "--energy",
            "--communication",
        ]

        figs_log = project_dir / "logs" / f"cloud_manager_figs_exp{job.exp}_run{job.run}.log"
        figs_rc, figs_output = run_process(figs_cmd, cwd=project_dir, log_file=figs_log)
        if figs_rc != 0:
            print(f"[!] generate_figs.py failed for exp{job.exp} run{job.run} with exit code {figs_rc}")
            clear_figs_directory(project_dir / "figs")
            shutdown_started_managers(started_managers)
            queue_completed = False
            break

        cleaned_figs_output = trim_figs_output(figs_output)
        metrics = parse_figs_metrics(cleaned_figs_output)
        metrics["device_avg_train_times"] = device_avg_times
        metrics.update(early_stop_info)
        update_benchmark_tracker(benchmark_tracker_file, job, metrics)

        discord_message = format_job_discord_message(
            job=job,
            cloud_cfg=cloud_cfg,
            dev_cfg=dev_cfg,
            metrics=metrics,
            benchmark_tracker_path=benchmark_tracker_file,
        )
        try:
            send_discord_report(discord_message)
        except Exception as exc:
            print(f"[!] Discord send failed for exp{job.exp} run{job.run}: {exc}")

        state["last_successful"] = {"exp": job.exp, "run": job.run, "timestamp": time.time()}
        state.setdefault("history", []).append({
            "exp": job.exp,
            "run": job.run,
            "timestamp": time.time(),
            "status": "success",
        })
        save_state(state_file, state)
        print(f"[+] Job exp{job.exp} run{job.run} completed and state updated.")
        clear_figs_directory(project_dir / "figs")
        print("[+] Cleared figs directory contents.")

    print("[+] Cloud queue manager finished.")

    if queue_completed:
        stop_remote_managers(all_manager_endpoints)
        try:
            send_discord_report("Done, queue is empty.")
        except Exception as exc:
            print(f"[!] Discord send failed for queue-empty notification: {exc}")


if __name__ == "__main__":
    main()
