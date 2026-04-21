from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


PROJECT_DIR = Path(__file__).resolve().parent

# Candidate 3-run families assembled from the M3 exploration batches.
# Some families reuse a run1 anchor from an earlier config and pair it with
# later run2/run3 follow-up experiments that use the same hyperparameters.
CANDIDATE_FAMILIES: Dict[str, List[Tuple[int, int]]] = {
    "sword_m7": [(371, 1), (369, 2), (373, 3)],
    "refined_m7": [(372, 1), (370, 2), (374, 3)],
    "explorer_fedavg_m7": [(341, 1), (375, 2), (376, 3)],
    "explorer_fedmax_m7": [(342, 1), (377, 2), (378, 3)],
    "sword_baseline_m9": [(199, 1), (402, 2), (402, 3)],
    "refined_baseline_m9": [(166, 1), (403, 2), (403, 3)],
    "explorer_fedavg_m9": [(341, 1), (404, 2), (404, 3)],
    "explorer_fedmax_m9": [(342, 1), (405, 2), (405, 3)],
    "sword_lr009": [(197, 1), (406, 2), (406, 3)],
    "sword_mu11": [(200, 1), (407, 2), (407, 3)],
    "explorer_fedprox_m10": [(209, 1), (408, 2), (408, 3)],
    "sword_mu09_m10": [(198, 1), (409, 2), (409, 3)],
    "refined_fedavg_m10": [(164, 1), (410, 2), (410, 3)],
    "deeper_fedavg_m9": [(399, 1), (399, 2), (399, 3)],
    "deeper_fedprox_m9": [(400, 1), (400, 2), (400, 3)],
    "groupnorm_m9": [(401, 1), (401, 2), (401, 3)],
}


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


def stddev(values: Sequence[float]) -> float:
    avg = mean(values)
    return (sum((value - avg) ** 2 for value in values) / len(values)) ** 0.5


def parse_script_lookup() -> Dict[Tuple[int, int], Dict[str, object]]:
    line_re = re.compile(r'^\s*"([^"]+)"')
    lookup: Dict[Tuple[int, int], Dict[str, object]] = {}

    for script_path in sorted(PROJECT_DIR.glob("config*.bash")):
        with script_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                match = line_re.match(line)
                if not match:
                    continue
                parts = match.group(1).split()
                if len(parts) != 10:
                    continue
                exp, run = int(parts[0]), int(parts[1])
                lookup[(exp, run)] = {
                    "model_name": parts[3],
                    "loss_type": parts[5],
                    "learning_rate": float(parts[4]),
                    "mu": float(parts[6]),
                    "beta": float(parts[7]),
                    "rpi_local_epochs": int(parts[8]),
                    "mc1_local_epochs": int(parts[9]),
                }
    return lookup


def load_config(exp: int, run: int, script_lookup: Dict[Tuple[int, int], Dict[str, object]]) -> Dict[str, object]:
    candidates = [
        PROJECT_DIR / f"configs/cloud_cfg_exp{exp}_run{run}.json",
        PROJECT_DIR / f"artifacts/checkpoints/exp{exp}_run{run}/cloud_cfg_exp{exp}_run{run}.json",
    ]
    for cloud_path in candidates:
        dev_path = cloud_path.with_name(f"dev_cfg_exp{exp}_run{run}.json")
        if cloud_path.exists() and dev_path.exists():
            with cloud_path.open("r", encoding="utf-8") as handle:
                cloud_cfg = json.load(handle)
            with dev_path.open("r", encoding="utf-8") as handle:
                dev_cfg = json.load(handle)
            return {
                "model_name": cloud_cfg["model_name"],
                "loss_type": cloud_cfg["loss_type"],
                "learning_rate": float(cloud_cfg["learning_rate"]),
                "mu": float(cloud_cfg.get("mu", 0.0)),
                "beta": float(cloud_cfg.get("beta", 0.0)),
                "rpi_local_epochs": int(dev_cfg["dev1"]["local_epochs"]),
                "mc1_local_epochs": int(dev_cfg["dev2"]["local_epochs"]),
            }

    fallback = script_lookup.get((exp, run))
    if fallback is None:
        raise FileNotFoundError(f"No config metadata found for exp{exp} run{run}")
    return fallback


def read_round_log(exp: int, run: int) -> Tuple[List[float], List[float]]:
    acc_values: List[float] = []
    time_values: List[float] = []
    path = PROJECT_DIR / f"logs/log_exp{exp}_run{run}.csv"
    with path.open("r", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            acc_values.append(float(row["Acc"]))
            time_values.append(float(row["Time"]))
    return acc_values, time_values


def read_energy_and_comm(exp: int, run: int) -> Tuple[List[float], List[float]]:
    avg_energies: List[float] = []
    avg_communications: List[float] = []

    for device in (0, 1):
        comm_rows = []
        comm_path = PROJECT_DIR / f"logs/device_logs/dev{device}_communication_log_exp{exp}_run{run}.csv"
        with comm_path.open("r", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                comm_rows.append(
                    {
                        "timestamp": float(row["Timestamp"]),
                        "round": int(row["CommRound"]),
                        "amount_bytes": float(row["CommAmount [B]"]) if row["CommAmount [B]"] else None,
                    }
                )

        pow_rows = []
        pow_path = PROJECT_DIR / f"logs/device_logs/dev{device}_pow_temp_log_exp{exp}_run{run}.csv"
        with pow_path.open("r", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                pow_rows.append((float(row["Timestamp"]), float(row["Power"])))

        comm_rounds = [row for row in comm_rows if row["round"] > 0]
        round_starts = [row["timestamp"] for row in comm_rows if row["round"] == -1][:-1]
        round_ends = [row["timestamp"] for row in comm_rounds]

        if not comm_rounds or not round_starts:
            raise ValueError(f"Missing communication markers for exp{exp} run{run} dev{device}")

        avg_communications.append(mean([row["amount_bytes"] for row in comm_rounds]) / (1024 * 1024))

        energy_per_round: List[float] = []
        for start, end in zip(round_starts, round_ends):
            energy_per_round.append(sum(power for timestamp, power in pow_rows if start <= timestamp <= end))
        avg_energies.append(mean(energy_per_round))

    return avg_energies, avg_communications


def evaluate_family(
    name: str,
    runs: Sequence[Tuple[int, int]],
    script_lookup: Dict[Tuple[int, int], Dict[str, object]],
) -> Dict[str, object]:
    acc_runs: List[List[float]] = []
    time_runs: List[List[float]] = []
    energy_runs: List[List[float]] = []
    comm_runs: List[List[float]] = []

    for exp, run in runs:
        acc_values, time_values = read_round_log(exp, run)
        acc_runs.append(acc_values)
        time_runs.append(time_values)
        energies, communications = read_energy_and_comm(exp, run)
        energy_runs.append(energies)
        comm_runs.append(communications)

    round_lengths = [len(values) for values in acc_runs]
    config = load_config(runs[-1][0], runs[-1][1], script_lookup)

    if len(set(round_lengths)) != 1:
        return {
            "family": name,
            "runs": list(runs),
            "config": config,
            "status": "skipped",
            "reason": f"incomplete logs ({round_lengths})",
        }

    n_rounds = round_lengths[0]
    mean_acc = [mean([run_values[idx] for run_values in acc_runs]) for idx in range(n_rounds)]
    std_acc = [stddev([run_values[idx] for run_values in acc_runs]) for idx in range(n_rounds)]
    mean_time = [mean([run_values[idx] for run_values in time_runs]) for idx in range(n_rounds)]

    best_idx = max(range(n_rounds), key=lambda idx: mean_acc[idx])
    convergence_round: Optional[int] = None
    for idx in range(n_rounds):
        if mean_acc[idx] >= 90.0 and all(value >= 90.0 for value in mean_acc[idx:]):
            convergence_round = idx
            break

    avg_energy = [mean([run_values[device] for run_values in energy_runs]) for device in range(2)]
    avg_comm = [mean([run_values[device] for run_values in comm_runs]) for device in range(2)]

    return {
        "family": name,
        "runs": list(runs),
        "config": config,
        "status": "completed",
        "convergence_round": convergence_round,
        "avg_time_per_round_s": mean(mean_time[1:]),
        "total_wall_clock_s": sum(mean_time),
        "time_to_90_s": sum(mean_time[: convergence_round + 1]) if convergence_round is not None else None,
        "global_acc_pct": mean_acc[best_idx],
        "global_acc_std_pct": std_acc[best_idx],
        "rpi_energy_j": avg_energy[0],
        "mc1_energy_j": avg_energy[1],
        "total_energy_j": sum(avg_energy),
        "avg_comm_mb": mean(avg_comm),
    }


def format_config(config: Dict[str, object]) -> str:
    return (
        f"{config['loss_type']}, {config['model_name']}, lr={config['learning_rate']}, "
        f"mu={config['mu']}, beta={config['beta']}, "
        f"epochs={config['rpi_local_epochs']}/{config['mc1_local_epochs']}"
    )


def format_runs(runs: Iterable[Tuple[int, int]]) -> str:
    return ", ".join(f"exp{exp}/run{run}" for exp, run in runs)


def render_table4_section(title: str, entry: Dict[str, object]) -> str:
    config = entry["config"]
    aggregation_method = (
        f"{config['loss_type']}, {config['model_name']}, lr={config['learning_rate']}, "
        f"mu={config['mu']}, beta={config['beta']}, "
        f"local_epochs={config['rpi_local_epochs']}/{config['mc1_local_epochs']}"
    )
    return "\n".join(
        [
            f"## {title}",
            "",
            f"Runs used: {format_runs(entry['runs'])}",
            "",
            f"Aggregation method: `{aggregation_method}`",
            "",
            "| Metric | Non-IID |",
            "| --- | ---: |",
            f"| Convergence time [#rounds] | {entry['convergence_round']} |",
            f"| Avg. time per communication round [s] | {entry['avg_time_per_round_s']:.2f} |",
            f"| Total wall clock time [s] | {entry['total_wall_clock_s']:.2f} |",
            f"| Total wall clock time to reach 90.00% [s] | {entry['time_to_90_s']:.2f} |",
            f"| Global test accuracy [%] | {entry['global_acc_pct']:.2f} +/- {entry['global_acc_std_pct']:.2f} |",
            f"| RPI avg. energy consumption per round [J] | {entry['rpi_energy_j']:.2f} |",
            f"| MC1 avg. energy consumption per round [J] | {entry['mc1_energy_j']:.2f} |",
            f"| Total avg. energy consumption per communication round [J] | {entry['total_energy_j']:.2f} |",
            f"| Avg. amount of communicated data per communication round [MB] | {entry['avg_comm_mb']:.3f} |",
            "",
        ]
    )


def build_summary() -> str:
    script_lookup = parse_script_lookup()
    evaluations = [evaluate_family(name, runs, script_lookup) for name, runs in CANDIDATE_FAMILIES.items()]
    completed = [entry for entry in evaluations if entry["status"] == "completed"]
    valid = [entry for entry in completed if entry["convergence_round"] is not None]
    skipped = [entry for entry in evaluations if entry["status"] == "skipped"]

    team_black = min(valid, key=lambda entry: (entry["time_to_90_s"], entry["total_energy_j"]))
    team_green = min(valid, key=lambda entry: (entry["total_energy_j"], entry["time_to_90_s"]))

    lines = [
        "# M3 Table 4 Summary",
        "",
        "Computed from the raw `logs/` and `logs/device_logs/` artifacts in this repo.",
        "",
        "## Ranked Completed Candidates",
        "",
        "| Family | Config | Runs | Conv. round | Time to 90 [s] | Total energy [J] | Global acc [%] |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]

    for entry in sorted(valid, key=lambda item: (item["time_to_90_s"], item["total_energy_j"])):
        lines.append(
            "| "
            f"{entry['family']} | {format_config(entry['config'])} | {format_runs(entry['runs'])} | "
            f"{entry['convergence_round']} | {entry['time_to_90_s']:.2f} | {entry['total_energy_j']:.2f} | "
            f"{entry['global_acc_pct']:.2f} +/- {entry['global_acc_std_pct']:.2f} |"
        )

    lines.extend(
        [
            "",
            render_table4_section("Recommended Table 4 for Team Black", team_black),
            render_table4_section("Recommended Table 4 for Team Green", team_green),
            "## Skipped Families",
            "",
            "These candidates reused the right hyperparameters but did not produce three full 30-round runs,",
            "so they were not considered official Table 4 finalists here.",
            "",
            "| Family | Config | Runs | Reason |",
            "| --- | --- | --- | --- |",
        ]
    )

    for entry in skipped:
        lines.append(
            f"| {entry['family']} | {format_config(entry['config'])} | {format_runs(entry['runs'])} | {entry['reason']} |"
        )

    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize paste-ready M3 Table 4 results from completed 3-run families.")
    parser.add_argument(
        "--output",
        default="m3_table4_summary.md",
        help="Where to write the markdown summary.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_summary()
    output_path = (PROJECT_DIR / args.output).resolve()
    output_path.write_text(summary + "\n", encoding="utf-8")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
