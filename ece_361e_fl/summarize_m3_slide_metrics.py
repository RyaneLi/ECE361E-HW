from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from summarize_m3_table4 import (
    load_config,
    mean,
    parse_script_lookup,
    read_energy_and_comm,
    read_round_log,
)


PROJECT_DIR = Path(__file__).resolve().parent

SLIDE_FAMILIES: Dict[str, List[Tuple[str, int, int]]] = {
    "Team Black": [("Run 1", 198, 1), ("Run 2", 409, 2), ("Run 3", 409, 3)],
    "Team Green": [("Run 1", 164, 1), ("Run 2", 410, 2), ("Run 3", 410, 3)],
}


def convergence_round(acc_values: Sequence[float]) -> Optional[int]:
    for idx, acc in enumerate(acc_values):
        if acc >= 90.0 and all(value >= 90.0 for value in acc_values[idx:]):
            return idx
    return None


def summarize_run(exp: int, run: int) -> Dict[str, float]:
    acc_values, time_values = read_round_log(exp, run)
    energies, communications = read_energy_and_comm(exp, run)
    conv_round = convergence_round(acc_values)
    if conv_round is None:
        raise ValueError(f"exp{exp} run{run} never reaches and maintains 90% accuracy")

    return {
        "avg_time_per_round_s": mean(time_values[1:]),
        "time_to_90_s": sum(time_values[: conv_round + 1]),
        "rpi_energy_j": energies[0],
        "mc1_energy_j": energies[1],
        "total_energy_j": energies[0] + energies[1],
        "avg_comm_mb": mean(communications),
    }


def summarize_average_curve(runs: Sequence[Tuple[int, int]]) -> Dict[str, float]:
    acc_runs: List[List[float]] = []
    time_runs: List[List[float]] = []

    for exp, run in runs:
        acc_values, time_values = read_round_log(exp, run)
        acc_runs.append(acc_values)
        time_runs.append(time_values)

    n_rounds = len(acc_runs[0])
    mean_acc = [mean([run_values[idx] for run_values in acc_runs]) for idx in range(n_rounds)]
    mean_time = [mean([run_values[idx] for run_values in time_runs]) for idx in range(n_rounds)]
    conv_round = convergence_round(mean_acc)
    if conv_round is None:
        raise ValueError("Average accuracy curve never reaches and maintains 90% accuracy")
    return {"conv_round": conv_round, "time_to_90_s": sum(mean_time[: conv_round + 1])}


def format_value(value: float, digits: int = 2) -> str:
    return f"{value:.{digits}f}"


def render_team_section(team_name: str, run_specs: Sequence[Tuple[str, int, int]], script_lookup: Dict) -> str:
    config = load_config(run_specs[-1][1], run_specs[-1][2], script_lookup)
    run_rows = []
    summaries = []
    for label, exp, run in run_specs:
        summary = summarize_run(exp, run)
        summaries.append(summary)
        run_rows.append(
            f"| {label} | exp{exp} run{run} | "
            f"{format_value(summary['avg_time_per_round_s'])} | "
            f"{format_value(summary['time_to_90_s'])} | "
            f"{format_value(summary['rpi_energy_j'])} | "
            f"{format_value(summary['mc1_energy_j'])} | "
            f"{format_value(summary['total_energy_j'])} |"
        )

    average_row = {
        "avg_time_per_round_s": mean([entry["avg_time_per_round_s"] for entry in summaries]),
        "time_to_90_s": mean([entry["time_to_90_s"] for entry in summaries]),
        "rpi_energy_j": mean([entry["rpi_energy_j"] for entry in summaries]),
        "mc1_energy_j": mean([entry["mc1_energy_j"] for entry in summaries]),
        "total_energy_j": mean([entry["total_energy_j"] for entry in summaries]),
    }
    avg_curve = summarize_average_curve([(exp, run) for _, exp, run in run_specs])

    lines = [
        f"## {team_name}",
        "",
        f"Most notable result config: `{config['loss_type']}, {config['model_name']}, "
        f"lr={config['learning_rate']}, mu={config['mu']}, beta={config['beta']}, "
        f"local_epochs={config['rpi_local_epochs']}/{config['mc1_local_epochs']}`",
        "",
        "| Slide row | Source run | Avg. time per communication round [s] | Total wall clock time to reach 90.00% [s] | RPI avg. energy consumption per round [J] | MC1 avg. energy consumption per round [J] | Total avg. energy consumption per communication round [J] |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
        *run_rows,
        f"| Average | arithmetic mean of the 3 runs | "
        f"{format_value(average_row['avg_time_per_round_s'])} | "
        f"{format_value(average_row['time_to_90_s'])} | "
        f"{format_value(average_row['rpi_energy_j'])} | "
        f"{format_value(average_row['mc1_energy_j'])} | "
        f"{format_value(average_row['total_energy_j'])} |",
        "",
        f"Official Table 4 averaged-curve convergence: round `{avg_curve['conv_round']}`",
        f" with `time to 90% = {format_value(avg_curve['time_to_90_s'])} s`.",
        "",
    ]
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate run-by-run slide metrics for the best completed M3 results.")
    parser.add_argument("--output", default="m3_slide_notable_results.md", help="Output markdown path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    script_lookup = parse_script_lookup()

    sections = [
        "# M3 Slide Notable Results",
        "",
        "This file is slide-focused: each table shows the three individual runs plus an average row.",
        "The average row is the arithmetic mean of the three run rows.",
        "",
    ]
    for team_name, run_specs in SLIDE_FAMILIES.items():
        sections.append(render_team_section(team_name, run_specs, script_lookup))

    output_path = (PROJECT_DIR / args.output).resolve()
    output_path.write_text("\n".join(sections) + "\n", encoding="utf-8")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
