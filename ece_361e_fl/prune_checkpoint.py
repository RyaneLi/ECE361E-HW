import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

from models.get_model import get_model
from utils.general_utils import get_loss_func
from utils.train_test import test


def load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def infer_defaults(args: argparse.Namespace) -> Tuple[str, str, int]:
    model_name = args.model_name
    loss_type = args.loss_type
    seed = args.seed

    if args.cloud_cfg:
        cfg = load_json(Path(args.cloud_cfg))
        model_name = model_name or str(cfg.get("model_name", "")).strip()
        loss_type = loss_type or str(cfg.get("loss_type", "")).strip()
        seed = seed if seed is not None else int(cfg.get("seed", 2))

    if not model_name:
        raise ValueError("model_name is required (or provide --cloud_cfg with model_name present).")
    if not loss_type:
        loss_type = "fedavg"
    if seed is None:
        seed = 2

    return model_name, loss_type, int(seed)


def collect_modules(
    model: nn.Module,
    include_conv: bool,
    include_linear: bool,
) -> List[Tuple[nn.Module, str, str]]:
    modules: List[Tuple[nn.Module, str, str]] = []
    for name, module in model.named_modules():
        if include_conv and isinstance(module, nn.Conv2d):
            modules.append((module, "weight", name))
        elif include_linear and isinstance(module, nn.Linear):
            modules.append((module, "weight", name))
    if not modules:
        raise ValueError("No prunable modules matched the requested module filters.")
    return modules


def apply_pruning(
    modules: Sequence[Tuple[nn.Module, str, str]],
    method: str,
    amount: float,
) -> None:
    params_to_prune = [(module, param_name) for module, param_name, _ in modules]
    if method == "global_l1":
        prune.global_unstructured(
            params_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=amount,
        )
    elif method == "layerwise_l1":
        for module, param_name, _ in modules:
            prune.l1_unstructured(module, name=param_name, amount=amount)
    else:
        raise ValueError(f"Unsupported pruning method: {method}")


def remove_reparametrization(modules: Sequence[Tuple[nn.Module, str, str]]) -> None:
    for module, param_name, _ in modules:
        prune.remove(module, param_name)


def tensor_sparsity(tensor: torch.Tensor) -> Dict[str, float]:
    zeros = int(torch.count_nonzero(tensor == 0).item())
    total = int(tensor.numel())
    return {
        "zeros": zeros,
        "total": total,
        "sparsity": (zeros / total) if total else 0.0,
    }


def summarize_sparsity(modules: Sequence[Tuple[nn.Module, str, str]]) -> Dict[str, object]:
    per_layer: Dict[str, Dict[str, float]] = {}
    total_zeros = 0
    total_weights = 0
    for module, _, layer_name in modules:
        stats = tensor_sparsity(module.weight.detach())
        per_layer[layer_name] = stats
        total_zeros += int(stats["zeros"])
        total_weights += int(stats["total"])
    return {
        "overall": {
            "zeros": total_zeros,
            "total": total_weights,
            "sparsity": (total_zeros / total_weights) if total_weights else 0.0,
        },
        "per_layer": per_layer,
    }


def evaluate_model(
    model_name: str,
    loss_type: str,
    checkpoint_path: Path,
    loss_name: str,
    cuda_name: str,
    seed: int,
) -> Dict[str, float]:
    model = get_model(model_name=model_name, loss_type=loss_type)
    model.load_state_dict(torch.load(str(checkpoint_path), map_location="cpu"))
    loss_func = get_loss_func(loss_name=loss_name)
    loss, acc = test(
        model=model,
        loss_func=loss_func,
        cuda_name=cuda_name,
        seed=seed,
        loss_type=loss_type,
    )
    return {"loss": float(loss), "acc": float(acc)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply post-hoc pruning to a trained checkpoint.")
    parser.add_argument("--source_checkpoint", type=str, required=True, help="Checkpoint to prune")
    parser.add_argument("--output_checkpoint", type=str, required=True, help="Where to write the pruned checkpoint")
    parser.add_argument("--cloud_cfg", type=str, default="", help="Optional cloud config to infer model/loss/seed")
    parser.add_argument("--model_name", type=str, default="", help="Model architecture name")
    parser.add_argument("--loss_type", type=str, default="", help="Loss type used by the model")
    parser.add_argument("--loss_name", type=str, default="cross_entropy", help="Loss function for optional evaluation")
    parser.add_argument("--seed", type=int, default=None, help="Seed for optional evaluation")
    parser.add_argument("--cuda_name", type=str, default="cpu", help="Device for optional evaluation")
    parser.add_argument("--method", choices=["global_l1", "layerwise_l1"], default="global_l1")
    parser.add_argument("--amount", type=float, required=True, help="Fraction of weights to prune")
    parser.add_argument("--include_conv", action="store_true", help="Prune Conv2d layers")
    parser.add_argument("--include_linear", action="store_true", help="Prune Linear layers")
    parser.add_argument("--eval", action="store_true", help="Evaluate the checkpoint before and after pruning")
    parser.add_argument("--report_json", type=str, default="", help="Optional JSON report path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_name, loss_type, seed = infer_defaults(args)
    source_checkpoint = Path(args.source_checkpoint)
    output_checkpoint = Path(args.output_checkpoint)
    output_checkpoint.parent.mkdir(parents=True, exist_ok=True)

    include_conv = args.include_conv or not args.include_linear
    include_linear = args.include_linear or not args.include_conv

    model = get_model(model_name=model_name, loss_type=loss_type)
    model.load_state_dict(torch.load(str(source_checkpoint), map_location="cpu"))

    baseline_metrics = None
    if args.eval:
        baseline_metrics = evaluate_model(
            model_name=model_name,
            loss_type=loss_type,
            checkpoint_path=source_checkpoint,
            loss_name=args.loss_name,
            cuda_name=args.cuda_name,
            seed=seed,
        )

    modules = collect_modules(model=model, include_conv=include_conv, include_linear=include_linear)
    apply_pruning(modules=modules, method=args.method, amount=args.amount)
    remove_reparametrization(modules)
    sparsity = summarize_sparsity(modules)

    torch.save(model.state_dict(), str(output_checkpoint))

    pruned_metrics = None
    if args.eval:
        pruned_metrics = evaluate_model(
            model_name=model_name,
            loss_type=loss_type,
            checkpoint_path=output_checkpoint,
            loss_name=args.loss_name,
            cuda_name=args.cuda_name,
            seed=seed,
        )

    report = {
        "source_checkpoint": str(source_checkpoint),
        "output_checkpoint": str(output_checkpoint),
        "cloud_cfg": str(args.cloud_cfg) if args.cloud_cfg else "",
        "model_name": model_name,
        "loss_type": loss_type,
        "loss_name": args.loss_name,
        "seed": seed,
        "method": args.method,
        "amount": args.amount,
        "include_conv": include_conv,
        "include_linear": include_linear,
        "sparsity": sparsity,
        "baseline_metrics": baseline_metrics,
        "pruned_metrics": pruned_metrics,
    }

    report_path = Path(args.report_json) if args.report_json else output_checkpoint.with_suffix(".json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    overall = report["sparsity"]["overall"]
    print(
        f"Saved pruned checkpoint to {output_checkpoint} | "
        f"sparsity={overall['sparsity']:.4f} ({overall['zeros']}/{overall['total']})"
    )
    if baseline_metrics:
        print(
            f"Baseline eval: acc={baseline_metrics['acc']:.2f}% "
            f"loss={baseline_metrics['loss']:.4f}"
        )
    if pruned_metrics:
        print(
            f"Pruned eval: acc={pruned_metrics['acc']:.2f}% "
            f"loss={pruned_metrics['loss']:.4f}"
        )
    print(f"Saved pruning report to {report_path}")


if __name__ == "__main__":
    main()
