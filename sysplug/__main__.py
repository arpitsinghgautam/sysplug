"""Command-line interface for SysPlug.

Usage::

    python -m sysplug --help
    python -m sysplug suggest --model gpt2 --batch-size 8
    python -m sysplug hardware
"""

from __future__ import annotations

import argparse


def _cmd_suggest(args: argparse.Namespace) -> None:
    """Run ``suggest_config`` from the command line."""
    import sysplug

    advisor = sysplug.Advisor(
        model=args.model,
        training_type=args.training_type,
        objective=args.objective,
        verbose=True,
    )
    config: dict = {
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "precision": args.precision,
        "optimizer": args.optimizer,
    }
    if args.grad_acc:
        config["gradient_accumulation"] = args.grad_acc
    advisor.suggest_config(config)


def _cmd_hardware(_args: argparse.Namespace) -> None:
    """Print a hardware summary."""
    from rich.console import Console
    from rich.table import Table

    from sysplug.hardware import HardwareProfiler

    profiler = HardwareProfiler()
    snap = profiler.snapshot()
    console = Console()

    if snap.is_cpu_only:
        console.print("[yellow]No CUDA GPUs found. Running in CPU-only mode.[/yellow]")
        return

    table = Table(title="Hardware Summary")
    table.add_column("Device", style="cyan")
    table.add_column("Name")
    table.add_column("Total (MiB)", justify="right")
    table.add_column("Free (MiB)", justify="right")
    table.add_column("Util %", justify="right")

    for gpu in snap.gpus:
        table.add_row(
            f"GPU {gpu.device_id}",
            gpu.gpu_name,
            f"{gpu.total_memory_mb:.0f}",
            f"{gpu.free_memory_mb:.0f}",
            f"{gpu.gpu_utilization_pct:.0f}%",
        )

    console.print(table)


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="sysplug",
        description="SysPlug: GPU-aware hyperparameter advisor for deep learning.",
    )
    subparsers = parser.add_subparsers(dest="command", help="Sub-command")

    # suggest
    suggest_p = subparsers.add_parser("suggest", help="Suggest training config")
    suggest_p.add_argument("--model", default="gpt2", help="Model name or param count")
    suggest_p.add_argument("--batch-size", type=int, default=8)
    suggest_p.add_argument("--learning-rate", type=float, default=1e-4)
    suggest_p.add_argument("--precision", default="bf16")
    suggest_p.add_argument("--optimizer", default="adamw")
    suggest_p.add_argument("--grad-acc", type=int, default=0)
    suggest_p.add_argument(
        "--training-type", default="supervised",
        choices=["supervised", "sft", "dpo", "rlhf", "grpo"],
    )
    suggest_p.add_argument(
        "--objective", default="balanced",
        choices=["throughput", "memory", "balanced"],
    )

    # hardware
    subparsers.add_parser("hardware", help="Show hardware summary")

    # version
    subparsers.add_parser("version", help="Show SysPlug version")

    args = parser.parse_args()

    if args.command == "suggest":
        _cmd_suggest(args)
    elif args.command == "hardware":
        _cmd_hardware(args)
    elif args.command == "version":
        import sysplug
        print(f"sysplug {sysplug.__version__}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
