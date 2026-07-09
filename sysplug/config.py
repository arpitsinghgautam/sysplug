"""SysPlugConfig: the recommended training configuration dataclass.

The central output type of the SysPlug advisor.  Carries the recommended
hyperparameters together with predicted performance metrics, warnings, and
helper methods for applying the config to popular training frameworks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass


@dataclass
class SysPlugConfig:
    """Recommended training configuration produced by the SysPlug advisor.

    All fields are set by :class:`~sysplug.solver.ConfigSolver`.  Callers
    should not instantiate this directly; use :meth:`Advisor.suggest_config`.

    Attributes:
        batch_size: Per-device micro-batch size.
        gradient_accumulation: Gradient accumulation steps.
        effective_batch_size: ``batch_size × gradient_accumulation × gpu_count``.
        learning_rate: Recommended learning rate.
        precision: Training precision string (``"fp32"`` / ``"fp16"`` /
            ``"bf16"`` / ``"int8"`` / ``"int4"``).
        optimizer: Optimizer name (``"adamw"`` / ``"adam"`` / ``"sgd"`` /
            ``"adafactor"``).
        parallelism: Parallelism strategy.
        use_gradient_checkpointing: Whether gradient checkpointing is advised.
        predicted_peak_memory_mb: Predicted peak GPU memory in MiB.
        predicted_throughput_samples_per_sec: Predicted training throughput.
        safety_margin_pct: Fraction of GPU VRAM reserved as a safety buffer.
        warnings: List of human-readable warning messages.
        notes: List of informational notes about the configuration.
        solver_objective: Objective used during solving
            (``"throughput"`` / ``"memory"`` / ``"balanced"``).
        training_type: Training regime (``"sft"`` / ``"rlhf"`` / etc.).
        gpu_count: Number of GPUs.
        sequence_length: Sequence length assumed during prediction.
        param_count: Model parameter count used for predictions.
    """

    # Core hyperparameters
    batch_size: int = 8
    gradient_accumulation: int = 1
    effective_batch_size: int = 8
    learning_rate: float = 1e-4
    precision: str = "bf16"
    optimizer: str = "adamw"
    parallelism: str = "none"
    use_gradient_checkpointing: bool = False

    # Predicted performance
    predicted_peak_memory_mb: float = 0.0
    # Conservative upper bound (OOM-safe): the solver guarantees this fits.
    predicted_peak_memory_upper_mb: float = 0.0
    predicted_throughput_samples_per_sec: float = 0.0
    safety_margin_pct: float = 0.85

    # Advisor metadata
    warnings: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    solver_objective: str = "balanced"
    training_type: str = "supervised"
    gpu_count: int = 1
    sequence_length: int = 512
    param_count: int = 0

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialise the config to a plain dictionary.

        Returns:
            Dictionary with all config fields.

        Examples:
            >>> cfg = SysPlugConfig(batch_size=4)
            >>> cfg.to_dict()["batch_size"]
            4
        """
        return {
            "batch_size": self.batch_size,
            "gradient_accumulation": self.gradient_accumulation,
            "effective_batch_size": self.effective_batch_size,
            "learning_rate": self.learning_rate,
            "precision": self.precision,
            "optimizer": self.optimizer,
            "parallelism": self.parallelism,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "predicted_peak_memory_mb": self.predicted_peak_memory_mb,
            "predicted_peak_memory_upper_mb": self.predicted_peak_memory_upper_mb,
            "predicted_throughput_samples_per_sec": self.predicted_throughput_samples_per_sec,
            "safety_margin_pct": self.safety_margin_pct,
            "warnings": list(self.warnings),
            "notes": list(self.notes),
            "solver_objective": self.solver_objective,
            "training_type": self.training_type,
            "gpu_count": self.gpu_count,
            "sequence_length": self.sequence_length,
            "param_count": self.param_count,
        }

    # ------------------------------------------------------------------
    # Framework adapters
    # ------------------------------------------------------------------

    def to_training_arguments(self, **kwargs: Any) -> Any:
        """Create a ``transformers.TrainingArguments`` from this config.

        Lazily imports transformers so the core library stays framework-free.

        Args:
            **kwargs: Additional keyword arguments forwarded to
                ``TrainingArguments.__init__``.

        Returns:
            A ``transformers.TrainingArguments`` instance.

        Raises:
            ImportError: If ``transformers`` is not installed.

        Examples:
            >>> cfg = SysPlugConfig(batch_size=4, learning_rate=2e-5)
            >>> # ta = cfg.to_training_arguments(output_dir="/tmp/model")
        """
        try:
            from transformers import TrainingArguments  # type: ignore[import]
        except ImportError:
            raise ImportError(
                "transformers is required. Install it with: pip install sysplug[hf]"
            ) from None

        bf16 = self.precision == "bf16"
        fp16 = self.precision == "fp16"

        return TrainingArguments(
            per_device_train_batch_size=self.batch_size,
            gradient_accumulation_steps=self.gradient_accumulation,
            learning_rate=self.learning_rate,
            bf16=bf16,
            fp16=fp16,
            gradient_checkpointing=self.use_gradient_checkpointing,
            **kwargs,
        )

    def to_deepspeed_config(self, base_config: dict[str, Any] | None = None) -> dict[str, Any]:
        """Merge SysPlugConfig settings into a DeepSpeed config dict.

        Args:
            base_config: Optional existing DeepSpeed config dict to update.
                If ``None``, a minimal config is created.

        Returns:
            A new DeepSpeed config dict with SysPlug settings applied.

        Examples:
            >>> cfg = SysPlugConfig(batch_size=4, precision="bf16", parallelism="zero2")
            >>> ds = cfg.to_deepspeed_config()
            >>> ds["train_micro_batch_size_per_gpu"]
            4
        """
        config: dict[str, Any] = dict(base_config or {})

        config["train_micro_batch_size_per_gpu"] = self.batch_size
        config["gradient_accumulation_steps"] = self.gradient_accumulation
        config["train_batch_size"] = self.batch_size * self.gradient_accumulation * self.gpu_count

        # Precision flags
        if self.precision == "bf16":
            config["bf16"] = {"enabled": True}
            config.pop("fp16", None)
        elif self.precision == "fp16":
            config["fp16"] = {"enabled": True}
            config.pop("bf16", None)
        else:
            config.pop("bf16", None)
            config.pop("fp16", None)

        # ZeRO stage
        zero_stage = {
            "zero1": 1,
            "zero2": 2,
            "zero3": 3,
            "fsdp": 3,
            "none": 0,
            "dp": 0,
            "ddp": 0,
        }.get(self.parallelism, 0)

        if zero_stage > 0:
            config.setdefault("zero_optimization", {})["stage"] = zero_stage

        return config

    def apply_to_optimizer(self, optimizer: Any) -> None:
        """Apply the recommended learning rate to an existing optimizer.

        Args:
            optimizer: A ``torch.optim.Optimizer`` instance.

        Examples:
            >>> import torch
            >>> model = torch.nn.Linear(4, 2)
            >>> opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
            >>> cfg = SysPlugConfig(learning_rate=2e-5)
            >>> cfg.apply_to_optimizer(opt)
            >>> opt.param_groups[0]["lr"]
            2e-05
        """
        for group in optimizer.param_groups:
            group["lr"] = self.learning_rate

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def summary(self, verbose: bool = True) -> str:
        """Return a rich-formatted table summarising the config.

        Args:
            verbose: If ``False``, return a minimal plain-text summary.

        Returns:
            A string with either a Rich table (ANSI) or plain text.

        Examples:
            >>> cfg = SysPlugConfig()
            >>> isinstance(cfg.summary(), str)
            True
        """
        import io

        from rich.console import Console
        from rich.table import Table

        if not verbose:
            return (
                f"SysPlugConfig(batch={self.batch_size}, "
                f"grad_acc={self.gradient_accumulation}, "
                f"lr={self.learning_rate:.2e}, "
                f"precision={self.precision}, "
                f"mem={self.predicted_peak_memory_mb:.0f}MB)"
            )

        table = Table(title="SysPlug Recommended Config", highlight=True)
        table.add_column("Parameter", style="cyan", no_wrap=True)
        table.add_column("Value", style="green")

        rows = [
            ("batch_size", str(self.batch_size)),
            ("gradient_accumulation", str(self.gradient_accumulation)),
            ("effective_batch_size", str(self.effective_batch_size)),
            ("learning_rate", f"{self.learning_rate:.2e}"),
            ("precision", self.precision),
            ("optimizer", self.optimizer),
            ("parallelism", self.parallelism),
            ("gradient_checkpointing", str(self.use_gradient_checkpointing)),
            ("pred. peak memory (MB)", f"{self.predicted_peak_memory_mb:.1f}"),
            ("pred. peak, conservative (MB)", f"{self.predicted_peak_memory_upper_mb:.1f}"),
            ("pred. throughput (samp/s)", f"{self.predicted_throughput_samples_per_sec:.1f}"),
            ("safety margin", f"{self.safety_margin_pct * 100:.0f}%"),
            ("training_type", self.training_type),
        ]
        for name, val in rows:
            table.add_row(name, val)

        buf = io.StringIO()
        console = Console(file=buf, highlight=False, safe_box=True, legacy_windows=False)
        console.print(table)
        result = buf.getvalue()

        if self.warnings:
            result += "\n[Warnings]\n" + "\n".join(f"  [!] {w}" for w in self.warnings)
        if self.notes:
            result += "\n[Notes]\n" + "\n".join(f"  [i] {n}" for n in self.notes)

        return result

    def __repr__(self) -> str:
        return (
            f"SysPlugConfig(batch_size={self.batch_size}, "
            f"grad_acc={self.gradient_accumulation}, "
            f"lr={self.learning_rate:.2e}, "
            f"precision={self.precision!r}, "
            f"peak_mem={self.predicted_peak_memory_mb:.0f}MB, "
            f"warnings={len(self.warnings)})"
        )
