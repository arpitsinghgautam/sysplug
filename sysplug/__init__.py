"""SysPlug: GPU-aware hyperparameter advisor for deep learning training.

Quick start::

    import sysplug

    advisor = sysplug.Advisor(model=model, training_type="sft")
    cfg = advisor.suggest_config({"batch_size": 8, "learning_rate": 2e-5})
    print(cfg.summary())

    # What-if analysis
    result = advisor.what_if({"batch_size": 32})

    # Online monitoring
    with advisor.monitor(check_interval_steps=100) as mon:
        for step, batch in enumerate(dataloader):
            loss = train_step(batch)
            mon.record(step=step, loss=loss.item())
"""

from sysplug.advisor import Advisor, WhatIfResult
from sysplug.config import SysPlugConfig
from sysplug.hardware import GPUSnapshot, HardwareProfiler, HardwareSnapshot
from sysplug.memory_model import MemoryModel, PrecisionMode
from sysplug.monitor import Monitor, MonitorEvent
from sysplug.solver import ConfigSolver, SolverConstraints
from sysplug.stability import StabilitySignal
from sysplug.throughput_model import ThroughputModel

__all__ = [
    "Advisor",
    "WhatIfResult",
    "SysPlugConfig",
    "HardwareProfiler",
    "HardwareSnapshot",
    "GPUSnapshot",
    "MemoryModel",
    "PrecisionMode",
    "Monitor",
    "MonitorEvent",
    "ConfigSolver",
    "SolverConstraints",
    "StabilitySignal",
    "ThroughputModel",
]

__version__ = "0.1.0"
__author__ = "Arpit Singh Gautam"
__email__ = "arpitsinghgautam777@gmail.com"
