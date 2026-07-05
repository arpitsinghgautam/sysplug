"""SysPlug framework integrations.

Optional integrations for popular training frameworks.  Each sub-module
uses lazy imports so the base ``sysplug`` package does not require
framework dependencies.

Available integrations:
    - :mod:`sysplug.integrations.huggingface` — ``SysPlugTrainerCallback``
    - :mod:`sysplug.integrations.deepspeed` — ``patch_deepspeed_config``
    - :mod:`sysplug.integrations.pytorch` — ``SysPlugContext``,
      ``SysPlugForwardHook``
    - :mod:`sysplug.integrations.rlhf` — ``RLHFAdvisor``, ``PPOConfigHelper``
"""
