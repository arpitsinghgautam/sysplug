"""Logging utilities for SysPlug.

Provides a shared Rich console and standard Python logger so all modules
emit output consistently and can be silenced via ``verbose=False``.
"""

from __future__ import annotations

import logging
import os
from typing import IO

from rich.console import Console

_console: Console | None = None
_logger: logging.Logger | None = None


def get_console(verbose: bool = True) -> Console:
    """Return the shared Rich console instance.

    Args:
        verbose: If ``False``, returns a console that writes to /dev/null
            (suppresses all output).

    Returns:
        A :class:`rich.console.Console` instance.

    Examples:
        >>> console = get_console(verbose=True)
        >>> console.print("[green]Ready[/green]")
    """
    global _console
    if not verbose:
        return Console(file=open(os.devnull, "w"), stderr=False)  # noqa: SIM115
    if _console is None:
        import io
        import sys as _sys

        # Wrap stdout with UTF-8 encoding to avoid Windows cp1252 issues
        safe_file: IO[str]
        if hasattr(_sys.stdout, "buffer"):
            safe_file = io.TextIOWrapper(
                _sys.stdout.buffer, encoding="utf-8", errors="replace", newline=""
            )
        else:
            safe_file = _sys.stdout
        _console = Console(
            highlight=False,
            safe_box=True,
            legacy_windows=False,
            file=safe_file,
        )
    return _console


def get_logger(name: str = "sysplug") -> logging.Logger:
    """Return a named Python logger configured for SysPlug.

    Args:
        name: Logger name, defaults to ``"sysplug"``.

    Returns:
        A :class:`logging.Logger` instance.

    Examples:
        >>> log = get_logger()
        >>> log.warning("No GPU found, running in CPU mode")
    """
    global _logger
    if _logger is None:
        logger = logging.getLogger(name)
        if not logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(
                logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
            )
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        _logger = logger
    return _logger
