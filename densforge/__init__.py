"""DensForge: density estimation and manifold learning fused into one closed loop.

Author: 晨星 <CJX0712@users.noreply.github.com>
License: MIT (c) 2026

Layering (strictly acyclic, see ``docs/architecture.md``)::

    L7  cli.py, examples/run_demo.py
    L6  pipeline/
    L5  density/, manifold/
    L4  hpo/
    L3  training/
    L2  eval/, data/
    L1  core/            <- sink: imports nothing from densforge above it
"""

from __future__ import annotations

__version__ = "0.1.0"
__author__ = "晨星"
__email__ = "CJX0712@users.noreply.github.com"
__license__ = "MIT"

#: The single source of truth for the random seed entry point.
#: Everything random in DensForge goes through :func:`densforge.core.seed.set_all`.
SEED_ENTRY_POINT = "densforge.core.seed.set_all"

__all__ = ["SEED_ENTRY_POINT", "__author__", "__email__", "__license__", "__version__"]
