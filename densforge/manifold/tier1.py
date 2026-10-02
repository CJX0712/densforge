"""Tier-1 pure-NumPy manifold fallbacks, re-exported from the density tier-1 module.

Author: 晨星 <CJX0712@users.noreply.github.com>

The implementations live in :mod:`densforge.density.tier1` because they share the
partial-eigendecomposition helper and the k-NN graph builder. This module exists so
that ``densforge.manifold`` exposes its own offline tier without the fallback
having to import any scikit-learn code.

All three satisfy the Tier-1 contract:

* no scikit-learn dependency (SciPy linear algebra only);
* partial eigendecomposition via ``subset_by_index`` -- a full ``eigh`` on an
  ``N x N`` matrix costs **6.2x** more at ``N = 2000`` (10.96 s versus 1.77 s);
* ``available()`` returns ``True`` unconditionally, because there is no optional
  backend to probe.
"""

from __future__ import annotations

from ..density.tier1 import (
    NumpyDiffusionMaps,
    NumpyIsomap,
    NumpySpectral,
)

#: Tier-1 manifold fallbacks, in reporting order.
TIER1_MANIFOLD: tuple[type, ...] = (NumpyIsomap, NumpySpectral, NumpyDiffusionMaps)

__all__ = [
    "TIER1_MANIFOLD",
    "NumpyDiffusionMaps",
    "NumpyIsomap",
    "NumpySpectral",
]
