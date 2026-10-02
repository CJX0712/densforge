"""L2 eval -- metrics and the leakage firewall.

Author: 晨星 <CJX0712@users.noreply.github.com>
"""

from __future__ import annotations

from .leakage import (
    ScoreRecorder,
    assert_no_overlap,
    assert_seed_isolation,
    assert_split_roles,
    assert_test_untouched,
    assert_train_only_model,
)
from .metrics import (
    TRUSTWORTHNESS_K,
    aggregate,
    calibration_error,
    continuity_score,
    format_mean_std,
    is_significant,
    local_structure_fidelity,
    mean_or_none,
    nll,
    nll_train_loo,
    summarize,
    trustworthiness_score,
)

__all__ = [
    "TRUSTWORTHNESS_K",
    "ScoreRecorder",
    "aggregate",
    "assert_no_overlap",
    "assert_seed_isolation",
    "assert_split_roles",
    "assert_test_untouched",
    "assert_train_only_model",
    "calibration_error",
    "continuity_score",
    "format_mean_std",
    "is_significant",
    "local_structure_fidelity",
    "mean_or_none",
    "nll",
    "nll_train_loo",
    "summarize",
    "trustworthiness_score",
]
