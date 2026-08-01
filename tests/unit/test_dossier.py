"""Tests for the deterministic experiment dossier.

The contract under test is narrow and load-bearing: a dossier never invents. Every
fact it carries must be traceable to something the session printed or a count of
the cell records, because this record is the Summarizer's fallback and a fallback
that could hallucinate would be worse than none.
"""

from __future__ import annotations

from typing import Any

from iterate.core import dossier
from iterate.schemas.experiment import Candidate, Experiment, ExperimentResult, Metrics

_CODE = (
    "from sklearn.preprocessing import OneHotEncoder\n"
    "from sklearn.ensemble import HistGradientBoostingClassifier\n"
    "enc = OneHotEncoder(); model = HistGradientBoostingClassifier()\n"
)


def _cell(
    stdout: str = "",
    *,
    error: str | None = None,
    source: str = "agent",
    timed_out: bool = False,
) -> dict[str, Any]:
    return {
        "code": "x = 1",
        "stdout": stdout,
        "stderr": "",
        "error": error,
        "source": source,
        "timed_out": timed_out,
    }


def _experiment(cells: list[dict[str, Any]], *, score: float | None = 0.61) -> Experiment:
    result = (
        ExperimentResult(
            experiment_id="e",
            metrics=Metrics(
                values={"f1": score}, primary="f1", direction="maximize", n_samples=100
            ),
        )
        if score is not None
        else ExperimentResult(experiment_id="e", error="code-gen contract: no predictions")
    )
    return Experiment(
        candidate=Candidate(
            description="Target encoding attempt",
            changes={"code": _CODE, "cells": cells},
            rationale="r",
        ),
        target="t",
        hypothesis="h",
        status="completed" if score is not None else "failed",
        result=result,
    )


def test_techniques_and_score_come_from_the_deterministic_sources() -> None:
    d = dossier.build(_experiment([_cell("done")], score=0.61))
    assert d.techniques == ["OneHotEncoder", "HistGradientBoostingClassifier"]
    assert d.score == 0.61


def test_data_facts_are_quoted_verbatim_never_paraphrased() -> None:
    """The anti-hallucination contract: every fact must appear literally in stdout."""
    printed = "train shape (1000, 14)\nmissing values: 37\nchurn class balance 0.27"
    d = dossier.build(_experiment([_cell(printed)]))
    for fact in d.data_facts:
        assert fact in printed


def test_progress_chatter_is_not_a_data_fact() -> None:
    d = dossier.build(_experiment([_cell("fitting model\ndone\nsaving")]))
    assert d.data_facts == []


def test_shape_tuples_qualify_without_a_keyword() -> None:
    d = dossier.build(_experiment([_cell("(1000, 14)")]))
    assert d.data_facts == ["(1000, 14)"]


def test_validation_scores_are_not_data_facts() -> None:
    """They have their own field; duplicating them would double-count in render()."""
    d = dossier.build(_experiment([_cell("val f1 0.5500\nrows 1000")]))
    assert d.val_trail == [0.55]
    assert all("0.5500" not in fact for fact in d.data_facts)


def test_val_trail_keeps_order_and_drops_consecutive_repeats() -> None:
    cells = [_cell("val 0.5500"), _cell("val 0.5500"), _cell("val 0.6100"), _cell("val 0.5800")]
    assert dossier.build(_experiment(cells)).val_trail == [0.55, 0.61, 0.58]


def test_failures_dedupe_on_the_error_signature() -> None:
    """Same signature as the coder's own breaker, so both agree on "same failure"."""
    cells = [
        _cell(error="Traceback...\nNameError: name 'Xb' is not defined"),
        _cell(error="Traceback (different)...\nNameError: name 'Xb' is not defined"),
        _cell(error="Traceback...\nValueError: could not convert"),
    ]
    d = dossier.build(_experiment(cells))
    assert d.failures == [
        "NameError: name 'Xb' is not defined",
        "ValueError: could not convert",
    ]


def test_session_shape_counts_errors_and_timeouts() -> None:
    cells = [_cell("ok"), _cell(error="X: y", timed_out=True), _cell(error="Z: w")]
    d = dossier.build(_experiment(cells))
    assert (d.cells_run, d.cells_errored, d.cells_timed_out) == (3, 2, 1)
    assert not d.clean
    assert dossier.build(_experiment([_cell("ok")])).clean


def test_host_cells_are_excluded() -> None:
    """The preamble and the fallback floor are host-authored; crediting the agent
    with what the harness printed would misread every session that banked a floor."""
    cells = [
        _cell("loaded: (96, 2) train / (24, 2) holdout", source="preamble"),
        _cell("fallback baseline banked 24 predictions", source="fallback"),
        _cell("train shape (96, 2)"),
    ]
    d = dossier.build(_experiment(cells))
    assert d.cells_run == 1
    assert d.data_facts == ["train shape (96, 2)"]


def test_a_failed_experiment_still_yields_a_record() -> None:
    d = dossier.build(_experiment([_cell(error="RuntimeError: boom")], score=None))
    assert d.score is None
    assert d.failures == ["RuntimeError: boom"]


def test_build_never_raises_on_a_malformed_session() -> None:
    """This is the path that runs when everything else already degraded."""
    # `changes` cannot be empty (the Candidate schema rejects that), so these are
    # the malformed shapes that are actually reachable at runtime.
    for changes in (
        {"cells": "not-a-list"},
        {"cells": [None, 7, "x"]},
        {"cells": [{"stdout": None, "source": "agent"}]},
        {"code": None, "cells": []},
    ):
        exp = Experiment(
            candidate=Candidate(description="d", changes=changes, rationale="r"),
            target="t",
            hypothesis="h",
            status="failed",
            result=None,
        )
        assert isinstance(dossier.build(exp), dossier.Dossier)


def test_render_is_bounded_and_states_the_session_shape() -> None:
    printed = "train shape (1000, 14)\nval 0.5500"
    text = dossier.build(_experiment([_cell(printed), _cell(error="A: b")])).render()
    assert "used: OneHotEncoder" in text
    assert "observed: train shape (1000, 14)" in text
    assert "validation: 0.5500" in text
    assert "failed: A: b" in text
    assert "2 cells, 1 errored" in text


def test_facts_and_failures_are_capped() -> None:
    many = "\n".join(f"rows {i} shape ({i}, 3)" for i in range(50))
    errors = [_cell(error=f"Err{i}: msg") for i in range(20)]
    d = dossier.build(_experiment([_cell(many), *errors]))
    assert len(d.data_facts) <= 12
    assert len(d.failures) <= 6
