"""Which modules the per-module AUPRC scatter highlights."""

import pandas as pd
import pytest

from pflex.plotting import _label_indices, _resolve_include, _select_scatter_modules

# x and y AUPRC of six modules.
SCORES = pd.DataFrame(
    {
        "x": [0.95, 0.90, 0.80, 0.10, 0.50, 0.85],
        "y": [0.10, 0.85, 0.20, 0.70, 0.45, 0.80],
    },
    index=["x_far", "both_best", "x_mid", "y_far", "diagonal", "both_second"],
)


def test_delta_picks_largest_difference_in_each_direction():
    x_only, y_only, shared = _select_scatter_modules(SCORES, "delta", 2, 0)

    assert list(x_only) == ["x_far", "x_mid"]
    assert list(y_only) == ["y_far"]  # only one module is better in y
    assert len(shared) == 0


def test_rank_picks_highest_auprc_per_dataset():
    x_only, y_only, shared = _select_scatter_modules(SCORES, "rank", 2, 0)

    assert list(x_only) == ["x_far", "both_best"]
    assert list(y_only) == ["y_far"]
    assert len(shared) == 0


def test_include_adds_modules_to_the_dataset_where_they_score_higher():
    x_only, y_only, shared = _select_scatter_modules(
        SCORES, "delta", 1, 0, include=["diagonal", "y_far", "x_far"]
    )

    assert list(x_only) == ["x_far", "diagonal"]  # x_far already picked, not repeated
    assert list(y_only) == ["y_far"]
    assert len(shared) == 0


def test_include_keeps_a_shared_module_shared():
    x_only, y_only, shared = _select_scatter_modules(SCORES, "rank", 1, 1, include=["both_best"])

    assert list(shared) == ["both_best"]
    assert "both_best" not in x_only.union(y_only)


def test_resolve_include_matches_names_and_keeps_best_duplicate():
    df = pd.DataFrame(
        {
            "Name": ["Exosome", "Exosome", "NELF complex", "Other"],
            "x": [0.8, 0.6, 0.5, 0.1],
            "y": [0.4, 0.3, 0.8, 0.1],
        },
        index=[10, 11, 12, 13],
    )

    assert _resolve_include(df, ("x", "y"), ["exosome", "NELF complex", "missing"]) == [10, 12]


@pytest.mark.parametrize("highlight", ["delta", "rank"])
def test_shared_are_good_in_both_and_not_reused(highlight):
    x_only, y_only, shared = _select_scatter_modules(SCORES, highlight, 3, 2)

    assert list(shared) == ["both_best", "both_second"]
    groups = [set(x_only), set(y_only), set(shared)]
    assert not (groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])


def test_label_only_restricts_labels_to_named_modules():
    df = pd.DataFrame(
        {"Name": ["A", "B", "C"], "x": [0.9, 0.1, 0.5], "y": [0.1, 0.9, 0.5]},
        index=[1, 2, 3],
    )
    highlighted = pd.Index([1, 2])

    assert list(_label_indices(df, ("x", "y"), highlighted, None)) == [1, 2]
    # C is not highlighted but is still labelled; B is highlighted but not labelled.
    assert list(_label_indices(df, ("x", "y"), highlighted, ["A", "C"])) == [1, 3]
