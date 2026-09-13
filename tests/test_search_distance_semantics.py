# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

import math
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import lance
import pyarrow as pa
import pytest
from lance_ray import search as search_mod
from lance_ray.pool import clear_global_pool, init_global_pool
from lance_ray.search import _compute_vector_distances, _get_index_metric, vector_search


@pytest.fixture(scope="module")
def search_pool(ray_context: None) -> Iterator[None]:
    init_global_pool(processes=2)
    try:
        yield
    finally:
        clear_global_pool(close=True)


def _table(ids: list[int], vectors: list[list[float]]) -> pa.Table:
    return pa.table(
        {
            "id": ids,
            "vector": pa.array(vectors, type=pa.list_(pa.float32(), 2)),
        }
    )


@pytest.mark.parametrize("metric", ["l2", "cosine", "dot"])
def test_fallback_distances_match_lance(tmp_path: Path, metric: str) -> None:
    table = _table([0, 1, 2], [[1.5, 0.0], [1.8, 0.6], [-2.0, 1.0]])
    dataset = lance.write_dataset(table, tmp_path / "distances.lance")
    query = [1.0, 0.0]
    native = dataset.to_table(
        columns=["id", "_distance"],
        nearest={
            "column": "vector",
            "q": query,
            "k": 3,
            "metric": metric,
            "use_index": False,
        },
    ).sort_by("id")

    distances = _compute_vector_distances(table["vector"], query, metric)

    assert distances.tolist() == pytest.approx(native["_distance"].to_pylist())


@pytest.mark.parametrize("metric", ["l2", "cosine", "dot"])
@pytest.mark.parametrize("explicit_metric", [True, False])
@pytest.mark.parametrize("filtered", [True, False])
def test_mixed_search_matches_lance(
    tmp_path: Path,
    search_pool: None,
    metric: str,
    explicit_metric: bool,
    filtered: bool,
) -> None:
    # IVF_FLAT with one partition avoids approximation and training randomness.
    # All comparisons exercise both an indexed shard and an appended flat shard.
    if metric == "l2":
        query, indexed, appended = [0.0, 0.0], [1.5, 0.0], [1.8, 0.0]
        fillers = [[10.0 + i, 0.0] for i in range(255)]
    else:
        query, indexed = [1.0, 0.0], [0.9, math.sqrt(0.19)]
        appended = [0.8, 0.6] if explicit_metric else [9.5, math.sqrt(9.75)]
        fillers = [[-1.0 - i * 0.01, 1.0] for i in range(255)]

    uri = tmp_path / "mixed.lance"
    dataset = lance.write_dataset(_table(list(range(256)), [indexed, *fillers]), uri)
    dataset.create_index(
        "vector", index_type="IVF_FLAT", metric=metric, num_partitions=1
    )
    dataset = lance.write_dataset(_table([256], [appended]), uri, mode="append")
    assert len(dataset.get_fragments()) == 2
    assert len(dataset.describe_indices()[0].segments) == 1
    # The indexed and flat shards must both remain on this dataset snapshot.
    lance.write_dataset(_table([257], [query]), uri, mode="append")

    filter_expr = "id != 0" if filtered else None
    for k in (1, 2):
        nearest: dict[str, Any] = {"column": "vector", "q": query, "k": k}
        if explicit_metric:
            nearest["metric"] = metric
        original_nearest = nearest.copy()
        expected = dataset.to_table(
            columns=["id", "_distance"],
            nearest=nearest,
            filter=filter_expr,
            prefilter=True,
        )
        actual = vector_search(
            dataset,
            nearest=nearest,
            columns=["id"],
            filter=filter_expr,
            scanner_options={"prefilter": True},
            num_workers=2,
            include_unindexed=True,
        )

        assert nearest == original_nearest
        assert isinstance(actual, pa.Table)
        assert "vector" not in actual.column_names
        assert actual["id"].to_pylist() == expected["id"].to_pylist()
        assert actual["_distance"].to_pylist() == pytest.approx(
            expected["_distance"].to_pylist(), abs=1e-6
        )


@pytest.mark.parametrize(
    "options", [{"include_unindexed": False}, {"fast_search": True}]
)
def test_indexed_only_search_still_excludes_appends(
    tmp_path: Path, search_pool: None, options: dict[str, Any]
) -> None:
    uri = tmp_path / "indexed_only.lance"
    dataset = lance.write_dataset(_table(list(range(256)), [[1.5, 0.0]] * 256), uri)
    dataset.create_index("vector", index_type="IVF_FLAT", num_partitions=1)
    dataset = lance.write_dataset(_table([256], [[0.0, 0.0]]), uri, mode="append")

    result = vector_search(
        dataset,
        nearest={"column": "vector", "q": [0.0, 0.0], "k": 1},
        columns=["id"],
        num_workers=2,
        **options,
    )

    assert isinstance(result, pa.Table)
    assert result["id"][0].as_py() != 256
    assert result["_distance"].to_pylist() == pytest.approx([2.25])


@pytest.mark.parametrize("metric", [None, "l2", "cosine", "dot"])
def test_unindexed_search_matches_lance(
    tmp_path: Path, search_pool: None, metric: str | None
) -> None:
    uri = tmp_path / "flat.lance"
    lance.write_dataset(_table([0, 1], [[1.5, 0.0], [-2.0, 1.0]]), uri)
    snapshot = lance.write_dataset(_table([2], [[1.8, 0.6]]), uri, mode="append")
    # A later append must not leak into workers searching an older snapshot.
    lance.write_dataset(_table([3], [[1.0, 0.0]]), uri, mode="append")
    nearest: dict[str, Any] = {"column": "vector", "q": [1.0, 0.0], "k": 3}
    if metric is not None:
        nearest["metric"] = metric
    expected = snapshot.to_table(columns=["id", "_distance"], nearest=nearest)

    actual = vector_search(snapshot, nearest=nearest, columns=["id"], num_workers=2)

    assert isinstance(actual, pa.Table)
    assert actual["id"].to_pylist() == expected["id"].to_pylist()
    assert actual["_distance"].to_pylist() == pytest.approx(
        expected["_distance"].to_pylist(), abs=1e-6
    )


@pytest.mark.parametrize(
    "query, k", [([1.0, 0.0], 1), ([1.0, 0.0], 3), ([0.0, 0.0], 3)]
)
def test_zero_norm_cosine_search_matches_lance(
    tmp_path: Path, search_pool: None, query: list[float], k: int
) -> None:
    uri = tmp_path / "zero_norm.lance"
    lance.write_dataset(_table([0, 1], [[0.0, 0.0], [-1.0, 0.0]]), uri)
    dataset = lance.write_dataset(_table([2], [[0.0, 1.0]]), uri, mode="append")
    nearest = {
        "column": "vector",
        "q": query,
        "k": k,
        "metric": "cosine",
        "use_index": False,
    }
    expected = dataset.to_table(columns=["id", "_distance"], nearest=nearest)

    actual = vector_search(dataset, nearest=nearest, columns=["id"], num_workers=2)

    assert isinstance(actual, pa.Table)
    assert actual.schema == expected.schema
    assert actual["id"].to_pylist() == expected["id"].to_pylist()
    assert actual["_distance"].to_pylist() == pytest.approx(
        expected["_distance"].to_pylist()
    )
    # Zero vectors must not displace valid candidates or fill a short result.
    assert actual.num_rows == (min(k, 2) if any(query) else 0)
    assert 0 not in actual["id"].to_pylist()


@pytest.mark.parametrize(
    "metric, nan_query",
    [("cosine", False), ("l2", True), ("dot", True), ("cosine", True)],
)
def test_nan_distance_search_matches_lance(
    tmp_path: Path, search_pool: None, metric: str, nan_query: bool
) -> None:
    uri = tmp_path / "nan.lance"
    lance.write_dataset(_table([0], [[float("nan"), 0.0]]), uri)
    dataset = lance.write_dataset(_table([1], [[1.0, 0.0]]), uri, mode="append")
    nearest = {
        "column": "vector",
        "q": [float("nan") if nan_query else 1.0, 0.0],
        "k": 2,
        "metric": metric,
        "use_index": False,
    }
    expected = dataset.to_table(columns=["id", "_distance"], nearest=nearest)
    actual = vector_search(dataset, nearest=nearest, columns=["id"], num_workers=2)

    assert isinstance(actual, pa.Table)
    assert actual.equals(expected)
    assert actual["id"].to_pylist() == ([] if nan_query else [1])


@pytest.mark.parametrize("metric", ["l2", "dot"])
def test_infinite_distance_search_matches_lance(
    tmp_path: Path, search_pool: None, metric: str
) -> None:
    dataset = lance.write_dataset(
        _table([0, 1, 2], [[float("inf"), 0.0], [1.0, 0.0], [float("nan"), 0.0]]),
        tmp_path / "infinite.lance",
    )
    nearest = {
        "column": "vector",
        "q": [1.0, 0.0],
        "k": 3,
        "metric": metric,
        "use_index": False,
    }
    expected = dataset.to_table(columns=["id", "_distance"], nearest=nearest)
    actual = vector_search(dataset, nearest=nearest, columns=["id"], num_workers=2)

    assert isinstance(actual, pa.Table)
    assert actual.equals(expected)
    assert actual.num_rows == 2
    expected_inf = float("inf") if metric == "l2" else float("-inf")
    assert expected_inf in actual["_distance"].to_pylist()


@pytest.mark.parametrize("as_dict", [True, False])
def test_get_index_metric_uses_manifest_details(as_dict: bool) -> None:
    # No stats API: reading manifest metadata must not open index files.
    dataset: Any = SimpleNamespace()
    values = {"name": "vector_idx", "details": {"metric_type": "COSINE"}}
    index = values if as_dict else SimpleNamespace(**values)
    assert _get_index_metric(dataset, index) == "cosine"


@pytest.mark.parametrize("analyze_plan", [True, False])
def test_mixed_search_only_resolves_metric_for_distance_computation(
    tmp_path: Path,
    search_pool: None,
    monkeypatch: pytest.MonkeyPatch,
    analyze_plan: bool,
) -> None:
    uri = tmp_path / "analyze.lance"
    dataset = lance.write_dataset(_table(list(range(256)), [[1.0, 0.0]] * 256), uri)
    dataset.create_index(
        "vector", index_type="IVF_FLAT", metric="cosine", num_partitions=1
    )
    dataset = lance.write_dataset(_table([256], [[0.8, 0.6]]), uri, mode="append")
    calls = []

    def unavailable_metric(dataset: Any, index: Any) -> str:
        calls.append(index)
        raise ValueError("legacy metric metadata unavailable")

    monkeypatch.setattr(search_mod, "_get_index_metric", unavailable_metric)
    nearest = {"column": "vector", "q": [1.0, 0.0], "k": 1}
    if analyze_plan:
        result = vector_search(
            dataset, nearest=nearest, columns=["id"], num_workers=2, analyze_plan=True
        )
        assert isinstance(result, str)
        assert "(indexed)" in result
        assert "(flat_fallback)" in result
        assert calls == []
    else:
        with pytest.raises(ValueError, match="legacy metric metadata unavailable"):
            vector_search(dataset, nearest=nearest, columns=["id"], num_workers=2)
        assert len(calls) == 1


def test_get_index_metric_reads_stats_for_legacy_index() -> None:
    def index_stats(name: str) -> dict[str, Any]:
        assert name == "legacy_idx"
        return {"indices": [{"metric_type": "DOT"}, {"metric_type": "dot"}]}

    dataset: Any = SimpleNamespace(stats=SimpleNamespace(index_stats=index_stats))
    assert _get_index_metric(dataset, {"name": "legacy_idx"}) == "dot"


@pytest.mark.parametrize(
    "segments",
    [[], [{}], [{"metric_type": "l2"}, {"metric_type": "dot"}]],
)
def test_get_index_metric_rejects_unknown_or_inconsistent_metric(
    segments: list[dict[str, str]],
) -> None:
    dataset: Any = SimpleNamespace(
        stats=SimpleNamespace(index_stats=lambda name: {"indices": segments})
    )
    with pytest.raises(ValueError, match="consistent distance metric.*legacy_idx"):
        _get_index_metric(dataset, {"name": "legacy_idx", "details": {}})
