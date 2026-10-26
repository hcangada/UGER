import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from uger.data import ObservedHistory, build_normalized_graph, load_interaction_stream, merge_edges, read_interactions, sample_recommendation_batch, split_prediction_window


class DataTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.tmp_path = Path(directory.name)

    def test_chronology_mapping_and_initial_validation(self):
        tmp_path = self.tmp_path
        source = tmp_path / "events.tsv"
        source.write_text(
            "user_id\titem_id\ttimestamp\n"
            "u2\ti2\t2020-01-03 00:00:00\n"
            "u0\ti0\t2020-01-01 00:00:00.900\n"
            "u1\ti1\t2020-01-01 00:00:00.100\n"
            "u0\ti1\t2020-01-04 00:00:00\n"
            "u3\ti3\t2020-01-05 00:00:00\n"
            "u4\ti4\t2020-01-06 00:00:00\n"
            "u5\ti5\t2020-01-07 00:00:00\n"
            "u6\ti6\t2020-01-08 00:00:00\n"
            "u7\ti7\t2020-01-09 00:00:00\n"
            "u8\ti8\t2020-01-10 00:00:00\n",
            encoding="utf-8",
        )
        stream = load_interaction_stream(source)
        assert stream.user_mapping["u1"] == 0
        assert stream.user_mapping["u0"] == 1
        assert stream.edges.dtype == np.int64
        assert np.all(np.diff(stream.edges[:, 2]) >= 0)
        assert len(stream.snapshots) == 6
        assert [len(snapshot) for snapshot in stream.snapshots] == [6, 1, 1, 1, 1, 0]
        np.testing.assert_array_equal(stream.initial_train, stream.snapshots[0][:-1])
        np.testing.assert_array_equal(stream.initial_validation, stream.snapshots[0][-1:])
        np.testing.assert_array_equal(np.concatenate(stream.snapshots), stream.edges)
        assert stream.fingerprint == load_interaction_stream(source).fingerprint

    def test_equal_time_partition_and_no_validation(self):
        tmp_path = self.tmp_path
        source = tmp_path / "events.tsv"
        source.write_text("user_id\titem_id\ttimestamp\n" + "\n".join(f"u{i}\ti{i}\t{100 + i}" for i in range(10)), encoding="utf-8")
        stream = load_interaction_stream(source, num_incremental=2, split_mode="time", validation_fraction=0)
        assert [len(snapshot) for snapshot in stream.snapshots] == [6, 2, 2]
        assert len(stream.initial_validation) == 0
        np.testing.assert_array_equal(stream.initial_train, stream.snapshots[0])

    def test_prediction_windows_are_temporal_halves(self):
        source = self.tmp_path / "windows.tsv"
        source.write_text("user_id\titem_id\ttimestamp\n" + "\n".join(f"u{i % 3}\ti{i % 7}\t{100 + i}" for i in range(50)), encoding="utf-8")
        stream = load_interaction_stream(source)
        self.assertEqual([len(snapshot) for snapshot in stream.snapshots], [30, 4, 4, 4, 4, 4])
        for snapshot, (validation, test) in zip(stream.snapshots[1:], stream.prediction_windows):
            self.assertEqual((len(validation), len(test)), (2, 2))
            np.testing.assert_array_equal(validation, snapshot[:2])
            np.testing.assert_array_equal(test, snapshot[2:])

    def test_prediction_window_preserves_cross_half_targets(self):
        edges = np.asarray([[0, 1, 4], [0, 2, 1], [0, 1, 2], [0, 3, 3], [1, 0, 5]])
        validation, test = split_prediction_window(edges)
        np.testing.assert_array_equal(validation[:, 2], [1, 2])
        np.testing.assert_array_equal(test[:, 2], [3, 4, 5])
        self.assertIn(1, validation[:, 1])
        self.assertIn(1, test[:, 1])
        self.assertEqual(split_prediction_window([])[0].shape, (0, 3))

    def test_timestamp_precision_preserves_distinct_events(self):
        source = self.tmp_path / "precision.tsv"
        source.write_text("user_id\titem_id\ttimestamp\nu0\ti0\t2020-01-01 00:00:00.900\nu0\ti0\t2020-01-01 00:00:00.100\n", encoding="utf-8")
        edges, _, _ = read_interactions(source)
        self.assertEqual(int(edges[1, 2] - edges[0, 2]), 800_000)
        self.assertEqual(len(merge_edges(edges, edges)), 2)

    def test_stream_deduplicates_events_before_partitioning(self):
        source = self.tmp_path / "duplicates.tsv"
        rows = ["u0\ti0\t100.100", "u0\ti0\t100.900"]
        rows.extend(f"u{index % 3}\ti{index % 4}\t{101 + index}" for index in range(8))
        source.write_text("user_id\titem_id\ttimestamp\n" + "\n".join([*rows, rows[0], rows[-1]]) + "\n", encoding="utf-8")
        raw, users, items = read_interactions(source)
        stream = load_interaction_stream(source)
        self.assertEqual(len(raw), 12)
        self.assertEqual(len(stream.edges), 10)
        self.assertEqual([len(snapshot) for snapshot in stream.snapshots], [6, 1, 1, 1, 1, 0])
        self.assertEqual(stream.user_mapping, users)
        self.assertEqual(stream.item_mapping, items)
        self.assertEqual(int(stream.edges[1, 2] - stream.edges[0, 2]), 800_000)
        np.testing.assert_array_equal(np.concatenate(stream.snapshots), merge_edges(raw))
        self.assertEqual(len(np.unique(np.concatenate(stream.snapshots), axis=0)), 10)

    def test_appending_future_events_preserves_existing_ids(self):
        original = self.tmp_path / "original.tsv"
        extended = self.tmp_path / "extended.tsv"
        content = "user_id\titem_id\ttimestamp\nu1\ti1\t100\nu0\ti0\t101\nu1\ti0\t102\n"
        original.write_text(content, encoding="utf-8")
        extended.write_text(content + "u2\ti2\t200\nu0\ti3\t201\n", encoding="utf-8")
        before, users_before, items_before = read_interactions(original)
        after, users_after, items_after = read_interactions(extended)
        np.testing.assert_array_equal(before, after[:len(before)])
        self.assertTrue(all(users_after[key] == value for key, value in users_before.items()))
        self.assertTrue(all(items_after[key] == value for key, value in items_before.items()))

    def test_history_excludes_only_observed_training_edges(self):
        initial = np.array([[0, 1, 0], [1, 0, 0], [1, 2, 0]])
        future = np.array([[0, 2, 9], [2, 3, 9]])
        history = ObservedHistory(initial)
        np.testing.assert_array_equal(history.eligible_items, [0, 1, 2])
        np.testing.assert_array_equal(history.available_items(0), [0, 2])
        assert 2 in history.sample_unobserved(0, 8, np.random.default_rng(0))
        assert 3 not in history.eligible_items
        history.add(future)
        np.testing.assert_array_equal(history.available_items(0), [0, 3])
        snapshot = history.eligible_items
        snapshot[0] = 999
        assert history.eligible_items[0] == 0

    def test_saturated_users_and_zero_item_are_handled(self):
        edges = np.array([[0, 0, 0], [0, 1, 0], [1, 1, 0]])
        history = ObservedHistory(edges)
        rng = np.random.default_rng(1)
        assert history.sample_unobserved(0, 4, rng).size == 0
        np.testing.assert_array_equal(history.sample_unobserved(1, 4, rng), [0])
        batch = sample_recommendation_batch(edges, history, 4, rng)
        assert batch["users"].tolist() == [1]
        assert batch["items"].tolist() == [[1, 0, 0, 0, 0]]
        assert batch["row_indices"].tolist() == [2]
        empty = sample_recommendation_batch(edges[:2], history, 4, rng)
        assert empty["users"].shape == (0,)
        assert empty["items"].shape == (0, 5)

    def test_recommendation_batch_does_not_draw_the_positive(self):
        history = ObservedHistory([[1, 0, 0], [1, 1, 0], [1, 2, 0]])
        batch = sample_recommendation_batch([[0, 1, 1]], history, 2, np.random.default_rng(0))
        assert batch["items"][0, 0].item() == 1
        assert set(batch["items"][0, 1:].tolist()) == {0, 2}

    def test_binary_normalization_and_isolated_nodes(self):
        edges = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [1, 1, 0]])
        graph = build_normalized_graph(edges, 3, 3)
        assert graph.is_coalesced()
        assert graph._nnz() == 6
        dense = graph.to_dense()
        torch.testing.assert_close(dense, dense.T)
        assert np.isclose(dense[0, 3].item(), 1 / np.sqrt(2))
        assert np.isclose(dense[0, 4].item(), 0.5)
        assert np.isclose(dense[1, 4].item(), 1 / np.sqrt(2))
        assert torch.isfinite(dense).all()
        assert not dense[2].any()
        assert not dense[5].any()
        empty = build_normalized_graph([], 2, 3)
        assert empty.shape == (5, 5)
        assert empty._nnz() == 0

    def test_merge_deduplicates_events_and_preserves_ties(self):
        first = np.array([[1, 1, 2], [0, 0, 1], [1, 1, 2]])
        second = np.array([[2, 2, 2], [0, 0, 3]])
        np.testing.assert_array_equal(merge_edges(first, second), [[0, 0, 1], [1, 1, 2], [2, 2, 2], [0, 0, 3]])
        assert merge_edges().shape == (0, 3)


if __name__ == "__main__":
    unittest.main()
