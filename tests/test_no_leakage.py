"""Backward-looking features must never see the current or any later transaction."""

import numpy as np
import pandas as pd

from fraud import db, features, text_norm


def _run_window_sql(rows: pd.DataFrame) -> pd.DataFrame:
    """Run sql/features/03-04 on a hand-built window_input table."""
    conn = db.connect()
    conn.register("_rows", rows)
    conn.execute("CREATE TABLE window_input AS SELECT * FROM _rows")
    db.run_sql(conn, "features/03_velocity.sql")
    db.run_sql(conn, "features/04_card_history.sql")
    out = conn.execute(
        "SELECT * FROM feat_velocity JOIN feat_history USING (TransactionID) ORDER BY TransactionID"
    ).df()
    conn.close()
    return out


def _toy(dts, amts, card="A"):
    n = len(dts)
    return pd.DataFrame(
        {
            "TransactionID": np.arange(1, n + 1),
            "TransactionDT": dts,
            "TransactionAmt": np.asarray(amts, dtype=float),
            "card_key": card,
            "uid": card,
            "card1": 1,
            "addr1": 100.0,
            "p_email_provider": "google",
            "device_family": "apple ios",
        }
    )


def test_first_transaction_has_no_history(feature_df):
    first = feature_df.sort_values(["TransactionDT", "TransactionID"]).groupby("uid").head(1)
    assert (first.uid_prior_n == 0).all()
    assert first.uid_prior_mean_amt.isna().all()
    assert (first.is_first_txn_uid == 1).all()
    first_card = (
        feature_df.sort_values(["TransactionDT", "TransactionID"]).groupby("card_key").head(1)
    )
    assert (first_card.card_tx_24h == 0).all()
    assert first_card.card_prior_mean_amt.isna().all()


def test_counts_exclude_current_and_same_second():
    out = _run_window_sql(_toy([1000, 1000, 1500, 4600, 4601], [10, 20, 30, 40, 50]))
    # rows 1 & 2 share a second: neither sees the other in time windows.
    # row 4 (t=4600) window is [1000, 4599] -> rows 1, 2, 3; row 5 (t=4601) -> rows 3, 4
    assert out.card_tx_1h.tolist() == [0, 0, 2, 3, 2]
    # prior mean is ordered by (DT, ID), so row 2 sees row 1 but never itself
    assert out.uid_prior_n.tolist() == [0, 1, 2, 3, 4]
    np.testing.assert_allclose(out.uid_prior_mean_amt.iloc[1:], [10, 15, 20, 25])
    assert np.isnan(out.uid_prior_mean_amt.iloc[0])


def test_window_boundary_is_inclusive_of_exactly_one_hour():
    out = _run_window_sql(_toy([0, 3600, 3601], [1, 1, 1]))
    assert out.card_tx_1h.tolist() == [0, 1, 1]


def test_future_rows_never_change_past_features():
    rows = _toy([100, 200, 300, 400, 500, 600], [5, 6, 7, 8, 9, 10])
    before = _run_window_sql(rows)
    perturbed = rows.copy()
    perturbed.loc[perturbed.TransactionDT > 300, "TransactionAmt"] *= 1000
    perturbed = pd.concat([perturbed, _toy([700], [1e6]).assign(TransactionID=99)])
    after = _run_window_sql(perturbed)
    cols = [c for c in before.columns if c != "TransactionID"]
    early = before.TransactionID <= 3
    pd.testing.assert_frame_equal(
        before.loc[early, cols].reset_index(drop=True),
        after.loc[after.TransactionID <= 3, cols].reset_index(drop=True),
    )


def test_frequency_tables_only_count_training_window(sample_paths):
    conn = db.connect(sample_paths.db, read_only=True)
    from fraud.io import load_json

    cut = load_json(sample_paths.processed / "split_cutoffs.json")["train_end"]
    n_train = conn.execute(f"SELECT count(*) FROM keyed WHERE TransactionDT <= {cut}").fetchone()[0]
    n_freq = conn.execute("SELECT sum(n) FROM freq_card1").fetchone()[0]
    conn.close()
    assert n_freq == n_train


def test_text_maps_cover_every_distinct_value(sample_paths):
    conn = db.connect(sample_paths.db, read_only=True)
    vals = features.distinct_text_values(conn)
    for kind, raw in vals.items():
        n = conn.execute(f"SELECT count(*) FROM map_{kind}").fetchone()[0]
        assert n == len(text_norm.lookup_table(kind, raw))
    conn.close()
