import pandas as pd

from fraud import split
from fraud.io import load_json


def test_no_transaction_in_two_splits(feature_df):
    groups = [set(g.TransactionID) for _, g in feature_df.groupby("split")]
    assert sum(len(g) for g in groups) == len(set().union(*groups)) == len(feature_df)


def test_splits_are_time_ordered(feature_df, sample_paths):
    cut = load_json(sample_paths.processed / "split_cutoffs.json")
    ranges = feature_df.groupby("split").TransactionDT.agg(["min", "max"])
    order = split.SPLITS
    for a, b in zip(order, order[1:], strict=False):
        assert ranges.loc[a, "max"] < ranges.loc[b, "min"]
    assert ranges.loc["train", "max"] <= cut["train_end"] < ranges.loc["valid", "min"]


def test_split_fractions_roughly_match_config(feature_df):
    share = feature_df.split.value_counts(normalize=True)
    assert abs(share["train"] - 0.60) < 0.02
    assert abs(share["test"] - 0.15) < 0.02


def test_assign_split_matches_sql(feature_df, sample_paths):
    cut = load_json(sample_paths.processed / "split_cutoffs.json")
    assert (split.assign_split(feature_df.TransactionDT, cut) == feature_df.split).all()


def test_new_uid_flag_only_on_unseen_test_accounts(feature_df):
    flag = split.new_uid_flag(feature_df)
    assert not flag[feature_df.split != "test"].any()
    seen = set(feature_df.loc[feature_df.split != "test", "uid"].dropna())
    assert not feature_df.loc[flag, "uid"].isin(seen).any()


def test_split_summary_has_every_split(feature_df):
    s = split.split_summary(feature_df)
    assert list(s.index) == split.SPLITS
    assert isinstance(s, pd.DataFrame) and (s.rows > 0).all()
