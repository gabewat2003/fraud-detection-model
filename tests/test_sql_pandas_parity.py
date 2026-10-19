"""The SQL window features and an independent pandas implementation must agree."""

import numpy as np
import pytest

from fraud.features_pandas import prior_stats, rolling_count


@pytest.mark.parametrize(
    ("key", "window", "col"),
    [
        ("card_key", 3600, "card_tx_1h"),
        ("card_key", 86400, "card_tx_24h"),
        ("card_key", 604800, "card_tx_7d"),
        ("uid", 86400, "uid_tx_24h"),
    ],
)
def test_velocity_counts_match(feature_df, key, window, col):
    df = feature_df[feature_df[key].notna()]
    expected = rolling_count(df, key, window)
    np.testing.assert_array_equal(df[col].to_numpy(), expected.to_numpy())


def test_prior_stats_match(feature_df):
    df = feature_df[feature_df.uid.notna()]
    ref = prior_stats(df, "uid")
    np.testing.assert_array_equal(df.uid_prior_n.to_numpy(), ref.prior_n.to_numpy())
    np.testing.assert_allclose(df.uid_prior_mean_amt, ref.prior_mean, rtol=1e-5, equal_nan=True)
    np.testing.assert_allclose(
        df.uid_prior_std_amt, ref.prior_std, rtol=1e-4, atol=1e-3, equal_nan=True
    )
