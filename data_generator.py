"""
Synthetic Online Payment Fraud Dataset Generator (PaySim-inspired + Modern Fintech Signals).
Generates realistic transaction records for Data Science Programming analysis and ML modeling.
"""

from __future__ import annotations

import os
from pathlib import Path
import numpy as np
import pandas as pd


TRANSACTION_TYPES = [
    "TRANSFER",
    "CASH_OUT",
    "ONLINE_PURCHASE",
    "CARD_PAYMENT",
    "UPI_P2P",
    "BILL_PAYMENT",
]

PAYMENT_CHANNELS = [
    "Mobile App",
    "Web Checkout",
    "UPI / QR",
    "POS Terminal",
    "API Gateway",
]

MERCHANT_CATEGORIES = [
    "E-Commerce & Retail",
    "Electronics & Luxury",
    "Crypto & Digital Assets",
    "Peer-to-Peer Transfer",
    "Travel & Airlines",
    "Gaming & Entertainment",
    "Utilities & Bills",
]


def generate_fraud_dataset(n_samples: int = 6000, random_state: int = 42) -> pd.DataFrame:
    """
    Generate a realistic online payment dataset with domain-accurate fraud patterns:
    - Account takeover (ATO) draining full or near-full sender balances via TRANSFER / CASH_OUT
    - Late-night transaction spikes (00:00 - 05:00)
    - High transaction velocity in 24h window
    - IP country mismatch and elevated device risk scores
    - Discrepancies in balance accounting
    """
    rng = np.random.default_rng(random_state)

    # 1. Transaction IDs & Timestamps over a 30-day window
    txn_ids = [f"TXN-{100000 + i}" for i in range(n_samples)]
    start_ts = pd.Timestamp("2026-09-25 00:00:00")
    random_minutes = rng.integers(0, 30 * 24 * 60, size=n_samples)
    timestamps = pd.Series(pd.to_datetime(start_ts + pd.to_timedelta(random_minutes, unit="m"))).sort_values().reset_index(drop=True)

    hours = timestamps.dt.hour.to_numpy()
    days_of_week = timestamps.dt.day_name().to_numpy()
    is_night = ((hours >= 0) & (hours <= 5)).astype(int)

    # 2. Transaction Type & Channel
    type_probs = [0.16, 0.18, 0.28, 0.18, 0.12, 0.08]
    txn_types = rng.choice(TRANSACTION_TYPES, size=n_samples, p=type_probs)

    channel_probs = [0.38, 0.30, 0.16, 0.10, 0.06]
    channels = rng.choice(PAYMENT_CHANNELS, size=n_samples, p=channel_probs)

    merchant_probs = [0.25, 0.14, 0.10, 0.22, 0.10, 0.10, 0.09]
    merchants = rng.choice(MERCHANT_CATEGORIES, size=n_samples, p=merchant_probs)

    # 3. Initial Balances & Transaction Amounts
    oldbalance_orig = np.round(rng.lognormal(mean=8.2, sigma=1.35, size=n_samples), 2)
    oldbalance_dest = np.round(rng.lognormal(mean=8.5, sigma=1.50, size=n_samples), 2)

    # Base transaction amounts
    amount = np.round(rng.lognormal(mean=5.8, sigma=1.45, size=n_samples), 2)
    amount = np.clip(amount, 5.0, 250000.0)

    # 4. Risk Signals (Device Risk, Velocity, IP Mismatch)
    device_risk_score = np.round(rng.beta(a=1.8, b=6.5, size=n_samples) * 100, 1)
    velocity_24h = rng.poisson(lam=2.2, size=n_samples) + 1
    ip_country_mismatch = rng.binomial(1, p=0.07, size=n_samples)

    # 5. Construct latent fraud probability from realistic domain rules + noise
    logit = np.full(n_samples, -4.6)

    # High-risk transaction types (TRANSFER, CASH_OUT, ONLINE_PURCHASE)
    logit += np.where(np.isin(txn_types, ["TRANSFER", "CASH_OUT"]), 1.65, 0.0)
    logit += np.where(txn_types == "ONLINE_PURCHASE", 0.65, 0.0)
    logit += np.where(np.isin(merchants, ["Crypto & Digital Assets", "Electronics & Luxury"]), 0.95, 0.0)
    logit += np.where(channels == "Web Checkout", 0.45, 0.0)
    logit += np.where(channels == "API Gateway", 0.65, 0.0)

    # Nighttime anomaly
    logit += is_night * 1.15

    # High amount & IP mismatch
    logit += ip_country_mismatch * 1.55
    logit += (device_risk_score > 55).astype(float) * 1.25
    logit += (velocity_24h >= 5).astype(float) * 1.10

    # Convert logit to probability and sample fraud labels (~5.5% - 7.5% fraud rate)
    prob_fraud = 1.0 / (1.0 + np.exp(-logit))
    is_fraud = (rng.uniform(0, 1, size=n_samples) < prob_fraud).astype(int)

    # Inject characteristic PaySim account-draining behavior for a large portion of fraudulent transactions
    fraud_indices = np.where(is_fraud == 1)[0]
    for idx in fraud_indices:
        if rng.uniform() < 0.58:
            # Fraudster attempts to drain 65% - 100% of sender balance
            drain_ratio = rng.uniform(0.65, 1.0)
            amount[idx] = np.round(max(oldbalance_orig[idx] * drain_ratio, 350.0), 2)
        if rng.uniform() < 0.52:
            device_risk_score[idx] = np.round(rng.uniform(52.0, 98.5), 1)
        if rng.uniform() < 0.44:
            velocity_24h[idx] = int(rng.integers(4, 15))
        if rng.uniform() < 0.38:
            ip_country_mismatch[idx] = 1

    # Compute post-transaction balances
    newbalance_orig = np.maximum(0.0, np.round(oldbalance_orig - amount, 2))

    # In some fraudulent or mule accounts, destination balance does not increase as expected (immediate cash-out)
    newbalance_dest = np.round(oldbalance_dest + amount, 2)
    for idx in fraud_indices:
        if rng.uniform() < 0.36:
            # Mule account immediately zeroes or hides incoming funds
            newbalance_dest[idx] = oldbalance_dest[idx]

    # Engineered features
    amount_to_balance_ratio = np.round(amount / (oldbalance_orig + 1.0), 4)
    balance_error_dest = np.round((oldbalance_dest + amount) - newbalance_dest, 2)
    is_account_drained = ((newbalance_orig == 0) & (oldbalance_orig > 500)).astype(int)
    is_flagged_rule = (amount > 50000).astype(int)

    # Assign human-readable risk tier based on composite signals
    composite_score = (
        0.35 * device_risk_score
        + 20.0 * np.clip(amount_to_balance_ratio, 0, 1.5)
        + 15.0 * ip_country_mismatch
        + 12.0 * is_night
        + 3.0 * np.clip(velocity_24h, 0, 12)
    )
    risk_tier = pd.cut(
        composite_score,
        bins=[-np.inf, 25, 45, 68, np.inf],
        labels=["Low", "Medium", "High", "Critical"],
    ).astype(str)

    df = pd.DataFrame(
        {
            "transaction_id": txn_ids,
            "timestamp": timestamps,
            "hour": hours,
            "day_of_week": days_of_week,
            "is_night": is_night,
            "transaction_type": txn_types,
            "payment_channel": channels,
            "merchant_category": merchants,
            "amount": amount,
            "oldbalance_orig": oldbalance_orig,
            "newbalance_orig": newbalance_orig,
            "oldbalance_dest": oldbalance_dest,
            "newbalance_dest": newbalance_dest,
            "amount_to_balance_ratio": amount_to_balance_ratio,
            "balance_error_dest": balance_error_dest,
            "is_account_drained": is_account_drained,
            "device_risk_score": device_risk_score,
            "velocity_24h": velocity_24h,
            "ip_country_mismatch": ip_country_mismatch,
            "is_flagged_rule": is_flagged_rule,
            "risk_tier": risk_tier,
            "is_fraud": is_fraud,
        }
    )
    return df


def load_or_create_dataset(data_path: str | Path = "data/transactions.csv", n_samples: int = 6000) -> pd.DataFrame:
    """Load existing dataset from disk or generate and save it if missing."""
    path = Path(data_path)
    if path.exists():
        df = pd.read_csv(path, parse_dates=["timestamp"])
        return df

    path.parent.mkdir(parents=True, exist_ok=True)
    df = generate_fraud_dataset(n_samples=n_samples)
    df.to_csv(path, index=False)
    return df


if __name__ == "__main__":
    out_file = Path(__file__).parent / "data" / "transactions.csv"
    dataset = load_or_create_dataset(out_file)
    print(f"Generated {len(dataset):,} transactions at {out_file}")
    print(f"Fraud count: {dataset['is_fraud'].sum():,} ({dataset['is_fraud'].mean() * 100:.2f}%)")
