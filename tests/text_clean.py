import numpy as np
import pandas as pd

from src import config
from src.preprocess.clean import (
    build_sample,
    clean_chunk,
    label_to_family,
    smart_chunk_sample,
    stratified_sample,
)


def test_label_to_family():
    assert label_to_family("BENIGN") == "Benign"
    assert label_to_family("FTP-Patator") == "BruteForce"
    assert label_to_family("SSH-Patator") == "BruteForce"
    assert label_to_family("DoS Hulk") == "DoS"
    assert label_to_family("DoS slowloris") == "DoS"
    assert label_to_family("Heartbleed") == "DoS"
    assert label_to_family("PortScan") == "PortScan"
    assert label_to_family("Bot") == "Other"


def make_chunk():
    return pd.DataFrame(
        {
            " Flow ID": ["a", "b", "c", "d", "e"],
            " Source IP": ["10.0.0.1"] * 5,
            " Destination Port": [80, 22, 80, 443, 80],
            " Flow Duration": [100.0, 200.0, 300.0, 400.0, -5.0],  # -5.0 is corrupt
            " Flow Bytes/s": [1.5, np.inf, np.nan, 2.5, 3.0],
            " Fwd Header Length": [20, 20, 20, 20, 20],
            " Fwd Header Length.1": [20, 20, 20, 20, 20],
            " Timestamp": ["3/7/2017 8:55"] * 5,
            " Label": ["BENIGN", "DoS Hulk", "PortScan", "FTP-Patator", "BENIGN"],
        }
    )


def test_clean_chunk_fixes_known_quirks_and_negative_durations():
    out = clean_chunk(make_chunk())
    assert "Destination Port" in out.columns  # spaces stripped
    assert "Fwd Header Length.1" not in out.columns  # duplicate column dropped
    assert len(out) == 2  # inf, NaN, and negative Flow Duration rows removed
    assert set(out["family"]) == {"Benign", "BruteForce"}
    assert out["Timestamp"].iloc[0].day == 3 and out["Timestamp"].iloc[0].month == 7


def test_stratified_sample_keeps_every_family():
    df = pd.DataFrame(
        {"family": ["Benign"] * 100 + ["PortScan"] * 20, "x": range(120)}
    )
    out = stratified_sample(df, 0.5)
    counts = out["family"].value_counts()
    assert counts["Benign"] == 50 and counts["PortScan"] == 10


def test_smart_chunk_sample_keeps_100_percent_rare_attacks():
    df = pd.DataFrame(
        {
            "family": ["Benign"] * 100 + ["BruteForce"] * 12 + ["DoS"] * 40,
            "x": range(152),
        }
    )
    out = smart_chunk_sample(
        df, retention={"Benign": 0.10, "BruteForce": 1.0, "DoS": 0.50}
    )
    counts = out["family"].value_counts()
    assert counts["BruteForce"] == 12
    assert counts["Benign"] == 10
    assert counts["DoS"] == 20


def test_build_sample_mixes_single_attack_files_across_shards(tmp_path, monkeypatch):
    """Simulate CICIDS2017 where Monday has only Benign, Tuesday has BruteForce,
    and Wednesday has DoS. Verify that build_sample mixes all attack types into sample.parquet.
    """
    raw_dir = tmp_path / "raw"
    proc_dir = tmp_path / "processed"
    raw_dir.mkdir()
    proc_dir.mkdir()

    monkeypatch.setattr(config, "DATA_RAW", raw_dir)
    monkeypatch.setattr(config, "DATA_PROCESSED", proc_dir)
    monkeypatch.setattr(config, "SAMPLE_PATH", proc_dir / "sample.parquet")

    rng = np.random.default_rng(0)
    # File 1: Monday (only BENIGN)
    pd.DataFrame(
        {
            "Flow Duration": rng.uniform(10, 1000, 100),
            "Flow Bytes/s": rng.uniform(10, 500, 100),
            "Label": ["BENIGN"] * 100,
        }
    ).to_csv(raw_dir / "Monday.csv", index=False)

    # File 2: Tuesday (BENIGN + SSH-Patator)
    pd.DataFrame(
        {
            "Flow Duration": rng.uniform(10, 1000, 80),
            "Flow Bytes/s": rng.uniform(10, 500, 80),
            "Label": ["BENIGN"] * 50 + ["SSH-Patator"] * 30,
        }
    ).to_csv(raw_dir / "Tuesday.csv", index=False)

    # File 3: Wednesday (BENIGN + DoS Hulk)
    pd.DataFrame(
        {
            "Flow Duration": rng.uniform(10, 1000, 80),
            "Flow Bytes/s": rng.uniform(10, 500, 80),
            "Label": ["BENIGN"] * 40 + ["DoS Hulk"] * 40,
        }
    ).to_csv(raw_dir / "Wednesday.csv", index=False)

    sample, stats = build_sample(raw_dir=raw_dir, chunk_rows=40, num_shards=4)
    assert config.SAMPLE_PATH.exists()
    assert set(sample["family"].unique()) == {"Benign", "BruteForce", "DoS"}
    # All 30 unique BruteForce flows must be preserved
    assert stats["family_counts"]["BruteForce"] == 30