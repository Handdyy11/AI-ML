import numpy as np
import pandas as pd

from src.p2_preprocess.clean import clean_chunk, label_to_family, stratified_sample


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
            " Flow ID": ["a", "b", "c", "d"],
            " Source IP": ["10.0.0.1"] * 4,
            " Destination Port": [80, 22, 80, 443],
            " Flow Bytes/s": [1.5, np.inf, np.nan, 2.5],
            " Fwd Header Length": [20, 20, 20, 20],
            " Fwd Header Length.1": [20, 20, 20, 20],
            " Timestamp": ["3/7/2017 8:55"] * 4,
            " Label": ["BENIGN", "DoS Hulk", "PortScan", "FTP-Patator"],
        }
    )


def test_clean_chunk_fixes_known_quirks():
    out = clean_chunk(make_chunk())
    assert "Destination Port" in out.columns  # spaces stripped
    assert "Fwd Header Length.1" not in out.columns  # duplicate column dropped
    assert len(out) == 2  # inf and NaN rows removed
    assert set(out["family"]) == {"Benign", "BruteForce"}
    assert out["Timestamp"].iloc[0].day == 3 and out["Timestamp"].iloc[0].month == 7


def test_stratified_sample_keeps_every_family():
    df = pd.DataFrame(
        {"family": ["Benign"] * 100 + ["PortScan"] * 20, "x": range(120)}
    )
    out = stratified_sample(df, 0.5)
    counts = out["family"].value_counts()
    assert counts["Benign"] == 50 and counts["PortScan"] == 10