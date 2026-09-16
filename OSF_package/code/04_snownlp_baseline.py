#!/usr/bin/env python3
"""Input (NOT redistributed; contains comment text — see README):
    data/gold_standard.csv  with columns:  content, gold   (gold in {negative, neutral, positive})
Expected output (matches Appendix 2): accuracy 0.199, macro-F1 0.202, Cohen kappa 0.044.
Package version: snownlp 0.12.3   (pip install snownlp==0.12.3)
Tested with: Python 3.11, snownlp 0.12.3, scikit-learn 1.4, pandas 2.x.
"""
from pathlib import Path
import pandas as pd
from snownlp import SnowNLP
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score, classification_report

GOLD = Path("data/gold_standard.csv")
LOW, HIGH = 0.44, 0.47   # p < 0.44 -> negative; 0.44-0.47 -> neutral; p > 0.47 -> positive


def snownlp_prob(text: str) -> float:
    return SnowNLP(str(text)).sentiments


def to_class(p: float) -> str:
    if p < LOW:  return "negative"
    if p > HIGH: return "positive"
    return "neutral"


def main() -> None:
    df = pd.read_csv(GOLD)
    df = df[df["gold"].isin(["negative", "neutral", "positive"])].copy()
    df = df[df["content"].astype(str).str.strip().ne("")].copy()

    df["prob"] = df["content"].map(snownlp_prob)
    df["pred"] = df["prob"].map(to_class)

    y_true, y_pred = df["gold"], df["pred"]
    print(f"N = {len(df)}  (cutoffs: LOW={LOW}, HIGH={HIGH})")
    print(f"accuracy  = {accuracy_score(y_true, y_pred):.3f}")
    print(f"macro-F1  = {f1_score(y_true, y_pred, average='macro', zero_division=0):.3f}")
    print(f"Cohen kappa = {cohen_kappa_score(y_true, y_pred):.3f}")
    print()
    print(classification_report(y_true, y_pred, zero_division=0))


if __name__ == "__main__":
    main()
