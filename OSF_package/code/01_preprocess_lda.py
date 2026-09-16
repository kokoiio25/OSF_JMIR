#!/usr/bin/env python3
"""
Preprocessing, LDA topic modeling, and assembly of the de-identified

PIPELINE (matches the manuscript):
  1. Read the raw scraped comments (NOT shared publicly; see README).
  2. Jieba word segmentation; remove stopwords/punctuation/emojis/non-semantic symbols.
  3. LDA (gensim, K = 8, random_state = 42, passes = 20, alpha = 'auto', eta = 'auto';
     dictionary.filter_extremes(no_below = 5, no_above = 0.5)) -> dominant topic per comment.
  4. PRIMARY sentiment is NOT computed here: each comment was labelled negative/neutral/positive
     by the validated large-language-model classifier using the prompt in
     02_sentiment_classification_prompt.md (Claude Opus 4.8), applied via the Anthropic API.
     The NTUSD lexicon and SnowNLP are BASELINES ONLY (Robustness Checks); see 04_snownlp_baseline.py.
  5. Decode posting time, bin into time of day, map IP location to an English region label,
     and write the DE-IDENTIFIED analysis dataset used by 03_logistic_regression.py.

Topic numbering/labels follow Table 2 of the manuscript. NOTE: exact LDA topic indices can vary
with gensim/numpy versions; the dominant-topic assignments underlying the manuscript are provided
in aggregated_outputs/table4_logistic_dataset.csv so that all downstream results are reproducible
even if a re-run relabels topics.

No file written by this script contains comment IDs or comment text.

Tested with: Python 3.11, pandas 2.x, numpy 1.26, jieba 0.42.1, gensim 4.3.3.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import jieba
from gensim import corpora
from gensim.models import LdaModel

RAW_CSV   = Path("data/comments.csv")            # raw comments — NOT redistributed (see README)
LLM_LABELS = Path("data/llm_sentiment_labels.csv")  # comment_id -> {negative,neutral,positive} from 02
STOPWORDS = Path("dictionaries/stopwords_zh.txt")   # optional Chinese stopword list
OUT       = Path("aggregated_outputs/table4_logistic_dataset.csv")

TOPIC_LABELS = {  # paper Topic 1-8 (Table 2)
    1: "Family Dynamics and Parental Relationships",
    2: "Social Anxiety and Interpersonal Struggles",
    3: "Existential Distress and Life Perspective",
    4: "Medical Diagnosis and Healthcare Navigation",
    5: "Workplace Stress & Daily Life Pressures",
    6: "Somatic Anxiety Symptoms",
    7: "Emotional Dysregulation & Physical Distress",
    8: "Clinical Depression and Treatment",
}

ZH2EN = {  # IP location (Chinese) -> English label; extend as needed
    "广东": "Guangdong", "江苏": "Jiangsu", "浙江": "Zhejiang", "山东": "Shandong",
    "四川": "Sichuan", "上海": "Shanghai", "北京": "Beijing", "河南": "Henan",
    "福建": "Fujian", "河北": "Hebei", "湖北": "Hubei", "湖南": "Hunan", "安徽": "Anhui",
    "辽宁": "Liaoning", "广西": "Guangxi", "陕西": "Shaanxi", "重庆": "Chongqing",
    "江西": "Jiangxi", "云南": "Yunnan", "天津": "Tianjin", "黑龙江": "Heilongjiang",
    "吉林": "Jilin", "山西": "Shanxi", "内蒙古": "Inner Mongolia", "贵州": "Guizhou",
    "马来西亚": "Malaysia",  # + other provinces / countries as present in the data
}


def time_of_day(hour: int) -> str:
    if 6 <= hour <= 11:  return "Morning"
    if 12 <= hour <= 17: return "Afternoon"
    if 18 <= hour <= 23: return "Evening"
    return "Overnight"                       # 00:00-05:59


def main() -> None:
    df = pd.read_csv(RAW_CSV)
    df["content"] = df["content"].astype(str)
    df = df[df["content"].str.strip().ne("") & df["content"].str.lower().ne("nan")].copy()

    stop = set()
    if STOPWORDS.exists():
        stop = {w.strip() for w in STOPWORDS.read_text(encoding="utf-8").splitlines() if w.strip()}

    def tokenize(text: str):
        return [w.strip() for w in jieba.lcut(str(text))
                if w.strip() and w.strip() not in stop and not w.strip().isdigit()]

    df["tokens"] = df["content"].map(tokenize)
    df = df[df["tokens"].map(len) > 0].copy()

    # ---- LDA (K = 8) ----
    dictionary = corpora.Dictionary(df["tokens"].tolist())
    dictionary.filter_extremes(no_below=5, no_above=0.5)
    corpus = [dictionary.doc2bow(toks) for toks in df["tokens"]]
    lda = LdaModel(corpus=corpus, id2word=dictionary, num_topics=8,
                   random_state=42, passes=20, alpha="auto", eta="auto")

    def dominant_topic(bow):
        dt = lda.get_document_topics(bow)
        return (int(max(dt, key=lambda kv: kv[1])[0]) + 1) if dt else np.nan  # 1-8

    df["topic"] = [dominant_topic(b) for b in corpus]
    df = df[df["topic"].notna()].copy()
    df["topic"] = df["topic"].astype(int)

    # ---- PRIMARY sentiment: validated LLM classifier (labels produced via 02's prompt) ----
    lab = pd.read_csv(LLM_LABELS)                       # columns: comment_id, sentiment
    df = df.merge(lab, on="comment_id", how="inner")
    df["negative"] = (df["sentiment"] == "negative").astype(int)

    # ---- posting time -> time of day ----
    df["dt"] = pd.to_datetime(df["create_time"], unit="ms", errors="coerce")
    df = df[df["dt"].notna()].copy()
    df["hour"] = df["dt"].dt.hour.astype(int)
    df["tod"] = df["hour"].map(time_of_day)

    # ---- IP location -> English region label ----
    df["region"] = df["ip_location"].map(lambda x: ZH2EN.get(x) if pd.notna(x) else None)

    # ---- write DE-IDENTIFIED dataset (no comment_id, no text) ----
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df[["negative", "topic", "hour", "tod", "region"]].to_csv(OUT, index=False)
    print(f"Saved {OUT}  (n={len(df)}; topics 1-8 = Table 2 labels)")
    for t in range(1, 9):
        print(f"  T{t} {TOPIC_LABELS[t]}: n={(df.topic == t).sum()}")


if __name__ == "__main__":
    main()
