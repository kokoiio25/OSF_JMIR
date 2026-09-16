# Reproducibility materials — JMIR #94706
"Thematic Framing, Geographic Location, and Temporal Patterns in Depression Discourse on a Chinese
Lifestyle Platform: Computational Analysis Using LDA and Sentiment Analysis"

## What is (and is not) shared
The raw scraped comments are NOT redistributed (they contain health-related disclosures by identifiable
users and are subject to platform terms of service). We share the analysis code and the DE-IDENTIFIED
aggregated outputs underlying every table and figure. No file here contains comment IDs or comment text.

**Local inputs (not included).** Two scripts read raw inputs that are NOT redistributed (they contain comment text or IDs): 01_preprocess_lda.py reads data/comments.csv and data/llm_sentiment_labels.csv; 04_snownlp_baseline.py reads data/gold_standard.csv. 03_logistic_regression.py runs entirely from the shared, de-identified aggregated_outputs/ files and reproduces the Table 4 statistics without any restricted input.

## Sentiment measure
The PRIMARY sentiment measure is a validated large-language-model classifier (Claude Opus 4.8), applied
with the prompt in `code/02_sentiment_classification_prompt.md` and validated against a human-coded gold
standard (accuracy 88.3%, macro-F1 0.86, Cohen kappa 0.77). The NTUSD lexicon and SnowNLP are BASELINES
only (Robustness Checks), reported for comparison; both showed low agreement with human coding (kappa <= .19).

## Contents
```
code/
  01_preprocess_lda.py               Jieba segmentation + LDA (K=8, random_state=42, passes=20);
                                     assembles the de-identified analysis dataset. Primary sentiment
                                     comes from the LLM classifier (02), not from a lexicon.
  02_sentiment_classification_prompt.md   LLM (Claude Opus 4.8) prompt, model, and settings — PRIMARY measure
  03_logistic_regression.py          Binary logistic regression (Topic x Time / Topic x Region), statsmodels
  04_snownlp_baseline.py             SnowNLP baseline (Robustness Checks only)
dictionaries/
  (NTUSD_positive.txt, NTUSD_negative.txt for the baseline; optional stopwords_zh.txt) — see READ_ME
aggregated_outputs/
  table1_topic_distribution.csv            Table 1 — 8 topics (Table 2 numbering): n, %, label, top-10 keywords
  table2_topic_sentiment.csv               Table 2 — topic x sentiment (n, %, mean net score)
  table3_timeofday_sentiment.csv           Table 3 — time-of-day x sentiment (N=7,886)
  table4_logistic_dataset.csv              De-identified analysis dataset (negative, topic, hour, tod, region; N=7,891)
  appendix4_region_sentiment.csv           Appendix 4 — sentiment by IP region (>=50 comments)
  classifier_performance_vs_goldstandard.csv   Appendix 2 — classifier accuracy/F1/kappa vs gold standard (N=196)
  LDA_topic_top10_keywords.csv             LDA topic-term probabilities (top 10 per topic)
  coherence_perplexity_by_K.csv            Figure 1 — coherence / log-perplexity across K
  figure4_region_topic_net_sentiment_matrix.csv   Figure 4 — Ward clustering input matrix
```

## Key numbers (must match the manuscript)
Corpus N = 7,891 comments; 7,692 users. Topic 3 (Existential Distress) = 3,310 (41.95%).
Topic x sentiment: chi-square(14, N=7,891) = 600.24, Cramer V = .195.
Time of day: chi-square(6, N=7,886) = 61.4; negative highest overnight (21.8%), lowest morning (12.1%).
Logistic: topic LR chi-square(7) = 580.21; time-of-day LR chi-square(3) = 34.59; Topic x Time n.s. (LR chi-square(21) = 19.19).

## Which file reproduces which table/figure
| Manuscript item | File |
|---|---|
| Table 1 / Figure 2 (topic prevalence) | aggregated_outputs/table1_topic_distribution.csv |
| Table 2 (topic x sentiment) | aggregated_outputs/table2_topic_sentiment.csv |
| Table 3 (time of day x sentiment) | aggregated_outputs/table3_timeofday_sentiment.csv |
| Table 4 (logistic regression) | aggregated_outputs/table4_logistic_dataset.csv (+ code/03_logistic_regression.py) |
| Figure 3 (negative % by region) | aggregated_outputs/appendix4_region_sentiment.csv |
| Figure 4 (Ward clustering of regions) | aggregated_outputs/figure4_region_topic_net_sentiment_matrix.csv |
| Appendix 2 (classifier performance) | aggregated_outputs/classifier_performance_vs_goldstandard.csv |
| Table 1 keywords | aggregated_outputs/LDA_topic_top10_keywords.csv |
| Figure 1 (coherence / log-perplexity vs K) | aggregated_outputs/coherence_perplexity_by_K.csv |

## SnowNLP baseline
SnowNLP 0.12.3 (pretrained Naive Bayes on product reviews) was applied as a baseline. Each comment
received a positive-sentiment probability in [0, 1], trichotomized into negative / neutral / positive
using cutoffs of <0.44 / 0.44-0.47 / >0.47 (these reproduce the SnowNLP row of Appendix 2:
accuracy 0.199, macro-F1 0.202, Cohen kappa 0.044). See code/04_snownlp_baseline.py, a complete runnable script that reads the human-coded gold
standard (data/gold_standard.csv; not redistributed because it contains comment text), applies
SnowNLP, and prints accuracy, macro-F1, and Cohen kappa reproducing the Appendix 2 row.
Package: https://github.com/isnowfy/snownlp

## Environment
See requirements.txt. LDA seed random_state=42. Note: exact LDA topic indices can vary with
gensim/numpy versions; the dominant-topic assignments underlying the manuscript are provided in the
aggregated outputs so downstream results reproduce regardless.
```
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cd aggregated_outputs && python ../code/03_logistic_regression.py   # reproduces Table 4 statistics
```
