# LLM sentiment-classification prompt (Multimedia Appendix 3)

Model: Anthropic Claude Opus 4.8 | Reasoning-effort: High | Zero-shot, one comment per request.
Temperature: not explicitly configured (provider default for high reasoning-effort mode).

PROMPT (applied identically to all 7,891 comments):

Classify the sentiment of the following Chinese-language depression-related comment into exactly one
of three categories — negative, neutral, or positive — using these definitions:
- negative: predominantly negative affect (sadness, hopelessness, distress, fear, anger, somatic suffering);
- neutral: informational, factual, help-seeking, or advisory content with no dominant affective charge,
  or mixed/ambiguous tone;
- positive: predominantly positive affect (encouragement, gratitude, hope, recovery, supportive humor).
Consider negation, irony, and platform-specific slang. Output only one word: negative, neutral, or positive.
Comment: "{comment_text}"
