# Lab 3 — NLP and an LLM Agent

## Topics

- Multinomial Naive Bayes sentiment classification
- Tokenization and word-vector processing
- TF–IDF document retrieval
- Attention-related numerical operations
- Prompt-based action selection with an LLM API

## Public implementation

[`src/ai_labs/nlp.py`](../src/ai_labs/nlp.py) provides:

- a transparent multinomial Naive Bayes classifier;
- a TF–IDF retriever with deterministic ranking;
- a small tokenizer suitable for the examples.

No API key, external call, course dataset or submitted answer file is included.

## Key takeaway

An LLM component needs a systems boundary: credentials belong in environment
variables, output formats require validation, and proposed actions must be
checked against the environment's legal-action set.
