"""Transparent NLP baselines: multinomial Naive Bayes and TF-IDF retrieval."""

from __future__ import annotations

from collections import Counter, defaultdict
import math
import re
from typing import Iterable

TOKEN_PATTERN = re.compile(r"[A-Za-z0-9']+")


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())


class MultinomialNaiveBayes:
    def __init__(self, alpha: float = 1.0) -> None:
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        self.alpha = alpha

    def fit(self, texts: Iterable[str], labels: Iterable[int]) -> "MultinomialNaiveBayes":
        texts = list(texts)
        labels = list(labels)
        if not texts or len(texts) != len(labels):
            raise ValueError("texts and labels must have the same non-zero length")

        self.classes_ = sorted(set(labels))
        self.vocabulary_: set[str] = set()
        self.class_documents_ = Counter(labels)
        self.class_tokens_: dict[int, Counter[str]] = defaultdict(Counter)

        for text, label in zip(texts, labels):
            tokens = tokenize(text)
            self.vocabulary_.update(tokens)
            self.class_tokens_[label].update(tokens)

        self.total_documents_ = len(texts)
        return self

    def _log_score(self, text: str, label: int) -> float:
        token_counts = self.class_tokens_[label]
        total_tokens = sum(token_counts.values())
        denominator = total_tokens + self.alpha * len(self.vocabulary_)
        prior = self.class_documents_[label] / self.total_documents_
        score = math.log(prior)
        for token, frequency in Counter(tokenize(text)).items():
            probability = (token_counts[token] + self.alpha) / denominator
            score += frequency * math.log(probability)
        return score

    def predict(self, texts: Iterable[str]) -> list[int]:
        if not hasattr(self, "classes_"):
            raise RuntimeError("call fit before prediction")
        return [
            max(self.classes_, key=lambda label: self._log_score(text, label))
            for text in texts
        ]


class TfidfRetriever:
    def fit(self, documents: dict[str, str]) -> "TfidfRetriever":
        if not documents:
            raise ValueError("documents cannot be empty")
        self.documents_ = documents.copy()
        self.tokens_ = {name: tokenize(text) for name, text in documents.items()}
        document_frequency: Counter[str] = Counter()
        for tokens in self.tokens_.values():
            document_frequency.update(set(tokens))
        count = len(documents)
        self.idf_ = {
            token: math.log((count + 1) / (frequency + 1)) + 1.0
            for token, frequency in document_frequency.items()
        }
        return self

    def rank(self, query: str, top_k: int = 3) -> list[tuple[str, float]]:
        if not hasattr(self, "documents_"):
            raise RuntimeError("call fit before ranking")
        query_tokens = set(tokenize(query))
        scored: list[tuple[str, float]] = []
        for name, tokens in self.tokens_.items():
            counts = Counter(tokens)
            length = max(len(tokens), 1)
            score = sum(
                counts[token] / length * self.idf_.get(token, 0.0)
                for token in query_tokens
            )
            scored.append((name, score))
        scored.sort(key=lambda item: (-item[1], item[0]))
        return scored[:top_k]
