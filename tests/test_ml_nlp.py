import unittest
import numpy as np

from ai_labs.ml import BinaryLogisticRegression
from ai_labs.nlp import MultinomialNaiveBayes, TfidfRetriever


class MachineLearningTests(unittest.TestCase):
    def test_logistic_regression_learns_or(self):
        x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
        y = np.array([0, 1, 1, 1])
        model = BinaryLogisticRegression(learning_rate=0.3, epochs=3000).fit(x, y)
        np.testing.assert_array_equal(model.predict(x), y)

    def test_naive_bayes(self):
        texts = ["excellent and clear", "great result", "bad failure", "poor result"]
        labels = [1, 1, 0, 0]
        model = MultinomialNaiveBayes().fit(texts, labels)
        self.assertEqual(model.predict(["great and clear", "bad and poor"]), [1, 0])

    def test_tfidf_ranking(self):
        retriever = TfidfRetriever().fit(
            {
                "search": "A star search uses a heuristic",
                "nlp": "text retrieval uses term frequency",
            }
        )
        self.assertEqual(retriever.rank("heuristic search", top_k=1)[0][0], "search")


if __name__ == "__main__":
    unittest.main()
