"""
tests/test_sentiment_analyzer.py
==================================
Unit tests for src/sentiment_analyzer.py

These tests use small synthetic inputs and do NOT load the real FinBERT
model (which is ~440 MB) during CI. The model-loading path is covered
by a separate integration test marked with @pytest.mark.slow.

Run fast tests only:
    pytest tests/test_sentiment_analyzer.py -v -m "not slow"

Run all tests including model download:
    pytest tests/test_sentiment_analyzer.py -v
"""

import pandas as pd
import pytest

from src.sentiment_analyzer import SentimentAnalyzer


# ── Fixtures ─────────────────────────────────────────────────────────────────

SAMPLE_HEADLINES: list[str] = [
    "Apple beats earnings estimates and raises full-year guidance.",
    "Markets crash as inflation data comes in hotter than expected.",
    "Trading volume was light ahead of the long weekend.",
]


# ── Unit Tests (no model loading) ─────────────────────────────────────────────

class TestSentimentAnalyzerInit:
    def test_default_model_name(self) -> None:
        analyzer = SentimentAnalyzer()
        assert "finbert" in analyzer.model_name.lower()

    def test_model_not_loaded_at_init(self) -> None:
        analyzer = SentimentAnalyzer()
        # Model should be None until score() is called (lazy loading).
        assert analyzer._model is None
        assert analyzer._tokenizer is None

    def test_repr_shows_not_loaded(self) -> None:
        analyzer = SentimentAnalyzer()
        assert "not loaded" in repr(analyzer)

    def test_custom_batch_size(self) -> None:
        analyzer = SentimentAnalyzer(batch_size=8)
        assert analyzer.batch_size == 8


class TestValidation:
    def test_raises_on_empty_list(self) -> None:
        analyzer = SentimentAnalyzer()
        with pytest.raises(ValueError, match="non-empty"):
            analyzer.score([])

    def test_raises_on_blank_strings(self) -> None:
        analyzer = SentimentAnalyzer()
        with pytest.raises(ValueError, match="whitespace"):
            analyzer.score(["   ", "\t", ""])


class TestResultStructure:
    """Test the shape and types of results using a mocked _run_inference."""

    def test_output_columns(self, mocker) -> None:
        """Patch _run_inference to avoid loading the real model."""
        analyzer = SentimentAnalyzer()

        # Stub out model loading and inference.
        mocker.patch.object(analyzer, "_load_model_if_needed")
        mocker.patch.object(
            analyzer,
            "_run_inference",
            return_value=[
                {"prob_positive": 0.8, "prob_negative": 0.1, "prob_neutral": 0.1},
                {"prob_positive": 0.1, "prob_negative": 0.7, "prob_neutral": 0.2},
            ],
        )

        result: pd.DataFrame = analyzer.score(SAMPLE_HEADLINES[:2])
        expected_cols = {
            "headline", "prob_positive", "prob_negative",
            "prob_neutral", "sentiment_score", "label",
        }
        assert expected_cols.issubset(set(result.columns))

    def test_sentiment_score_range(self, mocker) -> None:
        analyzer = SentimentAnalyzer()
        mocker.patch.object(analyzer, "_load_model_if_needed")
        mocker.patch.object(
            analyzer,
            "_run_inference",
            return_value=[
                {"prob_positive": 0.8, "prob_negative": 0.1, "prob_neutral": 0.1},
            ],
        )
        result = analyzer.score(["Any headline."])
        score = result["sentiment_score"].iloc[0]
        assert -1.0 <= score <= 1.0

    def test_label_is_argmax(self, mocker) -> None:
        analyzer = SentimentAnalyzer()
        mocker.patch.object(analyzer, "_load_model_if_needed")
        mocker.patch.object(
            analyzer,
            "_run_inference",
            return_value=[
                {"prob_positive": 0.05, "prob_negative": 0.90, "prob_neutral": 0.05},
            ],
        )
        result = analyzer.score(["Terrible news."])
        assert result["label"].iloc[0] == "negative"

    def test_row_count_matches_input(self, mocker) -> None:
        analyzer = SentimentAnalyzer()
        mocker.patch.object(analyzer, "_load_model_if_needed")
        mocker.patch.object(
            analyzer,
            "_run_inference",
            side_effect=lambda batch: [
                {"prob_positive": 0.33, "prob_negative": 0.33, "prob_neutral": 0.34}
                for _ in batch
            ],
        )
        result = analyzer.score(SAMPLE_HEADLINES)
        assert len(result) == len(SAMPLE_HEADLINES)


# ── Integration Test (downloads real model ~440 MB) ───────────────────────────

@pytest.mark.slow
class TestSentimentAnalyzerIntegration:
    """Requires internet access. Skipped in fast CI runs."""

    @pytest.fixture(scope="class")
    def analyzer(self) -> SentimentAnalyzer:
        return SentimentAnalyzer()

    def test_positive_headline_scores_positive(
        self, analyzer: SentimentAnalyzer
    ) -> None:
        result = analyzer.score(
            ["Company reports record profits, raises dividend by 20%."]
        )
        assert result["label"].iloc[0] == "positive"
        assert result["sentiment_score"].iloc[0] > 0

    def test_negative_headline_scores_negative(
        self, analyzer: SentimentAnalyzer
    ) -> None:
        result = analyzer.score(
            ["Firm files for bankruptcy amid massive accounting fraud scandal."]
        )
        assert result["label"].iloc[0] == "negative"
        assert result["sentiment_score"].iloc[0] < 0

    def test_probabilities_sum_to_one(
        self, analyzer: SentimentAnalyzer
    ) -> None:
        result = analyzer.score(["Markets closed flat on low volume."])
        total = (
            result["prob_positive"].iloc[0]
            + result["prob_negative"].iloc[0]
            + result["prob_neutral"].iloc[0]
        )
        assert abs(total - 1.0) < 1e-5
