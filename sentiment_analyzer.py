"""
src/sentiment_analyzer.py
==========================
NLP & Sentiment Engine (FinBERT)
---------------------------------
Wraps the Hugging Face ``ProsusAI/finbert`` model into a clean,
reusable class that scores financial text (news headlines, earnings
call excerpts, analyst reports) and returns structured sentiment
probabilities.

Model background
~~~~~~~~~~~~~~~~
FinBERT (Araci, 2019) is a BERT-base model further pre-trained on
the Financial PhraseBank dataset (~10,000 manually labelled financial
sentences). It outperforms general-purpose sentiment models on
financial text because it understands domain-specific language like
"miss on EPS", "raised guidance", and "credit facility".

Design decisions
~~~~~~~~~~~~~~~~
* **Lazy loading** — the model is NOT loaded at import time. It is
  loaded on the first call to :meth:`score`. This keeps startup time
  fast when the module is imported but not immediately used.
* **Batching** — headlines are processed in configurable mini-batches
  so GPU memory (if available) is used efficiently.
* **Device auto-detection** — the class automatically selects CUDA if
  a GPU is available, falling back to CPU transparently.
* **Scalar score** — alongside raw probabilities, a single scalar
  ``sentiment_score ∈ [-1, +1]`` is computed as
  ``P(positive) - P(negative)`` for easy downstream correlation.

Usage example
~~~~~~~~~~~~~
    >>> from src.sentiment_analyzer import SentimentAnalyzer
    >>> analyzer = SentimentAnalyzer()
    >>> headlines = [
    ...     "Apple beats earnings estimates, raises full-year guidance",
    ...     "Recession fears mount as jobless claims surge unexpectedly",
    ...     "Markets close mixed amid light trading volume",
    ... ]
    >>> results = analyzer.score(headlines)
    >>> print(results[["headline", "sentiment_score", "label"]])
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd
import torch
from transformers import BertForSequenceClassification, BertTokenizer

# ── Module-level logger ──────────────────────────────────────────────────────
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# ── Constants ────────────────────────────────────────────────────────────────
FINBERT_MODEL_NAME: str = "ProsusAI/finbert"

# FinBERT's label order as returned by the model's config.
# Index 0 → positive, 1 → negative, 2 → neutral
LABEL_ORDER: list[str] = ["positive", "negative", "neutral"]

# Maximum tokens FinBERT was trained with (BERT hard limit = 512).
MAX_TOKEN_LENGTH: int = 512

# Default mini-batch size. Reduce to 8 on CPU if RAM is limited.
DEFAULT_BATCH_SIZE: int = 16


class SentimentAnalyzer:
    """FinBERT-powered financial sentiment scorer.

    Parameters
    ----------
    model_name : str, optional
        Hugging Face model identifier. Defaults to
        ``"ProsusAI/finbert"``. Override to use a locally cached path.
    batch_size : int, optional
        Number of headlines to process per inference batch. Larger
        values are faster on GPU; smaller values use less VRAM/RAM.
        Defaults to ``16``.

    Attributes
    ----------
    model_name : str
        The Hugging Face model identifier in use.
    batch_size : int
        Inference batch size.
    device : torch.device
        ``cuda`` if a GPU is detected, otherwise ``cpu``.
    _tokenizer : BertTokenizer or None
        Loaded lazily on first :meth:`score` call.
    _model : BertForSequenceClassification or None
        Loaded lazily on first :meth:`score` call.
    """

    def __init__(
        self,
        model_name: str = FINBERT_MODEL_NAME,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> None:
        self.model_name: str = model_name
        self.batch_size: int = batch_size
        self.device: torch.device = self._resolve_device()

        # These are populated on first call to score() via lazy loading.
        self._tokenizer: Optional[BertTokenizer] = None
        self._model: Optional[BertForSequenceClassification] = None

        logger.info(
            "SentimentAnalyzer initialised — model=%s  device=%s  batch=%d",
            self.model_name,
            self.device,
            self.batch_size,
        )

    # ── Public API ───────────────────────────────────────────────────────────

    def score(self, headlines: list[str]) -> pd.DataFrame:
        """Score a list of financial text strings with FinBERT.

        Each headline is independently classified into one of three
        sentiment classes. Results are returned as a structured
        DataFrame for immediate use in downstream analysis.

        Parameters
        ----------
        headlines : list[str]
            A list of financial text strings. May be news headlines,
            earnings call sentences, or any short financial text
            (ideally under 512 tokens each).

        Returns
        -------
        pd.DataFrame
            One row per headline with the following columns:

            * ``headline``         — the original input text
            * ``prob_positive``    — P(positive) ∈ [0, 1]
            * ``prob_negative``    — P(negative) ∈ [0, 1]
            * ``prob_neutral``     — P(neutral)  ∈ [0, 1]
            * ``sentiment_score``  — scalar score = P(pos) - P(neg) ∈ [-1, 1]
            * ``label``            — argmax class label (str)

        Raises
        ------
        ValueError
            If ``headlines`` is empty or contains only whitespace.

        Examples
        --------
        >>> analyzer = SentimentAnalyzer()
        >>> df = analyzer.score(["Revenue grew 15% YoY, beating estimates."])
        >>> df["label"].iloc[0]
        'positive'
        """
        self._validate_input(headlines)
        self._load_model_if_needed()

        logger.info(
            "Scoring %d headline(s) in batches of %d …",
            len(headlines),
            self.batch_size,
        )

        all_probs: list[dict[str, float]] = []

        # ── Mini-batch inference loop ────────────────────────────────────────
        for batch_start in range(0, len(headlines), self.batch_size):
            batch: list[str] = headlines[batch_start : batch_start + self.batch_size]
            batch_probs: list[dict[str, float]] = self._run_inference(batch)
            all_probs.extend(batch_probs)

        # ── Assemble results DataFrame ───────────────────────────────────────
        result: pd.DataFrame = self._build_result_frame(headlines, all_probs)

        logger.info("Sentiment scoring complete.")
        return result

    def score_single(self, headline: str) -> dict[str, float]:
        """Convenience wrapper to score a single headline.

        Parameters
        ----------
        headline : str
            A single financial text string.

        Returns
        -------
        dict[str, float]
            Dictionary with keys ``prob_positive``, ``prob_negative``,
            ``prob_neutral``, ``sentiment_score``, and ``label``.
        """
        result_df: pd.DataFrame = self.score([headline])
        # Return the first (and only) row as a plain dict.
        return result_df.iloc[0].to_dict()

    # ── Private Helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _resolve_device() -> torch.device:
        """Select CUDA GPU if available, else fall back to CPU.

        Returns
        -------
        torch.device
            The compute device for model inference.
        """
        if torch.cuda.is_available():
            logger.info("GPU detected — using CUDA for inference.")
            return torch.device("cuda")
        logger.info("No GPU detected — using CPU for inference.")
        return torch.device("cpu")

    def _load_model_if_needed(self) -> None:
        """Lazily download and cache the FinBERT tokenizer and model.

        On first call this triggers a Hugging Face Hub download (≈440 MB).
        Subsequent calls are instant because the weights are cached in
        ``~/.cache/huggingface/``.
        """
        if self._model is not None:
            return  # Already loaded — nothing to do.

        logger.info(
            "Loading FinBERT from Hugging Face Hub ('%s'). "
            "This may take a moment on the first run …",
            self.model_name,
        )

        self._tokenizer = BertTokenizer.from_pretrained(self.model_name)
        self._model = BertForSequenceClassification.from_pretrained(
            self.model_name
        )

        # Move model weights to the selected device (GPU or CPU).
        self._model.to(self.device)

        # Set eval mode — disables dropout layers used only during training.
        self._model.eval()

        logger.info("FinBERT loaded and ready.")

    def _run_inference(self, batch: list[str]) -> list[dict[str, float]]:
        """Run a forward pass for a single mini-batch of headlines.

        Parameters
        ----------
        batch : list[str]
            A sub-list of headlines (length ≤ ``self.batch_size``).

        Returns
        -------
        list[dict[str, float]]
            Probability dictionaries for each headline in the batch.
        """
        # Tokenise: pad/truncate all sequences to the same length.
        encoded = self._tokenizer(  # type: ignore[misc]
            batch,
            padding=True,
            truncation=True,
            max_length=MAX_TOKEN_LENGTH,
            return_tensors="pt",  # PyTorch tensors
        )

        # Move input tensors to the same device as the model.
        encoded = {key: val.to(self.device) for key, val in encoded.items()}

        # Inference: disable gradient computation — we only need the
        # forward pass, not backpropagation.
        with torch.no_grad():
            outputs = self._model(**encoded)  # type: ignore[misc]

        # Logits → probabilities via softmax over the class dimension.
        probabilities: torch.Tensor = torch.softmax(outputs.logits, dim=-1)

        # Move back to CPU and convert to Python floats.
        probs_np = probabilities.cpu().numpy()

        results: list[dict[str, float]] = []
        for row in probs_np:
            # row order: [positive, negative, neutral] per LABEL_ORDER
            prob_dict: dict[str, float] = {
                "prob_positive": float(row[0]),
                "prob_negative": float(row[1]),
                "prob_neutral": float(row[2]),
            }
            results.append(prob_dict)

        return results

    @staticmethod
    def _build_result_frame(
        headlines: list[str],
        probs: list[dict[str, float]],
    ) -> pd.DataFrame:
        """Assemble the final output DataFrame.

        Parameters
        ----------
        headlines : list[str]
            Original input headlines (preserving order).
        probs : list[dict[str, float]]
            Probability dictionaries aligned with ``headlines``.

        Returns
        -------
        pd.DataFrame
            Fully annotated results table.
        """
        df: pd.DataFrame = pd.DataFrame(probs)
        df.insert(0, "headline", headlines)

        # Scalar sentiment score: positive probability minus negative.
        # Ranges from -1 (fully negative) to +1 (fully positive).
        df["sentiment_score"] = df["prob_positive"] - df["prob_negative"]

        # Human-readable label = class with highest probability.
        df["label"] = df[
            ["prob_positive", "prob_negative", "prob_neutral"]
        ].idxmax(axis=1).str.replace("prob_", "", regex=False)

        return df

    @staticmethod
    def _validate_input(headlines: list[str]) -> None:
        """Raise ``ValueError`` for empty or blank inputs.

        Parameters
        ----------
        headlines : list[str]
            The input list to validate.
        """
        if not headlines:
            raise ValueError("'headlines' must be a non-empty list of strings.")

        stripped: list[str] = [h.strip() for h in headlines if h.strip()]
        if not stripped:
            raise ValueError(
                "'headlines' contains only empty or whitespace strings."
            )

    # ── Dunder Helpers ───────────────────────────────────────────────────────

    def __repr__(self) -> str:
        loaded: str = "loaded" if self._model is not None else "not loaded"
        return (
            f"SentimentAnalyzer(model='{self.model_name}', "
            f"device='{self.device}', model_weights={loaded})"
        )
