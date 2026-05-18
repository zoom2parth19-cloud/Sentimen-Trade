"""
SentimenTrade — src package
============================
Exposes the three core engine modules so they can be imported cleanly
from anywhere in the project:

    from src import DataLoader, SentimentAnalyzer
"""

from src.data_loader import DataLoader
from src.sentiment_analyzer import SentimentAnalyzer

__all__ = ["DataLoader", "SentimentAnalyzer"]
__version__ = "0.1.0"
