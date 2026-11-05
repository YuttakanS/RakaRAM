"""
RakaRAM - News Intelligence Package
===================================
Fetches and analyzes global semiconductor and DRAM market news
to predict retail hardware price impacts.
"""

from .news_fetcher import fetch_dram_news
from .news_analyzer import analyze_news_articles, get_latest_news_analysis

__all__ = ["fetch_dram_news", "analyze_news_articles", "get_latest_news_analysis"]
