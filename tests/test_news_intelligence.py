"""
RakaRAM - News Intelligence Unit Tests
======================================
Tests news fetching, semiconductor ontology tagging, and price multiplier calculations.
"""

import unittest
from src.news.news_fetcher import fetch_dram_news, CURATED_FALLBACK_NEWS
from src.news.news_analyzer import (
    analyze_article,
    analyze_news_articles,
    get_latest_news_analysis,
    detect_generation
)

class TestNewsIntelligence(unittest.TestCase):
    def test_news_fetcher_fallback(self):
        news = fetch_dram_news(force_live=False)
        self.assertIsInstance(news, list)
        self.assertGreaterEqual(len(news), 3)
        sample = news[0]
        self.assertIn("title", sample)
        self.assertIn("source", sample)
        self.assertIn("summary", sample)
        self.assertIn("url", sample)

    def test_detect_generation(self):
        self.assertEqual(detect_generation("Samsung reduces legacy DDR4 wafer line"), "DDR4")
        self.assertEqual(detect_generation("Micron introduces high density DDR5 die"), "DDR5")
        self.assertEqual(detect_generation("Global DRAM pricing outlook across DDR4 and DDR5"), "ALL")

    def test_analyze_article_bullish_hbm(self):
        art = {
            "id": "test-1",
            "title": "Samsung reallocates DDR4 lines to HBM3e for AI servers",
            "source": "TechNews",
            "summary": "Wafer shift away from DDR4 causes supply cut.",
            "url": "https://example.com"
        }
        res = analyze_article(art)
        self.assertEqual(res["impact_direction"], "BULLISH")
        self.assertEqual(res["target_generation"], "DDR4")
        self.assertGreater(res["impact_multiplier_pct"], 0)
        self.assertIn("HBM", res["category"])
        self.assertIn("summary_th", res)

    def test_analyze_article_bearish_yield(self):
        art = {
            "id": "test-2",
            "title": "Micron 1-beta node yield scales up, reducing DDR5 manufacturing costs",
            "source": "TechNews",
            "summary": "Die density improvements lower retail DDR5 prices.",
            "url": "https://example.com"
        }
        res = analyze_article(art)
        self.assertEqual(res["impact_direction"], "BEARISH")
        self.assertEqual(res["target_generation"], "DDR5")
        self.assertLess(res["impact_multiplier_pct"], 0)

    def test_overall_news_analysis(self):
        res = analyze_news_articles(CURATED_FALLBACK_NEWS)
        self.assertIn("overall_status", res)
        self.assertIn("market_sentiment_th", res)
        self.assertIn("net_impact_ddr4_pct", res)
        self.assertIn("net_impact_ddr5_pct", res)
        self.assertIsInstance(res["articles"], list)
        self.assertGreater(len(res["articles"]), 0)

    def test_get_latest_news_analysis(self):
        res = get_latest_news_analysis()
        self.assertIsInstance(res, dict)
        self.assertIn("articles", res)

if __name__ == "__main__":
    unittest.main()
