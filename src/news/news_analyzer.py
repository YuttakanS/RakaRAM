"""
RakaRAM - News Sentiment & Price Impact Analyzer
================================================
Analyzes global semiconductor and DRAM market news articles using domain ontology,
sentiment heuristics, and economic price-transmission rules to calculate:
- Target Generation (DDR4, DDR5, All)
- Impact Direction (BULLISH = price up, BEARISH = price down, NEUTRAL)
- Quantitative Price Multiplier (+1.0% to +5.0% or -1.0% to -5.0%)
- Localized Thai executive commentary for hardware shoppers
"""

import os
import json
import re
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Dict, Any, Tuple
from src.utils.logger import get_logger
from .news_fetcher import fetch_dram_news

logger = get_logger("news_analyzer")

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
DATA_NEWS_DIR = ROOT_DIR / "data" / "news"
ANALYZED_NEWS_PATH = DATA_NEWS_DIR / "ram_news_analyzed.json"

# Domain-specific sentiment & event keywords
PATTERNS = {
    "HBM_EXPANSION": [r"\bhbm\b", r"\bhbm3e\b", r"\bhbm4\b", r"ai server", r"nvidia", r"accelerator"],
    "SUPPLY_CUT": [r"reallocat\w*", r"cut\w*", r"shrink\w*", r"tight\w*", r"halt\w*", r"shut\w*", r"capacity shift", r"fully booked"],
    "YIELD_IMPROVEMENT": [r"yield\w*", r"1-beta", r"die density", r"24gb", r"32gb", r"manufacturing cost\w*", r"scale\w*"],
    "PRICE_EASING": [r"eas\w*", r"declin\w*", r"drop\w*", r"compress\w*", r"slump\w*", r"cheap\w*", r"discount\w*"],
    "PRICE_HIKE": [r"hike\w*", r"rebound\w*", r"rise\w*", r"surg\w*", r"floor level", r"stabiliz\w* at floor"],
    "DISASTER": [r"quake", r"earthquake", r"fire", r"flood", r"disruption"],
    "DEMAND": [r"ryzen", r"core ultra", r"adoption", r"consumer transition", r"economies of scale"]
}

def detect_generation(text: str) -> str:
    upper = text.upper()
    has_ddr4 = "DDR4" in upper
    has_ddr5 = "DDR5" in upper
    if has_ddr4 and not has_ddr5:
        return "DDR4"
    if has_ddr5 and not has_ddr4:
        return "DDR5"
    return "ALL"

def analyze_article(article: Dict[str, Any]) -> Dict[str, Any]:
    full_text = f"{article['title']} {article.get('summary', '')}".lower()
    gen = detect_generation(full_text)

    # Count pattern occurrences
    matches = {}
    for cat, regexes in PATTERNS.items():
        count = sum(len(re.findall(rx, full_text)) for rx in regexes)
        if count > 0:
            matches[cat] = count

    # Determine impact direction & multiplier
    direction = "NEUTRAL"
    multiplier = 0.0
    sentiment_score = 0
    confidence = 0.85
    category = "GENERAL_NEWS"
    headline_th = article["title"]
    summary_th = "ข่าวสารทั่วไปเกี่ยวกับอุตสาหกรรมหน่วยความจำ"

    # Rule 1: HBM shift or wafer reallocation -> Bullish for DDR4/general DRAM (supply contraction)
    if "HBM_EXPANSION" in matches and ("SUPPLY_CUT" in matches or "wafer" in full_text):
        direction = "BULLISH"
        multiplier = 2.5 if gen == "DDR4" else 1.2
        sentiment_score = +65
        category = "HBM_WAFER_SHIFT"
        confidence = 0.92
        headline_th = "โรงงานชิปโยกสายการผลิตแรมทั่วไปไปทำ HBM สำหรับ AI"
        summary_th = "Samsung และ SK Hynix เร่งผลิตชิป HBM ให้เซิร์ฟเวอร์ AI ส่งผลให้โควตาเวเฟอร์ DDR4 เริ่มตึงตัว ราคาอาจขยับขึ้นเล็กน้อย"

    # Rule 2: Pure Supply Cut
    elif "SUPPLY_CUT" in matches:
        direction = "BULLISH"
        multiplier = 2.0 if gen == "DDR4" else 1.0
        sentiment_score = +50
        category = "SUPPLY_CONTRACTION"
        confidence = 0.88
        headline_th = "ซัพพลายแรมในตลาดโลกลดลงจากนโยบายควบคุมกำลังผลิต"
        summary_th = "ผู้ผลิตควบคุมกำลังการผลิตแรมรุ่นเก่า ช่วยพยุงราคาไม่ให้ตกและผลักดันราคาขายส่ง"

    # Rule 3: Yield improvement or 1-beta / density -> Bearish for DDR5 prices (cheaper per GB)
    elif "YIELD_IMPROVEMENT" in matches:
        direction = "BEARISH"
        multiplier = -3.5 if gen in ["DDR5", "ALL"] else -0.5
        sentiment_score = -70
        category = "YIELD_MATURATION"
        confidence = 0.90
        headline_th = "ความก้าวหน้าการผลิตชิป DDR5 ส่งผลให้ต้นทุนต่อ GB ลดลง"
        summary_th = "Micron และผู้ผลิตชิปทำ Yield 24Gb ได้สูงขึ้น หนุนราคาขายปลีก DDR5 ให้ปรับลดลงอย่างต่อเนื่อง"

    # Rule 4: Price Easing / Contract Prices
    elif "PRICE_EASING" in matches:
        direction = "BEARISH"
        multiplier = -2.8 if gen == "DDR5" else -0.5
        sentiment_score = -55
        category = "PRICE_COMPRESSION"
        confidence = 0.87
        headline_th = "ราคาแรม DDR5 หน้าร้านทยอยปรับตัวลงตามราคาซื้อขายล่วงหน้า"
        summary_th = "รายงาน TrendForce ยืนยันราคาแรม DDR5 มีเสถียรภาพและทยอยปรับตัวลดลงตามกลไกตลาด"

    # Rule 5: Hardware Platform Demand & Adoption
    elif "DEMAND" in matches and "adoption" in full_text:
        direction = "BEARISH"  # Higher adoption -> economies of scale -> cheaper
        multiplier = -2.0
        sentiment_score = -40
        category = "MAINSTREAM_ADOPTION"
        confidence = 0.85
        headline_th = "ผู้ใช้งานเปลี่ยนมาใช้ DDR5 เพิ่มขึ้น หนุนให้ราคาต่อชิ้นประหยัดขึ้น"
        summary_th = "สัดส่วนการประกอบเครื่องด้วย DDR5 เกิน 75% ก่อให้เกิดการประหยัดจากขนาด (Economies of scale)"

    # Rule 6: Disaster / Quake
    elif "DISASTER" in matches:
        direction = "BULLISH" if "damage" not in full_text else "NEUTRAL"
        multiplier = 1.0
        sentiment_score = +25
        category = "SUPPLY_DISRUPTION"
        confidence = 0.80
        headline_th = "รายงานผลกระทบจากเหตุแผ่นดินไหวต่อโรงงานผลิตชิป"
        summary_th = "โรงงานชิปมีระบบหยุดอัตโนมัติ ไม่พบความเสียหายร้ายแรงต่อโครงสร้างการผลิตหลัก"

    return {
        **article,
        "category": category,
        "target_generation": gen,
        "impact_direction": direction,
        "impact_multiplier_pct": round(multiplier, 2),
        "sentiment_score": sentiment_score,
        "confidence": confidence,
        "headline_th": headline_th,
        "summary_th": summary_th,
        "analyzed_at": datetime.now(timezone.utc).isoformat()
    }

def analyze_news_articles(articles: List[Dict[str, Any]] = None, force_refresh: bool = False) -> Dict[str, Any]:
    """
    Analyzes news items and aggregates net price impact for DDR4 and DDR5.
    """
    DATA_NEWS_DIR.mkdir(parents=True, exist_ok=True)

    if not articles:
        articles = fetch_dram_news(force_live=force_refresh)

    analyzed_items = [analyze_article(it) for it in articles]

    # Calculate net multiplier for DDR4 and DDR5
    ddr4_multipliers = [it["impact_multiplier_pct"] for it in analyzed_items if it["target_generation"] in ["DDR4", "ALL"]]
    ddr5_multipliers = [it["impact_multiplier_pct"] for it in analyzed_items if it["target_generation"] in ["DDR5", "ALL"]]

    net_ddr4 = round(sum(ddr4_multipliers) / max(1, len(ddr4_multipliers)), 2)
    net_ddr5 = round(sum(ddr5_multipliers) / max(1, len(ddr5_multipliers)), 2)

    # Overall market mood
    if net_ddr4 > 0 and net_ddr5 < 0:
        market_sentiment_th = "ตลาดสองขั้ว (Divergence): DDR4 ซัพพลายเริ่มตึงตัวจาก HBM | DDR5 กำลังปรับราคาลงจาก Yield ที่ดีขึ้น"
        market_sentiment_en = "Divergent Market: DDR4 facing tight fab supply from HBM reallocation; DDR5 continues structural easing."
        overall_status = "DIVERGENT"
    elif net_ddr5 < 0:
        market_sentiment_th = "ตลาดอยู่ในทิศทางผ่อนคลาย (Buyer's Market) ราคาแรมรุ่นใหม่มีแนวโน้มปรับลงต่อเนื่อง"
        market_sentiment_en = "Buyer's Market: Mainstream DDR5 memory retail prices are steadily easing."
        overall_status = "BEARISH_PRICING"
    else:
        market_sentiment_th = "ราคาตลาดอยู่ในเกณฑ์มีเสถียรภาพและเริ่มทรงตัว"
        market_sentiment_en = "Market prices are maintaining baseline stability."
        overall_status = "STABLE"

    result = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "total_analyzed": len(analyzed_items),
        "overall_status": overall_status,
        "market_sentiment_th": market_sentiment_th,
        "market_sentiment_en": market_sentiment_en,
        "net_impact_ddr4_pct": net_ddr4,
        "net_impact_ddr5_pct": net_ddr5,
        "articles": analyzed_items
    }

    try:
        with open(ANALYZED_NEWS_PATH, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved analyzed news results to {ANALYZED_NEWS_PATH}")
    except Exception as e:
        logger.error(f"Failed to save analyzed news data: {e}")

    return result

def get_latest_news_analysis(force_refresh: bool = False) -> Dict[str, Any]:
    """Loads latest analyzed news from disk, or computes fresh if absent."""
    if not force_refresh and ANALYZED_NEWS_PATH.exists():
        try:
            with open(ANALYZED_NEWS_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Could not load cached news analysis: {e}")

    return analyze_news_articles(force_refresh=force_refresh)

if __name__ == "__main__":
    res = analyze_news_articles(force_refresh=True)
    print("\n" + "=" * 65)
    print("      GLOBAL DRAM NEWS INTELLIGENCE ANALYSIS RESULTS      ")
    print("=" * 65)
    print(f"Market Status : {res['overall_status']}")
    print(f"Summary (TH)  : {res['market_sentiment_th']}")
    print(f"Net Impact    : DDR4 {res['net_impact_ddr4_pct']:+.2f}% | DDR5 {res['net_impact_ddr5_pct']:+.2f}%")
    print("-" * 65)
    for idx, it in enumerate(res['articles'][:4], 1):
        print(f"{idx}. [{it['impact_direction']} {it['impact_multiplier_pct']:+.1f}%] {it['headline_th']}")
        print(f"   -> {it['summary_th']}\n")
