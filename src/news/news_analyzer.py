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
    "HBM_EXPANSION": [r"\bhbm\b", r"\bhbm3e\b", r"\bhbm4\b", r"\bai server\b", r"\bnvidia\b", r"\baccelerator\b", r"\bai data center\b", r"\bai bottleneck\b"],
    "SUPPLY_CUT": [r"\breallocat\w*\b", r"\bcut\w*\b", r"\bshrink\w*\b", r"\btight\w*\b", r"\bhalt\w*\b", r"\bshut\w*\b", r"\bcapacity shift\w*\b", r"\bfully booked\b", r"\breduc\w* output\b"],
    "CHINA_SUPPLY": [r"\bcxmt\b", r"\bchangxin\b", r"\bchinese memory\b", r"\boversupply\b", r"\bchinese factor\w*\b"],
    "YIELD_IMPROVEMENT": [r"\byield\w*\b", r"\b1-beta\b", r"\bdie density\b", r"\b24gb\b", r"\b32gb\b", r"\bmanufacturing cost\w*\b"],
    "PRICE_EASING": [r"\beas\w*\b", r"\bdeclin\w*\b", r"\bdrop\w*\b", r"\bcompress\w*\b", r"\bslump\w*\b", r"\bcheap\w*\b", r"\bdiscount\w*\b", r"\bslip\w*\b"],
    "PRICE_HIKE": [r"\bhike\w*\b", r"\brebound\w*\b", r"\brise\w*\b", r"\brising\b", r"\bsurg\w*\b", r"\bfloor level\b", r"\bstabiliz\w* at floor\b", r"\bincreas\w*\b"],
    "DISASTER": [r"\bquake\b", r"\bearthquake\b", r"\bfire\b", r"\bflood\b", r"\bdisruption\w*\b"],
    "DEMAND": [r"\bryzen\b", r"\bcore ultra\b", r"\badoption\b", r"\bconsumer transition\b", r"\beconomies of scale\b"]
}

def detect_generation(text_or_title: str, summary: str = "") -> str:
    if not summary:
        upper = text_or_title.upper()
        has_ddr4 = "DDR4" in upper
        has_ddr5 = "DDR5" in upper
        if has_ddr4 and not has_ddr5:
            return "DDR4"
        if has_ddr5 and not has_ddr4:
            return "DDR5"
        return "ALL"

    title_upper = text_or_title.upper()
    has_d4_title = "DDR4" in title_upper
    has_d5_title = "DDR5" in title_upper
    if has_d4_title and not has_d5_title:
        return "DDR4"
    if has_d5_title and not has_d4_title:
        return "DDR5"

    full_upper = f"{text_or_title} {summary}".upper()
    has_ddr4 = "DDR4" in full_upper
    has_ddr5 = "DDR5" in full_upper
    if has_ddr4 and not has_ddr5:
        return "DDR4"
    if has_ddr5 and not has_ddr4:
        return "DDR5"
    return "ALL"

def analyze_article(article: Dict[str, Any]) -> Dict[str, Any]:
    full_text = f"{article['title']} {article.get('summary', '')}".lower()
    gen = detect_generation(article["title"], article.get("summary", ""))

    # Count pattern occurrences
    matches = {}
    for cat, regexes in PATTERNS.items():
        count = sum(len(re.findall(rx, full_text)) for rx in regexes)
        if count > 0:
            matches[cat] = count

    # Determine impact direction & multipliers
    direction = "NEUTRAL"
    multiplier = 0.0
    d4_impact = 0.0
    d5_impact = 0.0
    sentiment_score = 0
    confidence = 0.85
    category = "GENERAL_NEWS"
    headline_th = article["title"]
    summary_th = "ข่าวสารทั่วไปเกี่ยวกับอุตสาหกรรมหน่วยความจำ"

    # Rule 1: Disaster / Quake
    if "DISASTER" in matches:
        direction = "BULLISH" if "damage" not in full_text else "NEUTRAL"
        d4_impact = 0.5
        d5_impact = 0.5
        multiplier = 0.5
        sentiment_score = +25
        category = "SUPPLY_DISRUPTION"
        confidence = 0.80
        headline_th = "รายงานผลกระทบจากเหตุแผ่นดินไหวต่อโรงงานผลิตชิป"
        summary_th = "โรงงานชิปมีระบบหยุดอัตโนมัติ ไม่พบความเสียหายร้ายแรงต่อโครงสร้างการผลิตหลัก"

    # Rule 2: HBM shift or wafer reallocation -> Bullish for DDR4/general DRAM (supply contraction)
    elif "HBM_EXPANSION" in matches and ("SUPPLY_CUT" in matches or "wafer" in full_text):
        direction = "BULLISH"
        d4_impact = 2.5
        d5_impact = 0.8 if gen in ["DDR5", "ALL"] else 0.0
        multiplier = d4_impact if gen == "DDR4" else (d5_impact if gen == "DDR5" else 1.8)
        sentiment_score = +65
        category = "HBM_WAFER_SHIFT"
        confidence = 0.92
        headline_th = "โรงงานชิปโยกสายการผลิตแรมทั่วไปไปทำ HBM สำหรับ AI"
        summary_th = "Samsung และ SK Hynix เร่งผลิตชิป HBM ให้เซิร์ฟเวอร์ AI ส่งผลให้โควตาเวเฟอร์ DDR4 เริ่มตึงตัว ราคาอาจขยับขึ้นเล็กน้อย"

    # Rule 3: Chinese Manufacturer Supply & Competition (CXMT expansion)
    elif "CHINA_SUPPLY" in matches:
        direction = "BEARISH"
        d4_impact = -1.5
        d5_impact = -2.5
        multiplier = -2.5 if gen in ["DDR5", "ALL"] else -1.5
        sentiment_score = -55
        category = "COMPETITION_SUPPLY"
        confidence = 0.89
        headline_th = "ซัพพลายจากผู้ผลิตจีน (CXMT) เข้าสู่ตลาด เพิ่มการแข่งขันด้านราคา"
        summary_th = "การเร่งขยายกำลังการผลิตของผู้ผลิตจีนช่วยเพิ่มปริมาณชิปในตลาดและกดดันราคาขายปลีก"

    # Rule 4: Yield improvement or 1-beta / density -> Bearish for DDR5 prices (cheaper per GB)
    elif "YIELD_IMPROVEMENT" in matches:
        direction = "BEARISH"
        d4_impact = 0.0
        d5_impact = -3.5
        multiplier = -3.5 if gen in ["DDR5", "ALL"] else -0.5
        sentiment_score = -70
        category = "YIELD_MATURATION"
        confidence = 0.90
        headline_th = "ความก้าวหน้าการผลิตชิป DDR5 ส่งผลให้ต้นทุนต่อ GB ลดลง"
        summary_th = "Micron และผู้ผลิตชิปทำ Yield 24Gb ได้สูงขึ้น หนุนราคาขายปลีก DDR5 ให้ปรับลดลงอย่างต่อเนื่อง"

    # Rule 5: Dual / Divergent contract price signals (DDR4 floor + DDR5 easing)
    elif ("PRICE_HIKE" in matches or "floor" in full_text) and "PRICE_EASING" in matches:
        direction = "DIVERGENT"
        d4_impact = 1.2
        d5_impact = -2.8
        multiplier = d4_impact if gen == "DDR4" else (d5_impact if gen == "DDR5" else -1.0)
        sentiment_score = -20
        category = "PRICE_COMPRESSION"
        confidence = 0.91
        headline_th = "ราคาแรม DDR4 ทรงตัวที่ฐานราคา ขณะที่ DDR5 ปรับลดลงต่อเนื่อง"
        summary_th = "DRAMeXchange รายงานราคาซื้อขายล่วงหน้า: DDR4 แตะระดับราคาต่ำสุดแล้ว ส่วน DDR5 ทยอยลดลง 5-8%"

    # Rule 6: Hardware Platform Demand & Adoption
    elif "DEMAND" in matches and "adoption" in full_text:
        direction = "BEARISH"  # Higher adoption -> economies of scale -> cheaper
        d4_impact = 0.0
        d5_impact = -2.0
        multiplier = -2.0
        sentiment_score = -40
        category = "MAINSTREAM_ADOPTION"
        confidence = 0.85
        headline_th = "ผู้ใช้งานเปลี่ยนมาใช้ DDR5 เพิ่มขึ้น หนุนให้ราคาต่อชิ้นประหยัดขึ้น"
        summary_th = "สัดส่วนการประกอบเครื่องด้วย DDR5 เกิน 75% ก่อให้เกิดการประหยัดจากขนาด (Economies of scale)"

    # Rule 7: Pure Supply Cut
    elif "SUPPLY_CUT" in matches:
        direction = "BULLISH"
        d4_impact = 2.0
        d5_impact = 0.8
        multiplier = 2.0 if gen == "DDR4" else 1.0
        sentiment_score = +50
        category = "SUPPLY_CONTRACTION"
        confidence = 0.88
        headline_th = "ซัพพลายแรมในตลาดโลกลดลงจากนโยบายควบคุมกำลังผลิต"
        summary_th = "ผู้ผลิตควบคุมกำลังการผลิตแรมรุ่นเก่า ช่วยพยุงราคาไม่ให้ตกและผลักดันราคาขายส่ง"

    # Rule 8: Direct Price Hike / Rebound / Surge
    elif "PRICE_HIKE" in matches:
        direction = "BULLISH"
        d4_impact = 2.0
        d5_impact = 1.0
        multiplier = 2.0 if gen == "DDR4" else 1.0
        sentiment_score = +55
        category = "PRICE_SURGE"
        confidence = 0.88
        headline_th = "ราคาชิปหน่วยความจำในตลาดสัญญาซื้อขายเริ่มปรับตัวขึ้น"
        summary_th = "สัญญาสั่งซื้อล่วงหน้าและราคาตลาดสปอตมีแนวโน้มขยับขึ้นตามความต้องการชิปเซิร์ฟเวอร์"

    # Rule 9: Price Easing / Contract Prices
    elif "PRICE_EASING" in matches:
        direction = "BEARISH"
        d4_impact = -0.5
        d5_impact = -2.8
        multiplier = -2.8 if gen == "DDR5" else -0.5
        sentiment_score = -55
        category = "PRICE_COMPRESSION"
        confidence = 0.87
        headline_th = "ราคาแรม DDR5 หน้าร้านทยอยปรับตัวลงตามราคาซื้อขายล่วงหน้า"
        summary_th = "รายงาน TrendForce ยืนยันราคาแรม DDR5 มีเสถียรภาพและทยอยปรับตัวลดลงตามกลไกตลาด"

    return {
        **article,
        "category": category,
        "target_generation": gen,
        "impact_direction": direction,
        "impact_multiplier_pct": round(multiplier, 2),
        "impact_ddr4": round(d4_impact, 2),
        "impact_ddr5": round(d5_impact, 2),
        "sentiment_score": sentiment_score,
        "confidence": confidence,
        "headline_th": headline_th,
        "summary_th": summary_th,
        "analyzed_at": datetime.now(timezone.utc).isoformat()
    }

def calculate_net_multiplier(items: List[Dict[str, Any]], target_gen: str) -> float:
    """
    Computes conviction-weighted net price multiplier for a target generation
    from active market catalysts, avoiding dilution by neutral stock/general articles.
    """
    key = "impact_ddr4" if target_gen == "DDR4" else "impact_ddr5"
    active = [
        it[key] for it in items
        if it.get(key, 0.0) != 0.0
    ]
    if not active:
        active = [
            it["impact_multiplier_pct"] for it in items
            if it.get("target_generation") in [target_gen, "ALL"] and it.get("impact_multiplier_pct", 0.0) != 0.0
        ]
    if not active:
        return 0.0

    avg_active = sum(active) / len(active)
    # Conviction factor: 1 catalyst = 75% weight, 2+ catalysts = 100% full weight
    conviction = min(1.0, 0.6 + 0.2 * (len(active) - 1)) if len(active) >= 1 else 0.5
    return round(max(-5.0, min(5.0, avg_active * conviction)), 2)

def analyze_news_articles(articles: List[Dict[str, Any]] = None, force_refresh: bool = False) -> Dict[str, Any]:
    """
    Analyzes news items and aggregates net price impact for DDR4 and DDR5.
    """
    try:
        DATA_NEWS_DIR.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass

    if not articles:
        articles = fetch_dram_news(force_live=force_refresh)

    analyzed_items = [analyze_article(it) for it in articles]

    # Calculate net multiplier for DDR4 and DDR5 using conviction-weighted active signals
    net_ddr4 = calculate_net_multiplier(analyzed_items, "DDR4")
    net_ddr5 = calculate_net_multiplier(analyzed_items, "DDR5")

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
