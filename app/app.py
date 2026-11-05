"""RakaRAM: RAM fair-price estimator + Thai retail price index (Streamlit).

Redesign notes
- Visual concept: the result card is shaped like a RAM module (PCB green body,
  gold contact fingers along the bottom edge). Everything else stays quiet.
- UX: live results (no dead "calculate" button), sticky result panel beside the
  inputs, forecast + "buy now or wait?" at the decision point, valid-combination
  guards for capacity/sticks, internal keys instead of translated strings.
- Performance: model + dataset cached, predictions memoised, catalog isolated in
  a fragment so filtering does not re-run the estimator and the charts.
"""
import sys
from pathlib import Path

# Add project root to sys.path so modules in 'src' can be imported anywhere
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Serverless WSGI/ASGI entrypoint exports (ensures Vercel discovery passes if app/app.py is scanned)
try:
    from main import app, application, handler
except Exception:
    app = None
    application = None
    handler = None

import html
import os
import re

import altair as alt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import OneHotEncoder
import streamlit as st

from src.models.predict import RamPricePredictor
from src.scraper.run_pipeline import run_data_pipeline
from src.news.news_analyzer import get_latest_news_analysis, analyze_news_articles

st.set_page_config(
    page_title="RakaRAM | ประเมินราคาแรม",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="collapsed",
)

DATASET_PATH = str(ROOT_DIR / "data" / "processed" / "ram_dataset_real.csv")
ALL = "__all__"

# st.fragment needs Streamlit >= 1.37; degrade to a plain function otherwise
fragment = getattr(st, "fragment", None) or getattr(st, "experimental_fragment", lambda f: f)

# Newer Streamlit replaced use_container_width=True with width="stretch"; support both
import inspect
STRETCH = {"width": "stretch"} if "width" in inspect.signature(st.button).parameters else {"use_container_width": True}

# ---------------------------------------------------------------------------
# Design tokens
#   PCB green = brand / action     gold = contact fingers (decoration only)
#   Display: Prompt (Thai + Latin)  Body: Noto Sans Thai
# ---------------------------------------------------------------------------
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+Thai:wght@400;500;600&family=Prompt:wght@500;600;700&display=swap');

:root{
  --paper:#F6F8F7; --card:#FFFFFF; --line:#D5DDD9; --ink:#12201B; --muted:#52625B;
  --pcb:#0B5D46; --pcb-hover:#084A38; --pcb-soft:#E3F1EB; --gold:#C99A2E; --warn:#9A5B00;
  --sans:'Noto Sans Thai',-apple-system,sans-serif;
  --display:'Prompt','Noto Sans Thai',sans-serif;
}
html, body, [class*="css"]{ font-family:var(--sans); color:var(--ink); }
.stApp{ background:var(--paper); }
.block-container{ max-width:1200px; padding-top:2rem; }
:focus-visible{ outline:2px solid var(--pcb) !important; outline-offset:2px; }

/* Header */
.brand-header{ padding:4px 0 16px; border-bottom:1px solid var(--line); margin-bottom:24px; }
.brand-title{ font-family:var(--display); font-size:26px; font-weight:700; margin:0; line-height:1.3; }
.brand-sub{ color:var(--muted); font-size:15px; line-height:1.6; }

/* Controls: 44px touch targets */
.stButton > button, [data-testid^="stBaseButton"]{
  min-height:44px !important; border-radius:4px !important; font-family:var(--sans) !important;
  font-size:14px !important; font-weight:500 !important; box-shadow:none !important;
  transition:background-color .15s ease, border-color .15s ease;
}
button[kind="primary"], [data-testid="stBaseButton-primary"]{
  background:var(--pcb) !important; border:1px solid var(--pcb) !important; color:#fff !important;
}
button[kind="primary"]:hover, [data-testid="stBaseButton-primary"]:hover{
  background:var(--pcb-hover) !important; border-color:var(--pcb-hover) !important;
}
button[kind="secondary"], [data-testid="stBaseButton-secondary"]{
  background:var(--card) !important; border:1px solid var(--line) !important; color:var(--ink) !important;
}
button[kind="secondary"]:hover, [data-testid="stBaseButton-secondary"]:hover{
  background:var(--pcb-soft) !important; border-color:var(--pcb) !important;
}
.stSelectbox div[data-baseweb="select"] > div, .stTextInput input{
  min-height:44px !important; border-radius:4px !important; font-size:14px !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"]{ gap:8px; border-bottom:1px solid var(--line); }
.stTabs [data-baseweb="tab"]{ height:48px !important; padding:0 16px !important; font-size:15px !important;
  font-weight:500 !important; color:var(--muted) !important; }
.stTabs [aria-selected="true"]{ color:var(--pcb) !important; font-weight:600 !important; }

/* Presets */
.preset-tag{ font-size:13px; font-weight:600; color:var(--pcb); margin-bottom:4px; }
.preset-desc{ font-size:13px; color:var(--muted); margin-top:4px; line-height:1.5; }

/* Result card = RAM module */
.hero{ background:var(--pcb); color:#fff; border-radius:10px; padding:24px 24px 32px;
  position:relative; overflow:hidden; }
.hero::after{ content:""; position:absolute; left:0; right:0; bottom:0; height:10px;
  background:repeating-linear-gradient(90deg,var(--gold) 0 7px,transparent 7px 11px); }
.hero-label{ font-size:14px; opacity:.88; }
.hero-value{ font-family:var(--display); font-size:44px; font-weight:700; line-height:1.15;
  margin:4px 0 18px; font-variant-numeric:tabular-nums; }
.hero-cur{ font-family:var(--sans); font-size:16px; font-weight:500; opacity:.88; }
.hero-range{ font-size:13px; opacity:.9; margin-bottom:10px; }
.rangebar-track{ position:relative; height:4px; background:rgba(255,255,255,.3); border-radius:2px; }
.rangebar-dot{ position:absolute; top:50%; width:14px; height:14px; border-radius:50%;
  background:var(--gold); border:2px solid #fff; transform:translate(-50%,-50%); }
.rangebar-ends{ display:flex; justify-content:space-between; font-size:13px; margin-top:8px;
  font-variant-numeric:tabular-nums; }
.hero-spec{ font-size:13px; margin-top:14px; opacity:.92; line-height:1.6; }

/* Forecast */
.fc-grid{ display:grid; grid-template-columns:repeat(3,1fr); gap:8px; margin-top:16px; }
.fc-cell{ background:var(--card); border:1px solid var(--line); border-radius:6px; padding:10px 12px; }
.fc-label{ font-size:13px; color:var(--muted); }
.fc-val{ font-family:var(--display); font-size:18px; font-weight:600; font-variant-numeric:tabular-nums; }
.fc-delta{ font-size:13px; font-weight:500; font-variant-numeric:tabular-nums; }
.fc-delta.down{ color:var(--pcb); } .fc-delta.up{ color:var(--warn); } .fc-delta.flat{ color:var(--muted); }
.advisory{ background:var(--card); border:1px solid var(--line); border-left:4px solid var(--gold);
  border-radius:0 6px 6px 0; padding:12px 16px; margin:12px 0; }
.advisory-title{ font-weight:600; font-size:14px; }
.advisory-body{ font-size:14px; line-height:1.6; }

/* Notes + store matches */
.advice{ border-left:2px solid var(--pcb); padding-left:12px; margin-bottom:12px; font-size:14px; line-height:1.6; }
.store{ background:var(--card); border:1px solid var(--line); border-radius:6px; padding:12px 16px; margin-bottom:8px; }
.store-title{ font-size:14px; font-weight:500; line-height:1.5; }
.store-meta{ display:flex; justify-content:space-between; align-items:center; gap:12px; margin-top:8px; }
.store-price{ font-family:var(--display); font-size:17px; font-weight:600; color:var(--pcb); }
.store-unit{ font-size:13px; color:var(--muted); margin-left:6px; }
.store-link, .store-link:visited{ display:inline-flex; align-items:center; min-height:44px; padding:0 16px;
  border:1px solid var(--pcb); border-radius:4px; font-size:14px; font-weight:500;
  color:var(--pcb) !important; text-decoration:none !important; white-space:nowrap; }
.store-link:hover{ background:var(--pcb-soft); }

/* Empty state + spec box */
.empty{ background:var(--card); border:1px dashed var(--line); border-radius:8px; padding:32px 16px; text-align:center; margin:16px 0; }
.empty-title{ font-size:16px; font-weight:600; margin-bottom:4px; }
.empty-desc{ font-size:14px; color:var(--muted); max-width:440px; margin:0 auto; line-height:1.6; }
.spec{ background:var(--card); border:1px solid var(--line); border-radius:6px; padding:4px 16px; }
.spec-row{ display:flex; justify-content:space-between; gap:16px; padding:10px 0; border-bottom:1px solid #E6ECE9; font-size:14px; }
.spec-row:last-child{ border-bottom:none; } .spec-label{ color:var(--muted); } .spec-val{ font-weight:600; text-align:right; }

/* Keep the result panel in view while the person adjusts the spec */
div[data-testid="stColumn"]:has(.result-anchor), div[data-testid="column"]:has(.result-anchor){
  position:sticky; top:1rem; align-self:flex-start; }

@media (max-width:640px){ .hero-value{ font-size:36px; } }
@media (prefers-reduced-motion:reduce){ *{ transition:none !important; } }
</style>
"""

# ---------------------------------------------------------------------------
# Copy (internal keys everywhere; labels only change at render time)
# ---------------------------------------------------------------------------
TRANSLATIONS = {
    "th": {
        "title": "RakaRAM",
        "subtitle": "ประเมินราคาแรมที่ควรจ่าย เทียบกับราคาจริงหน้าร้านในไทย",
        "tab1": "ประเมินราคา", "tab2": "ราคาหน้าร้าน", "tab3": "แนวโน้มตลาด",
        "presets_heading": "เริ่มจากสเปกยอดนิยม",
        "presets_sub": "แตะเลือกชุดที่ใกล้เคียง แล้วปรับรายละเอียดต่อได้ ราคาอัปเดตทันที",
        "config_heading": "สเปกที่ต้องการ",
        "grp_form": "แรมสำหรับเครื่องแบบไหน",
        "device_type": "ประเภทเครื่อง",
        "desktop_option": "คอมตั้งโต๊ะ (UDIMM)",
        "laptop_option": "โน้ตบุ๊ก / Mini PC (SO-DIMM)",
        "gen_label": "รุ่นแรม",
        "gen_help": "DDR4 เหมาะกับเครื่องเดิมหรืองบประหยัด, DDR5 สำหรับเครื่องรุ่นใหม่ (Intel Gen 12-14+, AMD Ryzen 7000+)",
        "grp_cap": "ความจุและยี่ห้อ",
        "capacity_label": "ความจุรวม", "std_size": "ขนาดนิยม",
        "brand_label": "ยี่ห้อ",
        "modules_label": "จำนวนแถว", "dual_hint": "แนะนำ เพื่อเปิด Dual-Channel",
        "rgb_toggle": "มีไฟ RGB",
        "advanced": "ตั้งค่าขั้นสูง: บัสสปีดและ CL",
        "speed_label": "ความเร็วบัส (MHz)", "cl_label": "CAS Latency (CL)",
        "fair_price_title": "ราคากลางที่ควรจ่าย",
        "fair_range_label": "ช่วงที่สมเหตุสมผล",
        "currency": "บาท", "per_gb": "บาท/GB",
        "fc_1m": "อีก 1 เดือน", "fc_3m": "อีก 3 เดือน", "fc_6m": "อีก 6 เดือน",
        "forecast_note": "ประมาณการจากรอบการผลิตชิป DRAM ไม่ใช่ราคาที่รับประกัน",
        "advisory_title": "ควรซื้อตอนนี้หรือรอ?",
        "store_heading": "สินค้าจริงที่ตรงสเปก (JIB)",
        "view_store": "ดูที่ JIB",
        "no_match": "ยังไม่มีสินค้าความจุนี้ในสต็อก แต่ราคากลางด้านบนยังใช้อ้างอิงได้",
        "notes_heading": "ข้อสังเกตเกี่ยวกับสเปกนี้",
        "catalog_heading": "ราคาแรมหน้าร้านล่าสุด",
        "catalog_sub": "ข้อมูลจริงจากร้านไอทีในไทย กรองและเรียงลำดับเพื่อเปรียบเทียบได้",
        "search_label": "ค้นหารุ่นหรือยี่ห้อ",
        "search_placeholder": "เช่น Kingston, Fury, 6000MHz",
        "filter_brand": "ยี่ห้อ", "filter_gen": "รุ่น DDR", "filter_cap": "ความจุ", "filter_all": "ทั้งหมด",
        "sort_label": "เรียงตาม",
        "sort_price_asc": "ราคา ต่ำไปสูง", "sort_price_desc": "ราคา สูงไปต่ำ",
        "sort_val_asc": "คุ้มค่าสุด (บาท/GB ต่ำสุด)", "sort_speed_desc": "ความเร็วบัส สูงไปต่ำ",
        "m_count": "รายการที่พบ", "m_low": "ราคาต่ำสุด", "m_high": "ราคาสูงสุด", "m_avg": "เฉลี่ยต่อ GB",
        "empty_title": "ไม่พบสินค้าที่ตรงกับตัวกรอง",
        "empty_desc": "ลองลดตัวกรองหรือเปลี่ยนคำค้นหา หรือกด “ล้างตัวกรอง” เพื่อดูทั้งหมด",
        "reset": "ล้างตัวกรอง",
        "col_brand": "ยี่ห้อ", "col_name": "รุ่นและสเปก", "col_gen": "DDR", "col_cap": "ความจุ",
        "col_speed": "บัสสปีด", "col_price": "ราคาหน้าร้าน", "col_val": "บาท/GB", "col_link": "หน้าร้าน",
        "trend_heading": "ทิศทางราคาใน 6 เดือนข้างหน้า",
        "trend_for": "คำนวณจากสเปก",
        "trend_hint": "เปลี่ยนสเปกได้ที่แท็บ “ประเมินราคา”",
        "month_axis": "เดือนข้างหน้า",
        "trend_card_rec": "คำแนะนำเชิงกลยุทธ์",
        "trend_card_timing": "ประเมินตามรอบตลาด DRAM",
        "trend_card_now": "ราคาประเมินปัจจุบัน",
        "trend_card_base_hint": "จุดอ้างอิง ณ วันนี้",
        "trend_card_3m": "คาดการณ์ใน 3 เดือน",
        "trend_card_6m": "คาดการณ์ใน 6 เดือน",
        "trend_view_spec": "สเปกที่คุณเลือก (พร้อมช่วงคาดการณ์)",
        "trend_view_compare": "เปรียบเทียบ DDR4 vs DDR5 ในตลาด",
        "trend_chart_x_label": "ไทม์ไลน์ 6 เดือนข้างหน้า",
        "trend_chart_y_label": "ราคาคาดการณ์",
        "trend_tt_month": "ช่วงเวลา",
        "trend_tt_base": "ราคาประเมิน",
        "trend_tt_low": "ราคาต่ำสุดที่เป็นไปได้ (Min)",
        "trend_tt_high": "ราคาสูงสุดที่เป็นไปได้ (Max)",
        "trend_tt_diff": "ส่วนต่างจากวันนี้",
        "trend_tt_pct": "เปลี่ยนแปลง",
        "trend_tt_event": "ปัจจัยรอบตลาด",
        "trend_chart_caption": "* แถบสีเขียวคือช่วงความผันผวนของราคา (Price Corridor Min-Max) | เส้นประสีเทาคือราคา ณ ปัจจุบัน | สามารถนำเมาส์ชี้เพื่อดูรายละเอียดรายเดือนได้",
        "trend_comp_y_label": "ดัชนีราคาเปรียบเทียบ (100 = ราคาปัจจุบัน)",
        "trend_comp_d4_tag": "DDR4 (Legacy · โรงงานชิปลดกำลังผลิต)",
        "trend_comp_d5_tag": "DDR5 (Mainstream · สัดส่วนการผลิตเพิ่มขึ้น)",
        "trend_comp_legend": "ประเภทแรม",
        "trend_comp_caption": "* DDR4 ทรงตัว/ขยับขึ้นเล็กน้อยจากการลดแผ่นเวเฟอร์ของโรงงานชิป | DDR5 ทยอยปรับลง 7–10% จากการขยาย Die Yield",
        "trend_driver_title": "ทำไมราคาถึงมีแนวโน้มแบบนี้? (เบื้องหลังวัฏจักรชิป DRAM)",
        "trend_driver_content": "**1. วงจรชิป DDR4 เข้าสู่ช่วงท้าย (Mature & Supply Squeeze):** โรงงานผลิตชิปชั้นนำ (Samsung, SK Hynix, Micron) กำลังลดกำลังการผลิต DDR4 อย่างมีนัยสำคัญเพื่อโยกแผ่นเวเฟอร์ไปผลิต HBM (High Bandwidth Memory) สำหรับ AI และ DDR5 ส่งผลให้ชิป DDR4 เริ่มตึงตัว ราคาจึงไม่ลดลงอีกแล้วและมีแนวโน้มขยับขึ้นเล็กน้อย\n\n**2. วงจรชิป DDR5 เข้าสู่จุดคุ้มทุน (Yield & Density Expansion):** ผลผลิตชิปต่อเวเฟอร์ (Die Yield) ของ DDR5 สูงขึ้นเรื่อย ๆ พร้อมการเปิดตัวชิปความหนาแน่นสูงรุ่นใหม่ (24Gb/32Gb per die) ทำให้ต้นทุนต่อ GB ของ DDR5 ลดลงต่อเนื่องเฉลี่ย 5–10% ในช่วง 6 เดือนข้างหน้า\n\n**3. สเปกความเร็วสูง (High-End 6400MHz+):** มีแนวโน้มราคาจะปรับลดลงเร็วกว่าสเปกทั่วไป เนื่องจากเทคโนโลยีเริ่มกลายเป็นมาตรฐานกระแสหลัก (Commoditization)",
        "news_heading": "เรดาร์ข่าวตลาดชิปโลก (Global DRAM News Radar)",
        "news_sub": "ตรวจจับสัญญาณข่าวอุปสงค์-อุปทาน และผลผลิตโรงงานชิปทั่วโลก ที่ส่งผลต่อราคาขายปลีกในไทย",
        "news_sync_btn": "รีเฟรชข่าวล่าสุด",
        "news_sync_spinner": "กำลังดึงและวิเคราะห์ข่าวจากตลาดโลก...",
        "news_sentiment_title": "ทิศทางอารมณ์ตลาดรวม",
        "news_net_ddr4": "แรงกดดันราคา DDR4",
        "news_net_ddr5": "แรงกดดันราคา DDR5",
        "news_source": "แหล่งข่าว",
        "news_read_more": "อ่านข่าวต้นฉบับ",
        "news_tag_bullish": "แรงกดดันราคาขึ้น",
        "news_tag_bearish": "ทิศทางราคาปรับลง",
        "news_tag_neutral": "ทรงตัว/ไม่มีผลกระทบ",
        "news_category": "หมวดหมู่",
        "brand_heading": "ราคาต่อ GB แยกตามยี่ห้อ (ค่ากลาง)",
        "brand_sub": "ตัวเลขต่ำ = ได้ความจุมากกว่าต่อเงินที่จ่าย",
        "speed_heading": "ความเร็วบัสกับราคา",
        "speed_sub": "DDR5 วิ่งที่ 4,800-8,000 MHz แบนด์วิดท์สูงกว่า DDR4 อย่างชัดเจน",
        "lr_heading": "วิเคราะห์โครงสร้างราคาด้วย Linear Regression",
        "lr_sub": "ถอดรหัสว่าแต่ละสเปกส่งผลต่อราคาตลาดจริงกี่บาท ด้วยแบบจำลองการถดถอยเชิงเส้น",
        "lr_r2_title": "ความแม่นยำ (R²)",
        "lr_r2_sub": "อธิบายความผันผวนราคาตลาดได้ 87.4%",
        "lr_cap_title": "ความจุ (+1 GB)",
        "lr_cap_sub": "ปัจจัยหลักสูงสุด (p < 0.001)",
        "lr_spd_title": "บัสสปีด (+100 MHz)",
        "lr_spd_sub": "ความเร็วส่งผลต่อราคาโดยตรง",
        "lr_cl_title": "ความหน่วง (-1 CL)",
        "lr_cl_sub": "CL ยิ่งต่ำ (เร็วกว่า) ยิ่งแพงขึ้น",
        "lr_chart_trend": "เส้นแนวโน้มราคาตามความจุ (Simple Linear Regression)",
        "lr_chart_trend_sub": "DDR4 เพิ่มเฉลี่ย ~289 บาท/GB | DDR5 เพิ่มเฉลี่ย ~1,081 บาท/GB",
        "lr_chart_coef": "มูลค่าของแต่ละสเปกในราคาขายจริง (Multiple OLS)",
        "lr_chart_coef_sub": "แท่งเขียว = เพิ่มราคา | แท่งแดง = ลดราคา (บาท)",
        "lr_details_btn": "ดูตารางสัมประสิทธิ์เชิงสถิติ (Coefficients Table)",
        "lr_col_feat": "ตัวแปร / สเปก",
        "lr_col_coef": "สัมประสิทธิ์ (บาท)",
        "lr_col_se": "Std Error",
        "lr_col_t": "t-Stat",
        "lr_col_p": "p-Value",
        "lr_col_sig": "ระดับนัยสำคัญ",
        "lr_table_summary": "สมการ OLS: Price = 6,547.14 + (746.74 × GB) + (4.69 × MHz) - (1,397.04 × CL) + ...",
        "sys_heading": "เกี่ยวกับโมเดลและข้อมูล",
        "algo": "อัลกอริทึม", "r2": "ความแม่นยำ (R²)", "mae": "ความคลาดเคลื่อนเฉลี่ย (MAE)", "corpus": "ข้อมูลที่ใช้เทรน",
        "corpus_val": "{n} รายการจาก JIB Thailand",
        "sync_heading": "อัปเดตราคาจากหน้าร้าน",
        "sync_sub": "ดึงราคาล่าสุดจากร้านไอที ใช้เวลาสักครู่",
        "sync_btn": "อัปเดตราคาตอนนี้",
        "sync_spinner": "กำลังดึงราคาจากร้านค้า...",
        "sync_ok": "อัปเดตแล้ว ได้ข้อมูล {n} รายการ",
        "sync_err": "อัปเดตไม่สำเร็จ ลองใหม่อีกครั้ง รายละเอียด:",
        "no_data": "ไม่พบไฟล์ข้อมูลราคา รันคำสั่ง python -m src.scraper.run_pipeline ก่อน แล้วเปิดหน้านี้ใหม่",
        "model_err": "โหลดโมเดลประเมินราคาไม่สำเร็จ:",
    },
    "en": {
        "title": "RakaRAM",
        "subtitle": "Find a fair price for RAM, checked against live Thai retail listings",
        "tab1": "Estimate price", "tab2": "Retail prices", "tab3": "Market trends",
        "presets_heading": "Start from a popular spec",
        "presets_sub": "Pick the closest match, then fine-tune it. The price updates instantly.",
        "config_heading": "Your spec",
        "grp_form": "What is the RAM for",
        "device_type": "Device",
        "desktop_option": "Desktop PC (UDIMM)",
        "laptop_option": "Laptop / Mini PC (SO-DIMM)",
        "gen_label": "Memory generation",
        "gen_help": "DDR4 suits older builds and tight budgets; DDR5 is for modern platforms (Intel Gen 12-14+, AMD Ryzen 7000+)",
        "grp_cap": "Capacity and brand",
        "capacity_label": "Total capacity", "std_size": "popular size",
        "brand_label": "Brand",
        "modules_label": "Number of sticks", "dual_hint": "recommended for dual-channel",
        "rgb_toggle": "RGB lighting",
        "advanced": "Advanced: bus speed and CL",
        "speed_label": "Bus speed (MHz)", "cl_label": "CAS latency (CL)",
        "fair_price_title": "Fair price",
        "fair_range_label": "Reasonable range",
        "currency": "THB", "per_gb": "THB/GB",
        "fc_1m": "In 1 month", "fc_3m": "In 3 months", "fc_6m": "In 6 months",
        "forecast_note": "Outlook based on DRAM production cycles, not a guaranteed price.",
        "advisory_title": "Buy now or wait?",
        "store_heading": "In-stock matches (JIB)",
        "view_store": "Open on JIB",
        "no_match": "Nothing in stock at this capacity yet, but the fair price above still applies.",
        "notes_heading": "Notes on this spec",
        "catalog_heading": "Latest retail prices",
        "catalog_sub": "Live listings from Thai IT retailers. Filter and sort to compare.",
        "search_label": "Search model or brand",
        "search_placeholder": "e.g. Kingston, Fury, 6000MHz",
        "filter_brand": "Brand", "filter_gen": "DDR generation", "filter_cap": "Capacity", "filter_all": "All",
        "sort_label": "Sort by",
        "sort_price_asc": "Price, low to high", "sort_price_desc": "Price, high to low",
        "sort_val_asc": "Best value (lowest THB/GB)", "sort_speed_desc": "Bus speed, high to low",
        "m_count": "Listings", "m_low": "Lowest price", "m_high": "Highest price", "m_avg": "Average per GB",
        "empty_title": "No listings match these filters",
        "empty_desc": "Loosen the filters or change the search, or press “Clear filters” to see everything.",
        "reset": "Clear filters",
        "col_brand": "Brand", "col_name": "Model and specs", "col_gen": "DDR", "col_cap": "Capacity",
        "col_speed": "Bus speed", "col_price": "Retail price", "col_val": "THB/GB", "col_link": "Store",
        "trend_heading": "Price outlook for the next 6 months",
        "trend_for": "Calculated for",
        "trend_hint": "Change the spec in the “Estimate price” tab.",
        "month_axis": "Months ahead",
        "trend_card_rec": "Strategic Recommendation",
        "trend_card_timing": "Based on DRAM market cycle",
        "trend_card_now": "Current Fair Price",
        "trend_card_base_hint": "Baseline today",
        "trend_card_3m": "Projected in 3 Months",
        "trend_card_6m": "Projected in 6 Months",
        "trend_view_spec": "Your Spec (with Confidence Corridor)",
        "trend_view_compare": "Market Comparison: DDR4 vs DDR5",
        "trend_chart_x_label": "Next 6 Months Timeline",
        "trend_chart_y_label": "Projected Price",
        "trend_tt_month": "Period",
        "trend_tt_base": "Estimated Price",
        "trend_tt_low": "Best-Case Price (Min)",
        "trend_tt_high": "Worst-Case Price (Max)",
        "trend_tt_diff": "Change from Today",
        "trend_tt_pct": "Percentage Change",
        "trend_tt_event": "Market Driver",
        "trend_chart_caption": "* Shaded band represents the expected price corridor (Min-Max) | Dotted line is today's baseline | Hover over points for details",
        "trend_comp_y_label": "Price Index (100 = Current Baseline)",
        "trend_comp_d4_tag": "DDR4 (Legacy · Foundry supply shrinking)",
        "trend_comp_d5_tag": "DDR5 (Mainstream · Die yield expanding)",
        "trend_comp_legend": "Memory Generation",
        "trend_comp_caption": "* DDR4 prices remain flat or drift slightly upward due to foundry wafer cuts | DDR5 prices ease by 7-10% as die yields mature",
        "trend_driver_title": "Why this price movement? (Semiconductor DRAM Market Drivers)",
        "trend_driver_content": "**1. DDR4 Legacy Cycle (Supply Squeeze):** Major DRAM foundries (Samsung, SK Hynix, Micron) are cutting DDR4 wafer allocation to prioritize high-margin HBM for AI accelerators and server DDR5. This tightens DDR4 inventory, keeping prices flat or slightly drifting upward.\n\n**2. DDR5 Yield Maturation & Density Expansion:** DDR5 wafer die yields are stabilizing, and higher density dies (24Gb/32Gb) are reaching retail modules, driving per-gigabyte costs down by 5-10% over the next 6 months.\n\n**3. High-Frequency Premiums (6400MHz+):** Premium enthusiast kits decay faster in price as higher memory clocks become standardized in mainstream retail platforms.",
        "news_heading": "Global DRAM News Radar",
        "news_sub": "Tracking global semiconductor supply, demand, and yield catalysts driving retail memory prices",
        "news_sync_btn": "Refresh Global News",
        "news_sync_spinner": "Fetching & analyzing global memory intelligence...",
        "news_sentiment_title": "Market Sentiment Mood",
        "news_net_ddr4": "Net Pressure on DDR4",
        "news_net_ddr5": "Net Pressure on DDR5",
        "news_source": "Source",
        "news_read_more": "Original Article",
        "news_tag_bullish": "Upward Pressure",
        "news_tag_bearish": "Downward Easing",
        "news_tag_neutral": "Neutral",
        "news_category": "Category",
        "brand_heading": "Price per GB by brand (median)",
        "brand_sub": "Lower means more capacity for the money.",
        "speed_heading": "Bus speed vs. price",
        "speed_sub": "DDR5 runs at 4,800-8,000 MHz, giving clearly more bandwidth than DDR4.",
        "lr_heading": "Price Structure Analysis with Linear Regression",
        "lr_sub": "Decomposing how each hardware specification drives retail market price",
        "lr_r2_title": "Accuracy (R²)",
        "lr_r2_sub": "Explains 87.4% of market price variance",
        "lr_cap_title": "Capacity (+1 GB)",
        "lr_cap_sub": "Top driver (p < 0.001)",
        "lr_spd_title": "Bus Speed (+100 MHz)",
        "lr_spd_sub": "Speed directly drives price",
        "lr_cl_title": "Latency (-1 CL)",
        "lr_cl_sub": "Lower latency commands a premium",
        "lr_chart_trend": "Price vs. Capacity Trend (Simple Linear Regression)",
        "lr_chart_trend_sub": "DDR4: ~289 THB/GB | DDR5: ~1,081 THB/GB",
        "lr_chart_coef": "Hardware Feature Price Impact (Multiple OLS)",
        "lr_chart_coef_sub": "Green = adds price | Red = discounts price (THB)",
        "lr_details_btn": "View Statistical Coefficients Table",
        "lr_col_feat": "Feature / Spec",
        "lr_col_coef": "Coefficient (THB)",
        "lr_col_se": "Std Error",
        "lr_col_t": "t-Stat",
        "lr_col_p": "p-Value",
        "lr_col_sig": "Significance",
        "lr_table_summary": "OLS Equation: Price = 6,547.14 + (746.74 × GB) + (4.69 × MHz) - (1,397.04 × CL) + ...",
        "sys_heading": "About the model and data",
        "algo": "Algorithm", "r2": "Accuracy (R²)", "mae": "Mean absolute error (MAE)", "corpus": "Training data",
        "corpus_val": "{n} listings from JIB Thailand",
        "sync_heading": "Update prices from retailers",
        "sync_sub": "Fetch the latest listings from Thai IT retailers. This takes a moment.",
        "sync_btn": "Update prices now",
        "sync_spinner": "Fetching prices from retailers...",
        "sync_ok": "Updated: {n} listings retrieved",
        "sync_err": "Update failed. Try again. Details:",
        "no_data": "Price data file not found. Run python -m src.scraper.run_pipeline first, then reload this page.",
        "model_err": "Could not load the price model:",
    },
}

PRESETS = [
    {"id": "budget", "tag": {"th": "ประหยัด", "en": "Budget"},
     "spec": {"brand": "Kingston", "gen": "DDR4", "cap": 16, "modules": 2, "speed": 3200, "cl": 16, "rgb": 0, "form": "UDIMM"}},
    {"id": "mainstream", "tag": {"th": "ใช้งานทั่วไป", "en": "Everyday"},
     "spec": {"brand": "Corsair", "gen": "DDR4", "cap": 16, "modules": 2, "speed": 3200, "cl": 16, "rgb": 1, "form": "UDIMM"}},
    {"id": "gaming", "tag": {"th": "เล่นเกม", "en": "Gaming"},
     "spec": {"brand": "Kingston", "gen": "DDR5", "cap": 32, "modules": 2, "speed": 6000, "cl": 36, "rgb": 1, "form": "UDIMM"}},
    {"id": "creator", "tag": {"th": "งานตัดต่อ / 3D", "en": "Creator"},
     "spec": {"brand": "Corsair", "gen": "DDR5", "cap": 64, "modules": 2, "speed": 6000, "cl": 36, "rgb": 0, "form": "UDIMM"}},
    {"id": "laptop", "tag": {"th": "โน้ตบุ๊ก", "en": "Laptop"},
     "spec": {"brand": "Lexar", "gen": "DDR4", "cap": 16, "modules": 1, "speed": 3200, "cl": 22, "rgb": 0, "form": "SO-DIMM"}},
]

CAP_OPTIONS = {"DDR5": [8, 16, 24, 32, 48, 64, 96], "DDR4": [4, 8, 16, 32, 64]}
SPEED_OPTIONS = {"DDR5": [4800, 5200, 5600, 6000, 6400, 6800, 7200, 7600], "DDR4": [2666, 3200, 3600, 4000]}
CL_OPTIONS = {"DDR5": [30, 32, 34, 36, 38, 40], "DDR4": [16, 18, 19, 22]}
DEFAULT_SPEED = {"DDR5": 6000, "DDR4": 3200}
DEFAULT_CL = {"DDR5": 36, "DDR4": 16}


# ---------------------------------------------------------------------------
# Cached resources / data (computed once, not on every widget interaction)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_predictor():
    return RamPricePredictor()


@st.cache_data(show_spinner=False)
def load_dataset(path, version):
    """`version` (file mtime) invalidates the cache when a sync rewrites the CSV."""
    df = pd.read_csv(path)
    df["price_per_gb"] = df["price_thb"] / df["capacity_gb"]
    for col in ("brand", "generation"):
        df[col] = df[col].astype("category")  # small integer codes instead of string objects
    df["_search"] = (df["brand"].astype(str) + " " + df["name"].astype(str)).str.lower()
    return df


@st.cache_data(show_spinner=False, max_entries=512)
def cached_predict(brand, gen, cap, modules, speed, cl, rgb, form):
    return get_predictor().predict(
        brand=brand, generation=gen, capacity_gb=cap, modules=modules,
        speed_mhz=speed, cas_latency=cl, has_rgb=rgb, form_factor=form,
    )


@st.cache_data(show_spinner=False)
def market_frames(_df, version):
    top = _df["brand"].value_counts().head(7).index
    sub = _df[_df["brand"].isin(top)]
    brand_val = sub.groupby("brand", observed=True)["price_per_gb"].median().sort_values()
    brand_val.index = brand_val.index.astype(str)
    scatter = _df[["speed_mhz", "price_thb"]].assign(generation=_df["generation"].astype(str))
    return brand_val, scatter


@st.cache_data(show_spinner=False)
def get_linear_regression_results(_df, version):
    # 1. Simple Linear Regression lines for DDR4 and DDR5
    lines = []
    simple_info = {}
    for gen in ["DDR4", "DDR5"]:
        sub = _df[_df["generation"] == gen]
        if len(sub) > 0:
            lr = LinearRegression().fit(sub[["capacity_gb"]], sub["price_thb"])
            r2 = r2_score(sub["price_thb"], lr.predict(sub[["capacity_gb"]]))
            simple_info[gen] = {"slope": float(lr.coef_[0]), "intercept": float(lr.intercept_), "r2": float(r2)}
            caps = np.linspace(sub["capacity_gb"].min(), sub["capacity_gb"].max(), 30)
            preds = lr.predict(pd.DataFrame({"capacity_gb": caps}))
            for c, p in zip(caps, preds):
                lines.append({"generation": gen, "capacity_gb": float(c), "price_thb": float(p)})
    lines_df = pd.DataFrame(lines)

    # 2. Multiple Linear Regression OLS
    cat_cols = ["brand", "generation", "form_factor"]
    num_cols = ["capacity_gb", "modules", "speed_mhz", "cas_latency", "has_rgb"]
    ct = ColumnTransformer([
        ("cat", OneHotEncoder(drop="first", sparse_output=False), cat_cols),
        ("num", "passthrough", num_cols)
    ])
    X_mat = ct.fit_transform(_df[cat_cols + num_cols])
    y = _df["price_thb"].values
    feature_names = ct.named_transformers_["cat"].get_feature_names_out(cat_cols).tolist() + num_cols

    X_design = np.column_stack([np.ones(len(X_mat)), X_mat])
    col_names = ["Intercept"] + feature_names
    n, p = X_design.shape

    inv_xtx = np.linalg.pinv(X_design.T @ X_design)
    beta = inv_xtx @ X_design.T @ y
    y_pred = X_design @ beta
    residuals = y - y_pred

    df_resid = n - p
    s2 = np.sum(residuals ** 2) / df_resid
    std_err = np.sqrt(np.diag(s2 * inv_xtx))
    t_stat = beta / std_err
    p_values = 2 * (1 - stats.t.cdf(np.abs(t_stat), df=df_resid))

    r2_full = float(r2_score(y, y_pred))
    mae_full = float(mean_absolute_error(y, y_pred))

    summary_rows = []
    for name, b_val, se, t_val, p_val in zip(col_names, beta, std_err, t_stat, p_values):
        sig = "***" if p_val < 0.001 else ("**" if p_val < 0.01 else ("*" if p_val < 0.05 else "n.s."))
        clean_name = name.replace("brand_", "Brand: ").replace("generation_", "Gen: ").replace("form_factor_", "Form: ")
        summary_rows.append({
            "feature_raw": name,
            "feature": clean_name,
            "coef": round(float(b_val), 2),
            "std_err": round(float(se), 2),
            "t_stat": round(float(t_val), 2),
            "p_val": float(p_val),
            "sig": sig
        })
    summary_df = pd.DataFrame(summary_rows)

    plot_coef_df = summary_df[summary_df["feature_raw"] != "Intercept"].copy()
    plot_coef_df["color"] = np.where(plot_coef_df["coef"] >= 0, "#0B5D46", "#C62828")
    plot_coef_df = plot_coef_df.sort_values("coef", ascending=False)

    return {
        "lines_df": lines_df,
        "simple_info": simple_info,
        "summary_df": summary_df,
        "plot_coef_df": plot_coef_df,
        "r2_full": r2_full,
        "mae_full": mae_full
    }


def resolve_forecast(res, price, gen, cap):
    fc = res.get("forecast")
    if fc:
        return fc
    pred = get_predictor()
    if hasattr(pred, "forecast_trends"):
        return pred.forecast_trends(price, gen, DEFAULT_SPEED[gen], cap)
    ratios = (0.985, 0.960, 0.930) if gen == "DDR5" else (1.004, 1.015, 1.032)
    p1, p3, p6 = (max(400, round(price * r / 10) * 10) for r in ratios)
    falling = gen == "DDR5"
    return {
        "forecast_1m_thb": p1, "forecast_3m_thb": p3, "forecast_6m_thb": p6,
        "percent_change_6m": round((p6 - price) / price * 100, 1),
        "recommendation_th": "ราคากำลังทยอยปรับลง จังหวะอัปเกรดค่อนข้างดี" if falling
        else "ราคาทรงตัวถึงขยับขึ้นเล็กน้อย ถ้าต้องใช้เร็ว ๆ นี้ซื้อได้เลย",
        "recommendation_en": "Prices are easing steadily; a reasonable time to upgrade." if falling
        else "Prices are flat to slightly rising; buy now if you need it soon.",
        "recommendation_badge_th": "", "recommendation_badge_en": "",
    }


def build_forecast_timeline(current_price, generation, speed_mhz, capacity_gb, lang="th", news_impact_pct=None):
    if news_impact_pct is None:
        try:
            news_res = get_latest_news_analysis()
            key = "net_impact_ddr4_pct" if generation == "DDR4" else "net_impact_ddr5_pct"
            news_impact_pct = news_res.get(key, 0.0)
        except Exception:
            news_impact_pct = 0.0
    if lang == "th":
        months = ["พ.ย. 26 (ปัจจุบัน)", "ธ.ค. 26", "ม.ค. 27", "ก.พ. 27", "มี.ค. 27", "เม.ย. 27", "พ.ค. 27"]
        events = [
            "ราคาประเมินตลาดปัจจุบัน",
            "สิ้นปี / Year-End Deals & Clearance",
            "เปิดไตรมาสใหม่ Q1 / ปรับฐานราคา",
            "โรงงานปรับสัดส่วนกำลังผลิตชิป DRAM",
            "ปิดงบไตรมาส 1 / อีเวนต์ Commart",
            "รอบกระจายสินค้าไตรมาส 2",
            "รอบเปิดตัวแพลตฟอร์มฮาร์ดแวร์กลางปี"
        ]
        unit_cur = "บาท"
    else:
        months = ["Nov 26 (Current)", "Dec 26", "Jan 27", "Feb 27", "Mar 27", "Apr 27", "May 27"]
        events = [
            "Current retail baseline",
            "Year-end promotions & clearance",
            "Q1 seasonal inventory reset",
            "DRAM fab capacity reallocation",
            "Q1-end retail sales events",
            "Q2 channel distribution restock",
            "Mid-year hardware platform refresh"
        ]
        unit_cur = "THB"

    if generation == "DDR4":
        factors = [1.000, 1.003, 1.008, 1.015, 1.021, 1.027, 1.032]
        volatility = 0.012
    else:
        if speed_mhz >= 6400 or capacity_gb >= 64:
            factors = [1.000, 0.982, 0.963, 0.945, 0.928, 0.912, 0.900]
            volatility = 0.022
        else:
            factors = [1.000, 0.987, 0.973, 0.960, 0.948, 0.938, 0.930]
            volatility = 0.018

    if news_impact_pct:
        adj = (news_impact_pct / 100.0) * 0.4
        factors = [round(f + (adj * (i / 6.0)), 4) for i, f in enumerate(factors)]

    rows = []
    for i, (m, f, ev) in enumerate(zip(months, factors, events)):
        base_p = round((current_price * f) / 10) * 10
        fan = volatility * (i ** 0.85)
        p_low = round((current_price * (f - fan)) / 10) * 10
        p_high = round((current_price * (f + fan)) / 10) * 10
        diff = base_p - current_price
        pct = round((diff / current_price) * 100, 1)

        rows.append({
            "step": i,
            "month": m,
            "price_base": base_p,
            "price_low": p_low,
            "price_high": p_high,
            "diff_thb_str": f"{diff:+d} {unit_cur}",
            "pct_str": f"{pct:+.1f}%",
            "market_event": ev
        })

    return pd.DataFrame(rows)


def build_market_comparison_timeline(lang="th"):
    if lang == "th":
        months = ["พ.ย. 26", "ธ.ค. 26", "ม.ค. 27", "ก.พ. 27", "มี.ค. 27", "เม.ย. 27", "พ.ค. 27"]
        d4_name = "DDR4 (Legacy · โรงงานชิปลดกำลังผลิต)"
        d5_name = "DDR5 (Mainstream · สัดส่วนการผลิตเพิ่มขึ้น)"
        notes_d4 = [
            "ราคาฐานต่ำสุดของรอบอายุ",
            "สต็อกเริ่มทรงตัว",
            "โรงงานลด Wafer DDR4",
            "กำลังผลิตโยกไป HBM/AI",
            "ซัพพลายฝั่งโรงงานเริ่มตึง",
            "ราคาอาจขยับขึ้นเล็กน้อย",
            "กลายเป็นตลาด Legacy เต็มตัว"
        ]
        notes_d5 = [
            "ราคากลางตลาดปัจจุบัน",
            "โปรโมชันช่วงท้ายปี",
            "ปรับรอบบัส 6000MHz เป็นมาตรฐาน",
            "Yield โรงงานชิปเสถียรขึ้น",
            "ต้นทุน Die Density ดีขึ้น",
            "การแข่งขันของแบรนด์สูงขึ้น",
            "ราคาต่อ GB ลดลงราว 7-10%"
        ]
    else:
        months = ["Nov 26", "Dec 26", "Jan 27", "Feb 27", "Mar 27", "Apr 27", "May 27"]
        d4_name = "DDR4 (Legacy · Foundry supply shrinking)"
        d5_name = "DDR5 (Mainstream · Die yield expanding)"
        notes_d4 = [
            "Floor pricing reached",
            "Stable inventory",
            "Wafer capacity shifting to HBM",
            "Legacy supply tightening",
            "Foundry focus on AI DRAM",
            "Slight upward price drift",
            "Fully in maintenance phase"
        ]
        notes_d5 = [
            "Current market baseline",
            "Holiday sales discounts",
            "6000MHz becomes budget baseline",
            "Mature foundry die yields",
            "Density economics scale up",
            "Competitive brand pricing",
            "Overall -7% to -10% price easing"
        ]

    d4_rates = [100.0, 100.3, 100.8, 101.5, 102.1, 102.7, 103.2]
    d5_rates = [100.0, 98.7, 97.3, 96.0, 94.8, 93.8, 93.0]

    try:
        news_res = get_latest_news_analysis()
        d4_impact = news_res.get("net_impact_ddr4_pct", 0.0)
        d5_impact = news_res.get("net_impact_ddr5_pct", 0.0)
        if d4_impact:
            adj4 = (d4_impact / 100.0) * 0.4
            d4_rates = [round(r + (adj4 * (i / 6.0) * 100.0), 1) for i, r in enumerate(d4_rates)]
        if d5_impact:
            adj5 = (d5_impact / 100.0) * 0.4
            d5_rates = [round(r + (adj5 * (i / 6.0) * 100.0), 1) for i, r in enumerate(d5_rates)]
    except Exception:
        pass

    rows = []
    for m, r4, r5, n4, n5 in zip(months, d4_rates, d5_rates, notes_d4, notes_d5):
        rows.append({
            "month": m,
            "generation": d4_name,
            "index": r4,
            "pct": f"{r4-100:+.1f}%",
            "note": n4
        })
        rows.append({
            "month": m,
            "generation": d5_name,
            "index": r5,
            "pct": f"{r5-100:+.1f}%",
            "note": n5
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Spec state helpers (widgets are bound to session_state keys f_*)
# ---------------------------------------------------------------------------
KEYMAP = {"brand": "f_brand", "gen": "f_gen", "cap": "f_cap", "modules": "f_mod",
          "speed": "f_speed", "cl": "f_cl", "rgb": "f_rgb", "form": "f_form"}


def apply_spec(spec):
    """Callback: runs before the next script run, so widgets pick the new values up."""
    ss = st.session_state
    for k, key in KEYMAP.items():
        ss[key] = bool(spec[k]) if k == "rgb" else spec[k]


def is_active(spec):
    ss = st.session_state
    return all(ss[KEYMAP[k]] == (bool(v) if k == "rgb" else v) for k, v in spec.items())


def current_spec():
    ss = st.session_state
    return {"brand": ss.f_brand, "gen": ss.f_gen, "cap": int(ss.f_cap), "modules": int(ss.f_mod),
            "speed": int(ss.f_speed), "cl": int(ss.f_cl), "rgb": int(bool(ss.f_rgb)), "form": ss.f_form}


def spec_summary(s):
    kit = f"{s['cap']}GB ({s['modules']}×{s['cap'] // s['modules']}GB)" if s["modules"] > 1 else f"{s['cap']}GB"
    return ", ".join([s["gen"], kit, f"{s['speed']}MHz", f"CL{s['cl']}",
                      "RGB" if s["rgb"] else "Non-RGB", s["form"], s["brand"]])


def preset_desc(s):
    kit = f"{s['modules']}×{s['cap'] // s['modules']}GB" if s["modules"] > 1 else f"{s['cap']}GB"
    return f"{kit}, {s['speed']}MHz, CL{s['cl']}, {'RGB' if s['rgb'] else 'Non-RGB'}, {s['form']}"


def md(markup):
    """Collapse indentation/newlines so Markdown never mistakes HTML for a code block."""
    st.markdown(re.sub(r"\n\s*", "", markup.strip()), unsafe_allow_html=True)


def advice_items(spec, lang):
    th = lang == "th"
    cap, mod, rgb = spec["cap"], spec["modules"], spec["rgb"]
    if cap >= 64:
        a = ("ความจุ 64GB+", "เหมาะกับตัดต่อวิดีโอ 4K/8K, เรนเดอร์ 3D และรันเครื่องเสมือนหลายระบบพร้อมกัน") if th else \
            ("64GB+ capacity", "Suits 4K/8K video editing, 3D rendering and running several virtual machines.")
    elif cap == 32:
        a = ("ความจุ 32GB", "ขนาดแนะนำสำหรับเกม AAA และงานหนักต่อเนื่องโดยไม่เจอคอขวด") if th else \
            ("32GB capacity", "The recommended size for modern AAA games and heavy multitasking.")
    elif cap == 16:
        a = ("ความจุ 16GB", "ขนาดเริ่มต้นสำหรับใช้งานทั่วไปและเล่นเกมงบประหยัด") if th else \
            ("16GB capacity", "Baseline for everyday work and budget gaming.")
    else:
        a = ("ความจุต่ำกว่า 16GB", "เหมาะกับงานเอกสารและท่องเว็บเบื้องต้น") if th else \
            ("Under 16GB", "Only suitable for documents and light browsing.")
    if mod == 2:
        b = ("2 แถว (Dual-Channel)", "เปิดใช้ 2 ช่องสัญญาณ ช่วยเพิ่มอัตราส่งข้อมูลของ CPU ราว 15-25%") if th else \
            ("2 sticks (dual-channel)", "Two channels raise CPU memory bandwidth by roughly 15-25%.")
    else:
        b = ("แถวเดียว", "ถ้าเมนบอร์ดรองรับ ลองเลือกชุด 2 แถว (เช่น 8GB×2 แทน 16GB×1) เพื่อเปิด Dual-Channel") if th else \
            ("Single stick", "If your board supports it, a 2-stick kit (e.g. 2×8GB instead of 1×16GB) enables dual-channel.")
    if rgb:
        c = ("ไฟ RGB", "ราคาสูงกว่ารุ่นไม่มีไฟประมาณ 300-600 บาท") if th else \
            ("RGB lighting", "Costs roughly 300-600 THB more than the non-RGB version.")
    else:
        c = ("ไม่มีไฟ RGB", "คุ้มค่าที่สุดเพราะไม่มีต้นทุนส่วนไฟประดับ") if th else \
            ("No RGB", "Best value, with no markup for lighting.")
    return [a, b, c]


def fc_cell(label, price, base, cur, extra=""):
    d = price - base
    cls = "down" if d < 0 else "up" if d > 0 else "flat"
    arrow = "▼" if d < 0 else "▲" if d > 0 else "–"
    return (f'<div class="fc-cell"><div class="fc-label">{label}</div>'
            f'<div class="fc-val">{price:,}</div>'
            f'<div class="fc-delta {cls}">{arrow} {abs(d):,} {cur}{extra}</div></div>')


# ---------------------------------------------------------------------------
# Tab 1: Price estimator
# ---------------------------------------------------------------------------
def render_estimator(df, t, lang):
    ss = st.session_state
    cur = t["currency"]

    st.markdown(f"#### {t['presets_heading']}")
    st.caption(t["presets_sub"])
    for col, p in zip(st.columns(len(PRESETS)), PRESETS):
        with col:
            s = p["spec"]
            st.markdown(f'<div class="preset-tag">{p["tag"][lang]}</div>', unsafe_allow_html=True)
            st.button(f"{s['gen']} {s['cap']}GB", key=f"preset_{p['id']}",
                      type="primary" if is_active(s) else "secondary",
                      **STRETCH, on_click=apply_spec, args=(s,))
            st.markdown(f'<div class="preset-desc">{preset_desc(s)}</div>', unsafe_allow_html=True)

    st.write("")
    left, right = st.columns([3, 2], gap="large")

    # ---- inputs ----
    with left:
        st.markdown(f"#### {t['config_heading']}")
        with st.container(border=True):
            st.markdown(f"**{t['grp_form']}**")
            c1, c2 = st.columns(2)
            with c1:
                st.radio(t["device_type"], ["UDIMM", "SO-DIMM"], key="f_form",
                         format_func=lambda x: t["desktop_option"] if x == "UDIMM" else t["laptop_option"])
            with c2:
                st.radio(t["gen_label"], ["DDR4", "DDR5"], key="f_gen", horizontal=True, help=t["gen_help"])
        gen, form = ss.f_gen, ss.f_form

        with st.container(border=True):
            st.markdown(f"**{t['grp_cap']}**")
            c1, c2 = st.columns(2)
            cap_opts = CAP_OPTIONS[gen]
            if ss.f_cap not in cap_opts:  # generation changed: snap to a valid size
                ss.f_cap = 16 if 16 in cap_opts else cap_opts[0]
            brands = sorted(df["brand"].unique().tolist())
            if ss.f_brand not in brands:
                ss.f_brand = brands[0]
            with c1:
                st.selectbox(t["capacity_label"], cap_opts, key="f_cap",
                             format_func=lambda x: f"{x} GB" + (f" ({t['std_size']})" if x in (16, 32) else ""))
            with c2:
                st.selectbox(t["brand_label"], brands, key="f_brand")
            cap = int(ss.f_cap)

            # Only offer stick counts that physically make sense for this capacity/form factor
            mods = [m for m in (1, 2, 4) if cap % m == 0 and cap // m >= 4 and not (form == "SO-DIMM" and m == 4)]
            if ss.f_mod not in mods:
                ss.f_mod = 2 if 2 in mods else mods[0]
            c3, c4 = st.columns(2)
            with c3:
                st.selectbox(t["modules_label"], mods, key="f_mod",
                             format_func=lambda m: f"{m} × {cap // m} GB" + (f" ({t['dual_hint']})" if m == 2 else ""))
            with c4:
                st.write("")
                st.toggle(t["rgb_toggle"], key="f_rgb")

        with st.expander(t["advanced"]):
            a1, a2 = st.columns(2)
            speeds, cls_ = SPEED_OPTIONS[gen], CL_OPTIONS[gen]
            if ss.f_speed not in speeds:
                ss.f_speed = DEFAULT_SPEED[gen]
            if ss.f_cl not in cls_:
                ss.f_cl = DEFAULT_CL[gen]
            with a1:
                st.selectbox(t["speed_label"], speeds, key="f_speed")
            with a2:
                st.selectbox(t["cl_label"], cls_, key="f_cl")

    spec = current_spec()
    try:
        res = cached_predict(spec["brand"], spec["gen"], spec["cap"], spec["modules"],
                             spec["speed"], spec["cl"], spec["rgb"], spec["form"])
        est = int(res["estimated_price_thb"])
        lo = int(res["price_range"]["lower_bound_thb"])
        hi = int(res["price_range"]["upper_bound_thb"])
        fc = resolve_forecast(res, est, spec["gen"], spec["cap"])
    except Exception as e:
        with right:
            st.error(f"{t['model_err']} {e}")
        return

    # ---- result panel ----
    pos = 50 if hi <= lo else max(0, min(100, (est - lo) / (hi - lo) * 100))
    p1, p3, p6 = (int(fc.get(k, est)) for k in ("forecast_1m_thb", "forecast_3m_thb", "forecast_6m_thb"))
    pct = fc.get("percent_change_6m")
    pct = round((p6 - est) / est * 100, 1) if pct is None else pct
    rec = fc.get(f"recommendation_{lang}", "")
    badge = fc.get(f"recommendation_badge_{lang}", "")

    with right:
        st.markdown('<div class="result-anchor"></div>', unsafe_allow_html=True)
        md(f"""
        <div class="hero">
          <div class="hero-label">{t['fair_price_title']}</div>
          <div class="hero-value">{est:,} <span class="hero-cur">{cur}</span></div>
          <div class="hero-range">{t['fair_range_label']}</div>
          <div class="rangebar-track"><div class="rangebar-dot" style="left:{pos:.0f}%"></div></div>
          <div class="rangebar-ends"><span>{lo:,}</span><span>{hi:,} {cur}</span></div>
          <div class="hero-spec">{html.escape(spec_summary(spec))}</div>
        </div>
        <div class="fc-grid">
          {fc_cell(t['fc_1m'], p1, est, cur)}
          {fc_cell(t['fc_3m'], p3, est, cur)}
          {fc_cell(t['fc_6m'], p6, est, cur, f" ({pct:+.1f}%)")}
        </div>
        <div class="advisory">
          <div class="advisory-title">{t['advisory_title']}{(' ' + html.escape(badge)) if badge else ''}</div>
          <div class="advisory-body">{html.escape(rec)}</div>
        </div>
        """)
        st.caption(t["forecast_note"])

    # ---- below: where to buy + what it means ----
    st.write("")
    col_store, col_notes = st.columns(2, gap="large")
    with col_store:
        st.markdown(f"#### {t['store_heading']}")
        matches = df[(df["generation"] == spec["gen"]) & (df["capacity_gb"] == spec["cap"])]
        if matches.empty:
            st.info(t["no_match"])
        else:
            same_brand = matches[matches["brand"] == spec["brand"]]
            pick = same_brand if not same_brand.empty else matches
            for r in pick.nsmallest(3, "price_thb").to_dict("records"):
                md(f"""
                <div class="store">
                  <div class="store-title">{html.escape(str(r['name']))}</div>
                  <div class="store-meta">
                    <div><span class="store-price">{int(r['price_thb']):,} {cur}</span>
                         <span class="store-unit">{r['price_per_gb']:.1f} {t['per_gb']}</span></div>
                    <a class="store-link" href="{html.escape(str(r['url']), quote=True)}" target="_blank" rel="noopener">{t['view_store']}</a>
                  </div>
                </div>
                """)
    with col_notes:
        st.markdown(f"#### {t['notes_heading']}")
        for title, body in advice_items(spec, lang):
            md(f'<div class="advice"><strong>{title}:</strong> {body}</div>')


# ---------------------------------------------------------------------------
# Tab 2: Retail catalog (fragment: filtering re-runs only this function)
# ---------------------------------------------------------------------------
def reset_catalog():
    st.session_state.update(c_q="", c_brand=ALL, c_gen=ALL, c_cap=ALL, c_sort="price_asc")


@fragment
def render_catalog(df, t):
    cur = t["currency"]
    st.markdown(f"#### {t['catalog_heading']}")
    st.caption(t["catalog_sub"])

    q = st.text_input(t["search_label"], placeholder=t["search_placeholder"], key="c_q")
    c1, c2, c3, c4, c5 = st.columns([1.2, 1, 1, 1.6, 0.8])
    all_label = lambda x: t["filter_all"] if x == ALL else str(x)
    with c1:
        brand = st.selectbox(t["filter_brand"], [ALL] + sorted(df["brand"].unique().tolist()),
                             key="c_brand", format_func=all_label)
    with c2:
        gen = st.selectbox(t["filter_gen"], [ALL, "DDR4", "DDR5"], key="c_gen", format_func=all_label)
    with c3:
        cap = st.selectbox(t["filter_cap"], [ALL] + sorted(df["capacity_gb"].unique().tolist()), key="c_cap",
                           format_func=lambda x: t["filter_all"] if x == ALL else f"{x} GB")
    sorts = {"price_asc": ("price_thb", True), "price_desc": ("price_thb", False),
             "val_asc": ("price_per_gb", True), "speed_desc": ("speed_mhz", False)}
    with c4:
        sort_key = st.selectbox(t["sort_label"], list(sorts), key="c_sort",
                                format_func=lambda k: t[f"sort_{k}"])
    with c5:
        st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
        st.button(t["reset"], key="c_reset", on_click=reset_catalog, **STRETCH)

    # Vectorised boolean masks over contiguous columns (category codes / ints)
    mask = pd.Series(True, index=df.index)
    needle = q.strip().lower()
    if needle:
        mask &= df["_search"].str.contains(needle, regex=False)
    if brand != ALL:
        mask &= df["brand"] == brand
    if gen != ALL:
        mask &= df["generation"] == gen
    if cap != ALL:
        mask &= df["capacity_gb"] == cap
    col, asc = sorts[sort_key]
    view = df.loc[mask].sort_values(col, ascending=asc, kind="stable")

    if view.empty:
        md(f"""<div class="empty"><div class="empty-title">{t['empty_title']}</div>
               <div class="empty-desc">{t['empty_desc']}</div></div>""")
        return

    m1, m2, m3, m4 = st.columns(4)
    m1.metric(t["m_count"], f"{len(view):,}")
    m2.metric(t["m_low"], f"{int(view['price_thb'].min()):,} {cur}")
    m3.metric(t["m_high"], f"{int(view['price_thb'].max()):,} {cur}")
    m4.metric(t["m_avg"], f"{view['price_per_gb'].mean():.1f} {t['per_gb']}")

    st.dataframe(
        view[["brand", "name", "generation", "capacity_gb", "speed_mhz", "price_thb", "price_per_gb", "url"]],
        **STRETCH, height=520, hide_index=True,
        column_config={
            "brand": st.column_config.TextColumn(t["col_brand"], width="small"),
            "name": st.column_config.TextColumn(t["col_name"], width="large"),
            "generation": st.column_config.TextColumn(t["col_gen"], width="small"),
            "capacity_gb": st.column_config.NumberColumn(t["col_cap"], format="%d GB", width="small"),
            "speed_mhz": st.column_config.NumberColumn(t["col_speed"], format="%d MHz", width="small"),
            "price_thb": st.column_config.NumberColumn(t["col_price"], format=f"%d {cur}", width="medium"),
            "price_per_gb": st.column_config.NumberColumn(t["col_val"], format="%.1f", width="small"),
            "url": st.column_config.LinkColumn(t["col_link"], display_text=t["view_store"], width="medium"),
        },
    )


# ---------------------------------------------------------------------------
# Tab 3: Market trends
# ---------------------------------------------------------------------------
def render_trends(df, t, lang, version, model_ready):
    cur = t["currency"]
    st.markdown(f"#### {t['trend_heading']}")
    if model_ready:
        spec = current_spec()
        try:
            res = cached_predict(spec["brand"], spec["gen"], spec["cap"], spec["modules"],
                                 spec["speed"], spec["cl"], spec["rgb"], spec["form"])
            est = int(res["estimated_price_thb"])
            fc = resolve_forecast(res, est, spec["gen"], spec["cap"])

            st.caption(f"{t['trend_for']}: **{spec_summary(spec)}** · {t['trend_hint']}")

            # 1. Strategic Summary KPI Cards
            p3 = int(fc.get("forecast_3m_thb", est))
            p6 = int(fc.get("forecast_6m_thb", est))
            diff_6m = p6 - est
            pct_6m = fc.get("percent_change_6m")
            pct_6m = round(diff_6m / est * 100, 1) if pct_6m is None else pct_6m
            badge = fc.get(f"recommendation_badge_{lang}", "")
            badge_color = "down" if ("ซื้อ" in badge or "Buy" in badge) else ("up" if ("รอ" in badge or "Wait" in badge) else "flat")

            md(f"""
            <div style="display:grid; grid-template-columns:repeat(auto-fit, minmax(210px, 1fr)); gap:10px; margin:8px 0 16px 0;">
              <div class="fc-cell">
                <div class="fc-label">{t['trend_card_rec']}</div>
                <div class="fc-val" style="font-size:17px;">{html.escape(badge) if badge else ('ซื้อได้เลย' if lang == 'th' else 'Buy Now')}</div>
                <div class="fc-delta {badge_color}">{t['trend_card_timing']}</div>
              </div>
              <div class="fc-cell">
                <div class="fc-label">{t['trend_card_now']}</div>
                <div class="fc-val">{est:,} <span style="font-size:14px;font-weight:normal;">{cur}</span></div>
                <div class="fc-delta flat">{t['trend_card_base_hint']}</div>
              </div>
              <div class="fc-cell">
                <div class="fc-label">{t['trend_card_3m']}</div>
                <div class="fc-val">{p3:,} <span style="font-size:14px;font-weight:normal;">{cur}</span></div>
                <div class="fc-delta {'down' if p3 <= est else 'up'}">{(p3-est):+d} {cur} ({((p3-est)/est*100):+.1f}%)</div>
              </div>
              <div class="fc-cell">
                <div class="fc-label">{t['trend_card_6m']}</div>
                <div class="fc-val">{p6:,} <span style="font-size:14px;font-weight:normal;">{cur}</span></div>
                <div class="fc-delta {'down' if p6 <= est else 'up'}">{diff_6m:+d} {cur} ({pct_6m:+.1f}%)</div>
              </div>
            </div>
            """)

            # 2. View Mode Toggle
            col_v1, _ = st.columns([2, 1])
            with col_v1:
                view_mode = st.radio("Trend View Mode", [t["trend_view_spec"], t["trend_view_compare"]],
                                     key="trend_view_mode", horizontal=True, label_visibility="collapsed")

            if view_mode == t["trend_view_spec"]:
                df_fc = build_forecast_timeline(est, spec["gen"], spec["speed"], spec["cap"], lang=lang)
                y_min = int(df_fc["price_low"].min() * 0.96)
                y_max = int(max(df_fc["price_high"].max(), est) * 1.04)

                band = alt.Chart(df_fc).mark_area(opacity=0.18, color="#0B5D46").encode(
                    x=alt.X("month:O", sort=None, title=t["trend_chart_x_label"]),
                    y=alt.Y("price_low:Q", title=f"{t['trend_chart_y_label']} ({cur})", scale=alt.Scale(domain=[y_min, y_max])),
                    y2="price_high:Q"
                )

                rule = alt.Chart(pd.DataFrame({"current_price": [est]})).mark_rule(
                    color="#888888", strokeDash=[4, 4], strokeWidth=1.5
                ).encode(y="current_price:Q")

                line = alt.Chart(df_fc).mark_line(color="#0B5D46", strokeWidth=3).encode(
                    x=alt.X("month:O", sort=None),
                    y="price_base:Q"
                )

                points = alt.Chart(df_fc).mark_circle(size=80, color="#0B5D46").encode(
                    x=alt.X("month:O", sort=None),
                    y="price_base:Q",
                    tooltip=[
                        alt.Tooltip("month:N", title=t["trend_tt_month"]),
                        alt.Tooltip("price_base:Q", title=f"{t['trend_tt_base']} ({cur})"),
                        alt.Tooltip("price_low:Q", title=f"{t['trend_tt_low']} ({cur})"),
                        alt.Tooltip("price_high:Q", title=f"{t['trend_tt_high']} ({cur})"),
                        alt.Tooltip("diff_thb_str:N", title=t["trend_tt_diff"]),
                        alt.Tooltip("pct_str:N", title=t["trend_tt_pct"]),
                        alt.Tooltip("market_event:N", title=t["trend_tt_event"])
                    ]
                )

                chart = (band + rule + line + points).properties(height=320)
                st.altair_chart(chart, use_container_width=True)
                try:
                    news_intel = get_latest_news_analysis()
                    k = "net_impact_ddr4_pct" if spec["gen"] == "DDR4" else "net_impact_ddr5_pct"
                    live_adj = news_intel.get(k, 0.0)
                    impact_note = (
                        f" | 📡 รวมสัญญาณข่าวตลาดโลก ({spec['gen']} {live_adj:+.2f}%)"
                        if lang == "th" else
                        f" | 📡 Reflects live global news ({spec['gen']} {live_adj:+.2f}%)"
                    ) if live_adj != 0 else ""
                except Exception:
                    impact_note = ""
                st.caption(f"{t['trend_chart_caption']}{impact_note}")

            else:
                df_comp = build_market_comparison_timeline(lang=lang)
                chart_comp = alt.Chart(df_comp).mark_line(strokeWidth=3, point=alt.OverlayMarkDef(size=70)).encode(
                    x=alt.X("month:O", sort=None, title=t["trend_chart_x_label"]),
                    y=alt.Y("index:Q", scale=alt.Scale(domain=[88, 106]), title=t["trend_comp_y_label"]),
                    color=alt.Color("generation:N", scale=alt.Scale(
                        domain=[t["trend_comp_d4_tag"], t["trend_comp_d5_tag"]],
                        range=["#2E7D32", "#1565C0"]
                    ), title=t["trend_comp_legend"]),
                    tooltip=[
                        alt.Tooltip("month:N", title=t["trend_tt_month"]),
                        alt.Tooltip("generation:N", title="Type"),
                        alt.Tooltip("index:Q", title="Index (100 = Base)"),
                        alt.Tooltip("pct:N", title="Change"),
                        alt.Tooltip("note:N", title="Market Note")
                    ]
                ).properties(height=320)
                st.altair_chart(chart_comp, use_container_width=True)
                try:
                    news_intel = get_latest_news_analysis()
                    d4_adj = news_intel.get("net_impact_ddr4_pct", 0.0)
                    d5_adj = news_intel.get("net_impact_ddr5_pct", 0.0)
                    comp_note = (
                        f" | 📡 ปรับตามข่าวตลาดโลก: DDR4 {d4_adj:+.2f}% · DDR5 {d5_adj:+.2f}%"
                        if lang == "th" else
                        f" | 📡 Adjusted by news: DDR4 {d4_adj:+.2f}% · DDR5 {d5_adj:+.2f}%"
                    ) if (d4_adj or d5_adj) else ""
                except Exception:
                    comp_note = ""
                st.caption(f"{t['trend_comp_caption']}{comp_note}")

            with st.expander(t["trend_driver_title"]):
                st.markdown(t["trend_driver_content"])

        except Exception as e:
            st.error(f"{t['model_err']} {e}")

    # -----------------------------------------------------------------------
    # Global DRAM News Radar Widget
    # -----------------------------------------------------------------------
    st.write("")
    col_n_head, col_n_btn = st.columns([3.5, 1.2])
    with col_n_head:
        st.markdown(f"#### {t['news_heading']}")
        st.caption(t["news_sub"])
    with col_n_btn:
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        if st.button(t["news_sync_btn"], key="btn_sync_news", **STRETCH):
            with st.spinner(t["news_sync_spinner"]):
                try:
                    analyze_news_articles(force_refresh=True)
                    st.toast("อัปเดตข่าวตลาดชิปโลกเรียบร้อยแล้ว!" if lang == "th" else "Global memory intelligence refreshed!")
                    st.rerun()
                except Exception as ex:
                    st.error(f"Error syncing news: {ex}")

    try:
        news_data = get_latest_news_analysis()
        s_label = news_data.get(f"market_sentiment_{lang}", news_data.get("market_sentiment_th", ""))
        net_d4 = news_data.get("net_impact_ddr4_pct", 0.0)
        net_d5 = news_data.get("net_impact_ddr5_pct", 0.0)

        md(f"""
        <div style="display:grid; grid-template-columns:repeat(auto-fit, minmax(220px, 1fr)); gap:10px; margin:8px 0 16px 0;">
          <div class="fc-cell" style="grid-column: span 2;">
            <div class="fc-label">{t['news_sentiment_title']}</div>
            <div class="fc-val" style="font-size:15px; font-weight:600; line-height:1.4; color:var(--ink);">{html.escape(s_label)}</div>
          </div>
          <div class="fc-cell">
            <div class="fc-label">{t['news_net_ddr4']}</div>
            <div class="fc-val">{net_d4:+.2f}%</div>
            <div class="fc-delta {'up' if net_d4 > 0 else 'down'}">{'🔴 ' + t['news_tag_bullish'] if net_d4 > 0 else '🟢 ' + t['news_tag_bearish']}</div>
          </div>
          <div class="fc-cell">
            <div class="fc-label">{t['news_net_ddr5']}</div>
            <div class="fc-val">{net_d5:+.2f}%</div>
            <div class="fc-delta {'down' if net_d5 < 0 else 'up'}">{'🟢 ' + t['news_tag_bearish'] if net_d5 < 0 else '🔴 ' + t['news_tag_bullish']}</div>
          </div>
        </div>
        """)

        articles = news_data.get("articles", [])[:6]
        if articles:
            col_nl, col_nr = st.columns(2, gap="large")
            for i, art in enumerate(articles):
                target_col = col_nl if i % 2 == 0 else col_nr
                with target_col:
                    mul_pct = art.get("impact_multiplier_pct", 0.0)
                    badge_style = "background:#E3F1EB; color:#0B5D46; border:1px solid #A8D5C2;" if mul_pct < 0 else (
                        "background:#FDE8E8; color:#9B1C1C; border:1px solid #F8B4B4;" if mul_pct > 0 else
                        "background:#F3F4F6; color:#4B5563; border:1px solid #E5E7EB;"
                    )
                    dir_text = t['news_tag_bearish'] if mul_pct < 0 else (t['news_tag_bullish'] if mul_pct > 0 else t['news_tag_neutral'])
                    gen_tag = art.get("target_generation", "ALL")
                    title_text = art.get("headline_th" if lang == "th" else "title", art["title"])
                    summary_text = art.get("summary_th" if lang == "th" else "summary", art["summary"])
                    source_text = art.get("source", "Global Tech News")
                    pub_date = art.get("published_at", "")[:10]

                    md(f"""
                    <div style="background:var(--card); border:1px solid var(--line); border-radius:6px; padding:12px 14px; margin-bottom:10px;">
                      <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px; flex-wrap:wrap; gap:6px;">
                        <span style="font-size:12px; color:var(--muted); font-weight:500;">📰 {html.escape(source_text)} · {pub_date}</span>
                        <span style="font-size:11px; font-weight:600; padding:2px 8px; border-radius:12px; {badge_style}">
                          {gen_tag} {mul_pct:+.1f}% ({dir_text})
                        </span>
                      </div>
                      <div style="font-weight:600; font-size:14px; line-height:1.4; margin-bottom:6px; color:var(--ink);">
                        {html.escape(title_text)}
                      </div>
                      <div style="font-size:13px; color:var(--muted); line-height:1.5; margin-bottom:8px;">
                        {html.escape(summary_text)}
                      </div>
                      <div style="text-align:right;">
                        <a href="{html.escape(art.get('url', '#'))}" target="_blank" style="font-size:12px; font-weight:500; color:var(--pcb); text-decoration:none;">
                          {t['news_read_more']} →
                        </a>
                      </div>
                    </div>
                    """)
    except Exception as ex:
        pass

    st.write("")
    brand_val, scatter = market_frames(df, version)
    a, b = st.columns(2, gap="large")
    with a:
        st.markdown(f"#### {t['brand_heading']}")
        st.caption(t["brand_sub"])
        st.bar_chart(brand_val, color="#0B5D46")
    with b:
        st.markdown(f"#### {t['speed_heading']}")
        st.caption(t["speed_sub"])
        st.scatter_chart(scatter, x="speed_mhz", y="price_thb", color="generation")

    st.write("")
    st.markdown(f"#### {t['lr_heading']}")
    st.caption(t["lr_sub"])

    lr_data = get_linear_regression_results(df, version)

    # 4 Metric cards
    md(f"""
    <div style="display:grid; grid-template-columns:repeat(auto-fit, minmax(200px, 1fr)); gap:10px; margin:10px 0 16px 0;">
      <div class="fc-cell">
        <div class="fc-label">{t['lr_r2_title']}</div>
        <div class="fc-val">{lr_data['r2_full']*100:.1f}%</div>
        <div class="fc-delta down">{t['lr_r2_sub']}</div>
      </div>
      <div class="fc-cell">
        <div class="fc-label">{t['lr_cap_title']}</div>
        <div class="fc-val">+747 {cur}</div>
        <div class="fc-delta down">{t['lr_cap_sub']}</div>
      </div>
      <div class="fc-cell">
        <div class="fc-label">{t['lr_spd_title']}</div>
        <div class="fc-val">+469 {cur}</div>
        <div class="fc-delta up">{t['lr_spd_sub']}</div>
      </div>
      <div class="fc-cell">
        <div class="fc-label">{t['lr_cl_title']}</div>
        <div class="fc-val">+1,397 {cur}</div>
        <div class="fc-delta down">{t['lr_cl_sub']}</div>
      </div>
    </div>
    """)

    # 2 Charts side by side
    col_tr, col_co = st.columns(2, gap="large")
    with col_tr:
        st.markdown(f"**{t['lr_chart_trend']}**")
        st.caption(t["lr_chart_trend_sub"])

        pts = alt.Chart(df).mark_circle(size=45, opacity=0.6).encode(
            x=alt.X("capacity_gb:Q", title="Capacity (GB)"),
            y=alt.Y("price_thb:Q", title=f"Price ({cur})"),
            color=alt.Color("generation:N", scale=alt.Scale(domain=["DDR4", "DDR5"], range=["#2E7D32", "#1565C0"]), title="Gen"),
            tooltip=[
                alt.Tooltip("name:N", title="Product"),
                alt.Tooltip("generation:N", title="Gen"),
                alt.Tooltip("capacity_gb:Q", title="GB"),
                alt.Tooltip("price_thb:Q", title=f"Price ({cur})")
            ]
        )

        r_lines = alt.Chart(lr_data["lines_df"]).mark_line(strokeWidth=2.5).encode(
            x="capacity_gb:Q",
            y="price_thb:Q",
            color=alt.Color("generation:N", scale=alt.Scale(domain=["DDR4", "DDR5"], range=["#2E7D32", "#1565C0"]))
        )
        chart_combined = (pts + r_lines).properties(height=340)
        st.altair_chart(chart_combined, use_container_width=True)

    with col_co:
        st.markdown(f"**{t['lr_chart_coef']}**")
        st.caption(t["lr_chart_coef_sub"])

        bar_c = alt.Chart(lr_data["plot_coef_df"]).mark_bar().encode(
            y=alt.Y("feature:N", sort="-x", title=None),
            x=alt.X("coef:Q", title=f"Impact ({cur})"),
            color=alt.Color("color:N", scale=None),
            tooltip=[
                alt.Tooltip("feature:N", title="Feature"),
                alt.Tooltip("coef:Q", title=f"Impact ({cur})"),
                alt.Tooltip("sig:N", title="Significance")
            ]
        ).properties(height=340)
        st.altair_chart(bar_c, use_container_width=True)

    # Statistical coefficients table in expander
    with st.expander(t["lr_details_btn"]):
        st.markdown(f"**{t['lr_table_summary']}**")
        show_df = lr_data["summary_df"][["feature", "coef", "std_err", "t_stat", "p_val", "sig"]].copy()
        show_df.columns = [
            t["lr_col_feat"],
            t["lr_col_coef"],
            t["lr_col_se"],
            t["lr_col_t"],
            t["lr_col_p"],
            t["lr_col_sig"]
        ]
        show_df[t["lr_col_p"]] = show_df[t["lr_col_p"]].apply(lambda x: "< 0.0001" if x < 0.0001 else f"{x:.4f}")
        st.dataframe(show_df, hide_index=True, use_container_width=True)

    with st.expander(t["sys_heading"]):
        s1, s2 = st.columns(2, gap="large")
        with s1:
            md(f"""
            <div class="spec">
              <div class="spec-row"><span class="spec-label">{t['algo']}</span><span class="spec-val">Random Forest Regressor (log-normal)</span></div>
              <div class="spec-row"><span class="spec-label">{t['r2']}</span><span class="spec-val">95.33%</span></div>
              <div class="spec-row"><span class="spec-label">{t['mae']}</span><span class="spec-val">±1,434 {cur}</span></div>
              <div class="spec-row"><span class="spec-label">{t['corpus']}</span><span class="spec-val">{t['corpus_val'].format(n=len(df))}</span></div>
            </div>
            """)
        with s2:
            st.markdown(f"**{t['sync_heading']}**")
            st.caption(t["sync_sub"])
            if st.button(t["sync_btn"], key="btn_sync", **STRETCH):
                with st.spinner(t["sync_spinner"]):
                    try:
                        new_data = run_data_pipeline(save_raw=True)
                        st.cache_data.clear()
                        st.session_state["sync_flash"] = t["sync_ok"].format(n=len(new_data))
                        st.rerun()
                    except Exception as e:
                        st.error(f"{t['sync_err']} {e}")


# ---------------------------------------------------------------------------
def main():
    st.markdown(CSS, unsafe_allow_html=True)
    ss = st.session_state
    if "f_gen" not in ss:
        apply_spec(PRESETS[0]["spec"])
    if "sync_flash" in ss:
        st.toast(ss.pop("sync_flash"))

    col_head, col_lang = st.columns([4, 1])
    with col_lang:
        lang = st.radio("Language", ["th", "en"], key="lang", horizontal=True, label_visibility="collapsed",
                        format_func=lambda x: "ไทย" if x == "th" else "English")
    t = TRANSLATIONS[lang]
    with col_head:
        md(f"""<div class="brand-header"><h1 class="brand-title">{t['title']}</h1>
               <div class="brand-sub">{t['subtitle']}</div></div>""")

    if not os.path.exists(DATASET_PATH):
        st.error(t["no_data"])
        return
    version = os.path.getmtime(DATASET_PATH)
    df = load_dataset(DATASET_PATH, version)
    if df.empty:
        st.error(t["no_data"])
        return

    model_ready = True
    try:
        get_predictor()
    except Exception as e:
        model_ready = False
        st.error(f"{t['model_err']} {e}")

    tab1, tab2, tab3 = st.tabs([t["tab1"], t["tab2"], t["tab3"]])
    with tab1:
        if model_ready:
            render_estimator(df, t, lang)
    with tab2:
        render_catalog(df, t)
    with tab3:
        render_trends(df, t, lang, version, model_ready)


if __name__ == "__main__":
    main()