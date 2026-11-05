import sys
from pathlib import Path

# Add project root directory to sys.path so modules in 'src' can be imported anywhere
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import streamlit as st
import pandas as pd
import numpy as np
import os
import json
from src.models.predict import RamPricePredictor
from src.scraper.run_pipeline import run_data_pipeline

# Page Configuration
st.set_page_config(
    page_title="RakaRAM | ดัชนีราคาแรมและระบบประเมินราคากลาง",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Design Tokens: "Akihabara Index" (Japanese Contemporary Hardware Index)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+Thai:wght@300;400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');
    
    :root {
        --color-canvas:       #FFFFFF;
        --color-surface:      #F1F5F9;
        --color-border:       #CBD5E1;
        --color-text-main:    #0F172A;
        --color-text-muted:   #475569;
        --color-accent:       #0284C7;
        --color-accent-hover: #0369A1;
        --color-accent-soft:  #E0F2FE;
        --color-success:      #16A34A;
        --color-warning:      #D97706;
        --color-disabled:     #94A3B8;

        --font-sans:          'Noto Sans Thai', -apple-system, sans-serif;
        --font-mono:          'Space Grotesk', monospace;

        --text-xs:            12px;
        --text-sm:            14px;
        --text-base:          16px;
        --text-lg:            18px;
        --text-xl:            22px;
        --text-hero:          36px;

        --space-1:            4px;
        --space-2:            8px;
        --space-3:            12px;
        --space-4:            16px;
        --space-6:            24px;
        --space-8:            32px;

        --radius:             2px;
        --target-min-size:    44px;
    }

    html, body, [class*="css"] {
        font-family: var(--font-sans);
        color: var(--color-text-main);
        background-color: var(--color-canvas);
    }

    /* Streamlit base override */
    .stApp {
        background-color: var(--color-canvas);
    }

    /* Global Header */
    .brand-header {
        border-bottom: 1px solid var(--color-border);
        padding: var(--space-4) 0;
        margin-bottom: var(--space-6);
    }
    .brand-title {
        font-size: var(--text-xl);
        font-weight: 700;
        letter-spacing: -0.01em;
        color: var(--color-text-main);
        margin: 0;
        line-height: 1.2;
    }
    .brand-subtitle {
        font-size: var(--text-sm);
        color: var(--color-text-muted);
        margin-top: var(--space-1);
    }

    /* Primary Buttons & Interactive Controls (44px min height) */
    .stButton > button {
        min-height: var(--target-min-size) !important;
        border-radius: var(--radius) !important;
        font-family: var(--font-sans) !important;
        font-size: var(--text-sm) !important;
        font-weight: 500 !important;
        border: 1px solid var(--color-border) !important;
        box-shadow: none !important;
        transition: background-color 0.15s ease, border-color 0.15s ease !important;
    }
    .stButton > button[kind="primary"] {
        background-color: var(--color-accent) !important;
        border-color: var(--color-accent) !important;
        color: #FFFFFF !important;
    }
    .stButton > button[kind="primary"]:hover {
        background-color: var(--color-accent-hover) !important;
        border-color: var(--color-accent-hover) !important;
    }
    .stButton > button[kind="secondary"] {
        background-color: var(--color-canvas) !important;
        color: var(--color-text-main) !important;
    }
    .stButton > button[kind="secondary"]:hover {
        background-color: var(--color-surface) !important;
        border-color: var(--color-text-muted) !important;
    }

    /* Input Fields & Selectboxes */
    .stSelectbox div[data-baseweb="select"] > div,
    .stTextInput > div > div > input {
        min-height: var(--target-min-size) !important;
        border-radius: var(--radius) !important;
        border-color: var(--color-border) !important;
        font-family: var(--font-sans) !important;
        font-size: var(--text-sm) !important;
        box-shadow: none !important;
    }
    .stSelectbox div[data-baseweb="select"]:focus-within {
        border-color: var(--color-accent) !important;
    }

    /* Hero Price Result (Single Dominant Element) */
    .hero-price-panel {
        background-color: var(--color-surface);
        border: 1px solid var(--color-border);
        border-radius: var(--radius);
        padding: var(--space-6);
        text-align: center;
        margin: var(--space-4) 0;
    }
    .hero-price-label {
        font-size: var(--text-xs);
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: var(--color-text-muted);
    }
    .hero-price-value {
        font-family: var(--font-mono);
        font-size: var(--text-hero);
        font-weight: 700;
        color: var(--color-text-main);
        margin: var(--space-2) 0;
        line-height: 1.1;
    }
    .hero-price-range {
        display: inline-block;
        background-color: var(--color-accent-soft);
        color: var(--color-accent-hover);
        border: 1px solid #BAE6FD;
        border-radius: var(--radius);
        padding: var(--space-1) var(--space-3);
        font-family: var(--font-mono);
        font-size: var(--text-xs);
        font-weight: 600;
    }

    /* Preset Selection Table/Row */
    .preset-row {
        background-color: var(--color-canvas);
        border: 1px solid var(--color-border);
        border-radius: var(--radius);
        padding: var(--space-3) var(--space-4);
        margin-bottom: var(--space-2);
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        min-height: 120px;
    }
    .preset-badge {
        display: inline-block;
        font-size: var(--text-xs);
        font-family: var(--font-mono);
        font-weight: 600;
        padding: 2px 6px;
        border-radius: var(--radius);
        border: 1px solid var(--color-border);
        background-color: var(--color-surface);
        color: var(--color-text-muted);
        width: fit-content;
        margin-bottom: var(--space-1);
    }
    .preset-title {
        font-size: var(--text-sm);
        font-weight: 600;
        color: var(--color-text-main);
        margin: 0;
    }
    .preset-desc {
        font-size: var(--text-xs);
        font-family: var(--font-mono);
        color: var(--color-text-muted);
        margin-top: var(--space-1);
    }

    /* Spec Advice Block */
    .advice-item {
        border-left: 2px solid var(--color-accent);
        padding-left: var(--space-3);
        margin-bottom: var(--space-3);
        font-size: var(--text-sm);
        line-height: 1.5;
        color: var(--color-text-main);
    }

    /* Real Product Store Row */
    .store-product-row {
        background-color: var(--color-canvas);
        border: 1px solid var(--color-border);
        border-radius: var(--radius);
        padding: var(--space-3) var(--space-4);
        margin-bottom: var(--space-2);
    }
    .store-product-title {
        font-size: var(--text-sm);
        font-weight: 500;
        color: var(--color-text-main);
        line-height: 1.4;
    }
    .store-product-meta {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-top: var(--space-2);
    }
    .store-product-price {
        font-family: var(--font-mono);
        font-size: var(--text-base);
        font-weight: 700;
        color: var(--color-accent);
    }
    .store-product-unit {
        font-size: var(--text-xs);
        font-family: var(--font-mono);
        color: var(--color-text-muted);
        margin-left: var(--space-1);
    }
    .store-link-btn {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        min-height: var(--target-min-size);
        padding: 0 var(--space-4);
        background-color: var(--color-surface);
        border: 1px solid var(--color-border);
        color: var(--color-text-main);
        font-size: var(--text-xs);
        font-weight: 500;
        text-decoration: none;
        border-radius: var(--radius);
        transition: background-color 0.15s ease, border-color 0.15s ease;
        box-sizing: border-box;
    }
    .store-link-btn:hover {
        background-color: var(--color-accent-soft);
        border-color: var(--color-accent);
        color: var(--color-accent-hover);
        text-decoration: none;
    }

    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: var(--space-2);
        border-bottom: 1px solid var(--color-border);
    }
    .stTabs [data-baseweb="tab"] {
        height: var(--target-min-size) !important;
        font-family: var(--font-sans) !important;
        font-size: var(--text-sm) !important;
        font-weight: 500 !important;
        border-radius: var(--radius) var(--radius) 0 0 !important;
        padding: 0 var(--space-4) !important;
        color: var(--color-text-muted) !important;
    }
    .stTabs [aria-selected="true"] {
        color: var(--color-accent) !important;
        border-bottom: 2px solid var(--color-accent) !important;
        font-weight: 600 !important;
    }

    /* Table & Catalog Index (Akihabara Index) */
    .index-status-bar {
        background-color: var(--color-surface);
        border: 1px solid var(--color-border);
        border-radius: var(--radius);
        padding: var(--space-2) var(--space-4);
        margin-bottom: var(--space-4);
        display: flex;
        flex-wrap: wrap;
        gap: var(--space-4);
        align-items: center;
        font-family: var(--font-mono);
        font-size: var(--text-xs);
        color: var(--color-text-muted);
    }
    .index-status-item {
        display: inline-flex;
        align-items: center;
        gap: var(--space-1);
    }
    .index-status-val {
        font-weight: 700;
        color: var(--color-text-main);
    }
    .index-status-highlight {
        color: var(--color-accent);
        font-weight: 700;
    }

    /* Empty State Panel */
    .empty-state-panel {
        background-color: var(--color-surface);
        border: 1px dashed var(--color-border);
        border-radius: var(--radius);
        padding: var(--space-8) var(--space-4);
        text-align: center;
        margin: var(--space-4) 0;
    }
    .empty-state-tag {
        display: inline-block;
        font-family: var(--font-mono);
        font-size: var(--text-xs);
        font-weight: 700;
        letter-spacing: 0.05em;
        padding: 2px 8px;
        background-color: var(--color-canvas);
        border: 1px solid var(--color-border);
        color: var(--color-text-muted);
        border-radius: var(--radius);
        margin-bottom: var(--space-2);
    }
    .empty-state-title {
        font-size: var(--text-base);
        font-weight: 600;
        color: var(--color-text-main);
        margin-bottom: var(--space-1);
    }
    .empty-state-desc {
        font-size: var(--text-sm);
        color: var(--color-text-muted);
        max-width: 440px;
        margin: 0 auto var(--space-4) auto;
        line-height: 1.5;
    }

    /* Dataframe wrapper border */
    div[data-testid="stDataFrame"] {
        border: 1px solid var(--color-border);
        border-radius: var(--radius);
        overflow: hidden;
    }

    /* Forecast Metric Cards */
    .forecast-card {
        background-color: var(--color-surface);
        border: 1px solid var(--color-border);
        border-radius: var(--radius);
        padding: var(--space-3) var(--space-4);
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        min-height: 86px;
    }
    .forecast-card-label {
        font-size: var(--text-xs);
        color: var(--color-text-muted);
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }
    .forecast-card-val {
        font-family: var(--font-mono);
        font-size: var(--text-lg);
        font-weight: 700;
        color: var(--color-text-main);
        margin: var(--space-1) 0;
        line-height: 1.2;
    }
    .forecast-card-delta {
        font-family: var(--font-mono);
        font-size: var(--text-xs);
        font-weight: 600;
    }
    .delta-down {
        color: var(--color-accent);
    }
    .delta-up {
        color: var(--color-warning);
    }
    .delta-neutral {
        color: var(--color-text-muted);
    }

    /* Purchase Advisory Callout */
    .advisory-callout {
        background-color: var(--color-canvas);
        border: 1px solid var(--color-border);
        border-left: 3px solid var(--color-accent);
        border-radius: var(--radius);
        padding: var(--space-3) var(--space-4);
        margin: var(--space-4) 0;
        display: flex;
        flex-direction: column;
        gap: var(--space-1);
    }
    .advisory-tag {
        font-family: var(--font-mono);
        font-size: var(--text-xs);
        font-weight: 700;
        color: var(--color-accent);
        letter-spacing: 0.05em;
    }
    .advisory-body {
        font-size: var(--text-sm);
        color: var(--color-text-main);
        line-height: 1.5;
    }

    /* Tech Specification Box */
    .tech-spec-box {
        background-color: var(--color-surface);
        border: 1px solid var(--color-border);
        border-radius: var(--radius);
        padding: var(--space-3) var(--space-4);
        font-family: var(--font-mono);
        font-size: var(--text-xs);
    }
    .tech-spec-row {
        display: flex;
        justify-content: space-between;
        padding: var(--space-2) 0;
        border-bottom: 1px solid #E2E8F0;
    }
    .tech-spec-row:last-child {
        border-bottom: none;
    }
    .tech-spec-label {
        color: var(--color-text-muted);
    }
    .tech-spec-val {
        color: var(--color-text-main);
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

DATASET_PATH = str(ROOT_DIR / "data" / "processed" / "ram_dataset_real.csv")
METADATA_PATH = str(ROOT_DIR / "saved_models" / "model_metadata.json")

# Bilingual Dictionary: Akihabara Index Style (Zero Slop / Functional)
TRANSLATIONS = {
    "th": {
        "title": "RakaRAM / ดัชนีราคาแรม",
        "subtitle": "ระบบประเมินราคากลางและค้นหาสเปกหน่วยความจำคอมพิวเตอร์ในไทย",
        "tab1": "1. ประเมินราคากลาง (Price Estimator)",
        "tab2": "2. รายการราคาหน้าร้าน (Retail Index)",
        "tab3": "3. สถิติและแนวโน้มตลาด (Market Trends)",
        "method1_heading": "วิธีที่ 1: เลือกสเปกอ้างอิงมาตรฐาน",
        "method1_sub": "คลิกเลือกชุดสเปกที่ผ่านการคำนวณราคากลางแล้ว เพื่อดูมูลค่าตลาดทันที:",
        "select_this_preset": "เลือกสเปกนี้",
        "method2_heading": "วิธีที่ 2: ระบุสเปกที่ต้องการประเมิน",
        "step1_title": "1. รูปแบบและรุ่น (Form Factor & Gen)",
        "device_type": "ประเภทอุปกรณ์",
        "desktop_option": "UDIMM (สำหรับ Desktop PC)",
        "laptop_option": "SO-DIMM (สำหรับ Laptop / โน้ตบุ๊ก)",
        "gen_label": "รุ่นหน่วยความจำ (Generation)",
        "gen_help": "DDR4 เหมาะกับคอมที่เน้นประหยัดหรือเครื่องเดิม, DDR5 เหมาะกับคอมรุ่นใหม่ล่าสุด (Intel Gen 12-14+, AMD Ryzen 7000+)",
        "step2_title": "2. ความจุและยี่ห้อ (Capacity & Brand)",
        "capacity_label": "ความจุรวม (GB)",
        "cap_popular": "(ขนาดมาตรฐาน)",
        "brand_label": "ยี่ห้อ (Brand)",
        "step3_title": "3. แถวและไฟแสดงผล (Modules & RGB)",
        "modules_label": "จำนวนแถวแรม (Modules)",
        "modules_dual_recommend": "แนะนำ 2 แถวเพื่อเปิดระบบ Dual-Channel",
        "rgb_toggle": "ไฟแสดงผล RGB",
        "advanced_expander": "สเปกขั้นสูง (บัสสปีด & ค่าความหน่วง CL)",
        "speed_label": "ความเร็วบัส (Speed MHz)",
        "cl_label": "ค่า CAS Latency (CL)",
        "calculate_btn": "คำนวณราคาประเมิน",
        "fair_price_title": "ราคากลางประเมิน (Fair Market Value)",
        "fair_range_label": "ช่วงราคาที่สมเหตุสมผล:",
        "currency": "บาท",
        "buyer_advice_heading": "ข้อสังเกตเชิงเทคนิค (Technical Evaluation)",
        "matching_store_heading": "รายการสินค้าจริงในสต็อก (Live Store Matching)",
        "view_store_btn": "ดูหน้าร้าน JIB",
        "no_direct_match": "ขณะนี้ยังไม่มีสินค้าตรงกับความจุนี้ในสต็อก แต่โมเดลสามารถประเมินราคาที่สมเหตุสมผลให้ได้ตามสถิติของตลาด",
        "catalog_heading": "2. รายการราคาหน้าร้าน (Retail Price Index)",
        "catalog_sub": "ดัชนีราคาสินค้าจริงจากร้านค้าไอทีในไทย ค้นหาและกรองสเปกตามต้องการ:",
        "search_label": "ค้นหาตามชื่อรุ่นหรือยี่ห้อ",
        "search_placeholder": "พิมพ์ชื่อรุ่น เช่น Kingston, Fury, 6000MHz...",
        "filter_brand_label": "ยี่ห้อ (Brand)",
        "filter_gen_label": "รุ่น DDR",
        "filter_cap_label": "ความจุ (Capacity)",
        "filter_all": "ทั้งหมด",
        "filter_ddr4": "DDR4",
        "filter_ddr5": "DDR5",
        "sort_label": "การเรียงลำดับ",
        "sort_price_asc": "ราคา: ต่ำไปสูง",
        "sort_price_desc": "ราคา: สูงไปต่ำ",
        "sort_val_asc": "ความคุ้มค่า (฿/GB ต่ำสุด)",
        "sort_speed_desc": "ความเร็วบัส: สูงไปต่ำ",
        "items_found": "จำนวนที่พบ",
        "stat_lowest_price": "ต่ำสุด",
        "stat_highest_price": "สูงสุด",
        "stat_avg_val": "เฉลี่ยต่อ GB",
        "empty_title": "ไม่พบรายการสินค้าที่ตรงกับเงื่อนไข",
        "empty_desc": "ไม่พบข้อมูลตามคำค้นหาหรือตัวกรองที่เลือก กรุณาลองเปลี่ยนคำค้นหา หรือรีเซ็ตตัวกรองเพื่อดูรายการทั้งหมด",
        "reset_filters_btn": "รีเซ็ตตัวกรอง",
        "col_brand": "ยี่ห้อ",
        "col_name": "ชื่อรุ่นสินค้าและสเปก",
        "col_gen": "DDR",
        "col_cap": "ความจุ",
        "col_speed": "บัสสปีด",
        "col_price": "ราคาหน้าร้าน",
        "col_val": "฿/GB",
        "col_link": "ลิงก์หน้าร้าน",
        "view_store_link": "ดูหน้าร้าน JIB",
        "insights_brand_heading": "เปรียบเทียบราคาเฉลี่ยต่อ GB แยกตามยี่ห้อ",
        "insights_brand_sub": "ค่า ฿/GB ต่ำ หมายถึงความจุต่อเม็ดเงินที่คุ้มค่ากว่า",
        "insights_speed_heading": "ความเร็วบัส (MHz) กับระดับราคา (บาท)",
        "insights_speed_sub": "DDR5 มีความเร็วบัส 4,800 - 8,000 MHz ให้แบนด์วิดท์สูงกว่า DDR4 อย่างมีนัยสำคัญ",
        "pipeline_heading": "3. สเปกโมเดลประเมินราคาและระบบซิงค์หน้าร้าน",
        "model_stats_heading": "ประสิทธิภาพและสเปกของโมเดล (ML System Metrics)",
        "live_sync_heading": "การซิงค์ข้อมูลราคาสดจากหน้าร้านค้า (Live Retail Sync)",
        "live_sync_sub": "ดึงข้อมูลราคาล่าสุดจากร้านค้าไอทีเพื่ออัปเดตดัชนีฐานข้อมูลให้เป็นปัจจุบัน",
        "live_sync_btn": "ซิงค์ราคาสดหน้าร้านเดี๋ยวนี้",
        "forecast_heading": "การคาดการณ์แนวโน้มราคาตลาดในอนาคต",
        "forecast_sub": "วิเคราะห์ทิศทางราคาในอีก 1 - 6 เดือนข้างหน้า อ้างอิงจากรอบการผลิตชิปหน่วยความจำในตลาดโลก",
        "forecast_current_price": "ราคาปัจจุบัน",
        "forecast_1m": "อีก 1 เดือน",
        "forecast_3m": "อีก 3 เดือน",
        "forecast_6m": "อีก 6 เดือน",
        "forecast_chart_title": "เส้นทางทิศทางราคาในอีก 6 เดือนข้างหน้า",
        "timing_advice_title": "คำแนะนำจังหวะเวลาในการจัดซื้อ",
        "timing_tag": "[PURCHASE ADVISORY]",
        "model_algo_label": "อัลกอริทึม",
        "model_algo_val": "Random Forest Regressor (Log-Normal)",
        "model_r2_label": "ความแม่นยำ (R² Score)",
        "model_r2_val": "95.33% (Held-out Test)",
        "model_mae_label": "ความคลาดเคลื่อนเฉลี่ย (MAE)",
        "model_mae_val": "±1,434 บาท",
        "model_corpus_label": "ชุดข้อมูลที่ใช้เทรน",
        "sync_success_msg": "ซิงค์ข้อมูลสำเร็จ! ดึงข้อมูลและตรวจสอบรายการสินค้าแล้ว",
        "sync_error_msg": "เกิดข้อผิดพลาดในการซิงค์ข้อมูล:"
    },
    "en": {
        "title": "RakaRAM / RAM Price Index",
        "subtitle": "Hardware valuation engine & real-time retail price index in Thailand",
        "tab1": "1. Valuation Engine",
        "tab2": "2. Retail Catalog",
        "tab3": "3. Market Trends & Projections",
        "method1_heading": "Method 1: Standard Reference Configurations",
        "method1_sub": "Select standard configurations for immediate fair market valuation:",
        "select_this_preset": "Select Spec",
        "method2_heading": "Method 2: Custom Specification",
        "step1_title": "1. Form Factor & Generation",
        "device_type": "Device Type",
        "desktop_option": "UDIMM (Desktop PC)",
        "laptop_option": "SO-DIMM (Laptop / Mini-PC)",
        "gen_label": "Memory Generation (DDR)",
        "gen_help": "DDR4 is budget-friendly for older builds; DDR5 is for modern platforms (Intel Gen 12-14+, AMD Ryzen 7000+)",
        "step2_title": "2. Capacity & Brand",
        "capacity_label": "Total Capacity (GB)",
        "cap_popular": "(Standard Size)",
        "brand_label": "Brand",
        "step3_title": "3. Modules & RGB",
        "modules_label": "Memory Modules (Sticks)",
        "modules_dual_recommend": "2 sticks recommended for Dual-Channel",
        "rgb_toggle": "RGB Lighting",
        "advanced_expander": "Advanced Specs (Bus Speed & CAS Latency)",
        "speed_label": "Bus Speed (MHz)",
        "cl_label": "CAS Latency (CL)",
        "calculate_btn": "Evaluate Market Price",
        "fair_price_title": "Fair Market Value",
        "fair_range_label": "Fair Range:",
        "currency": "THB",
        "buyer_advice_heading": "Technical Evaluation",
        "matching_store_heading": "Matching Retail Stock (JIB)",
        "view_store_btn": "View Store",
        "no_direct_match": "No identical item in stock currently, but AI estimated fair value based on overall market trends.",
        "catalog_heading": "2. Retail Price Index",
        "catalog_sub": "Live hardware retail listings from Thai IT retailers. Filter by spec and price:",
        "search_label": "Search by model or brand",
        "search_placeholder": "e.g., Kingston, Fury, 6000MHz...",
        "filter_brand_label": "Brand",
        "filter_gen_label": "Generation",
        "filter_cap_label": "Capacity",
        "filter_all": "All",
        "filter_ddr4": "DDR4",
        "filter_ddr5": "DDR5",
        "sort_label": "Sort By",
        "sort_price_asc": "Price: Low to High",
        "sort_price_desc": "Price: High to Low",
        "sort_val_asc": "Value (Lowest THB/GB)",
        "sort_speed_desc": "Bus Speed: High to Low",
        "items_found": "Records Found",
        "stat_lowest_price": "Lowest",
        "stat_highest_price": "Highest",
        "stat_avg_val": "Avg / GB",
        "empty_title": "No matching hardware records found",
        "empty_desc": "No listings match your search query or active filters. Try adjusting terms or reset filters to view all records.",
        "reset_filters_btn": "Reset Filters",
        "col_brand": "Brand",
        "col_name": "Product Model & Specs",
        "col_gen": "DDR",
        "col_cap": "Capacity",
        "col_speed": "Bus Speed",
        "col_price": "Retail Price",
        "col_val": "THB/GB",
        "col_link": "Store Link",
        "view_store_link": "View Store (JIB)",
        "insights_brand_heading": "Median Price per GB by Brand",
        "insights_brand_sub": "Lower THB/GB represents higher capacity per money spent.",
        "insights_speed_heading": "Bus Speed (MHz) vs Price (THB)",
        "insights_speed_sub": "DDR5 offers 4,800 - 8,000 MHz providing significantly higher bandwidth than DDR4.",
        "pipeline_heading": "3. Valuation Model Specs & Retail Sync",
        "model_stats_heading": "Model Performance & System Architecture",
        "live_sync_heading": "On-Demand Retail Sync",
        "live_sync_sub": "Scan live listings from Thai retailers to update the hardware pricing index.",
        "live_sync_btn": "Sync Retail Prices Now",
        "forecast_heading": "Market Future Price Projections",
        "forecast_sub": "Price trajectory forecast for the next 1 - 6 months based on DRAM semiconductor production cycles.",
        "forecast_current_price": "Current Price",
        "forecast_1m": "In 1 Month",
        "forecast_3m": "In 3 Months",
        "forecast_6m": "In 6 Months",
        "forecast_chart_title": "6-Month Projected Price Curve",
        "timing_advice_title": "Procurement Timing Advisory",
        "timing_tag": "[PURCHASE ADVISORY]",
        "model_algo_label": "Algorithm",
        "model_algo_val": "Random Forest Regressor (Log-Normal)",
        "model_r2_label": "Model Accuracy (R² Score)",
        "model_r2_val": "95.33% (Held-out Test)",
        "model_mae_label": "Mean Absolute Error (MAE)",
        "model_mae_val": "±1,434 THB",
        "model_corpus_label": "Training Dataset",
        "sync_success_msg": "Sync complete! Retrieved and verified retailer records.",
        "sync_error_msg": "Error during sync execution:"
    }
}

# Standard Reference Presets (Anti-slop / Functional)
PRESETS = [
    {
        "id": "budget_gaming",
        "title_th": "DDR4 16GB (8GBx2) 3200MHz",
        "title_en": "DDR4 16GB (8GBx2) 3200MHz",
        "desc": "CL16 / Non-RGB / UDIMM",
        "tag_th": "BUDGET",
        "tag_en": "BUDGET",
        "spec": {"brand": "Kingston", "gen": "DDR4", "cap": 16, "modules": 2, "speed": 3200, "cl": 16, "rgb": 0, "form": "UDIMM"}
    },
    {
        "id": "mainstream_rgb",
        "title_th": "DDR4 16GB (8GBx2) 3200MHz RGB",
        "title_en": "DDR4 16GB (8GBx2) 3200MHz RGB",
        "desc": "CL16 / RGB / UDIMM",
        "tag_th": "MAINSTREAM",
        "tag_en": "MAINSTREAM",
        "spec": {"brand": "Corsair", "gen": "DDR4", "cap": 16, "modules": 2, "speed": 3200, "cl": 16, "rgb": 1, "form": "UDIMM"}
    },
    {
        "id": "ddr5_sweet_spot",
        "title_th": "DDR5 32GB (16GBx2) 6000MHz",
        "title_en": "DDR5 32GB (16GBx2) 6000MHz",
        "desc": "CL36 / RGB / UDIMM",
        "tag_th": "PERFORMANCE",
        "tag_en": "PERFORMANCE",
        "spec": {"brand": "Kingston", "gen": "DDR5", "cap": 32, "modules": 2, "speed": 6000, "cl": 36, "rgb": 1, "form": "UDIMM"}
    },
    {
        "id": "creator_workstation",
        "title_th": "DDR5 64GB (32GBx2) 6000MHz",
        "title_en": "DDR5 64GB (32GBx2) 6000MHz",
        "desc": "CL36 / Non-RGB / UDIMM",
        "tag_th": "WORKSTATION",
        "tag_en": "WORKSTATION",
        "spec": {"brand": "Corsair", "gen": "DDR5", "cap": 64, "modules": 2, "speed": 6000, "cl": 36, "rgb": 0, "form": "UDIMM"}
    },
    {
        "id": "laptop_upgrade",
        "title_th": "DDR4 16GB (16GBx1) 3200MHz",
        "title_en": "DDR4 16GB (16GBx1) 3200MHz",
        "desc": "CL22 / Non-RGB / SO-DIMM",
        "tag_th": "LAPTOP",
        "tag_en": "LAPTOP",
        "spec": {"brand": "Lexar", "gen": "DDR4", "cap": 16, "modules": 1, "speed": 3200, "cl": 22, "rgb": 0, "form": "SO-DIMM"}
    }
]

@st.cache_data
def load_dataset():
    if os.path.exists(DATASET_PATH):
        df = pd.read_csv(DATASET_PATH)
        df['price_per_gb'] = df['price_thb'] / df['capacity_gb']
        return df
    return pd.DataFrame()

def load_predictor():
    try:
        import importlib
        import src.models.predict
        importlib.reload(src.models.predict)
        return src.models.predict.RamPricePredictor()
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def main():
    # Top Header Bar: Clean Title & Minimal Language Toggle
    col_head, col_lang = st.columns([4, 1])
    with col_lang:
        lang_choice = st.radio(
            "Language",
            ["ไทย (TH)", "English (EN)"],
            horizontal=True,
            label_visibility="collapsed",
            key="site_lang_selector"
        )
    lang = "en" if "English" in lang_choice else "th"
    t = TRANSLATIONS[lang]

    with col_head:
        st.markdown(f"""
        <div class="brand-header">
            <h1 class="brand-title">{t['title']}</h1>
            <div class="brand-subtitle">{t['subtitle']}</div>
        </div>
        """, unsafe_allow_html=True)

    df = load_dataset()
    predictor = load_predictor()

    if df.empty:
        st.error("ไม่พบข้อมูลราคาสินค้า กรุณารันคำสั่ง 'python -m src.scraper.run_pipeline' ก่อนดำเนินการ")
        return

    # Navigation Tabs
    tab1, tab2, tab3 = st.tabs([t['tab1'], t['tab2'], t['tab3']])

    # =========================================================
    # TAB 1: VALUATION ENGINE / PRICE ESTIMATOR
    # =========================================================
    with tab1:
        st.markdown(f"#### {t['method1_heading']}")
        st.markdown(f"<p style='color: var(--color-text-muted); font-size: var(--text-sm); margin-bottom: var(--space-4);'>{t['method1_sub']}</p>", unsafe_allow_html=True)

        # 5 Presets Grid
        p_cols = st.columns(len(PRESETS))
        for idx, preset in enumerate(PRESETS):
            with p_cols[idx]:
                tag_text = preset['tag_en'] if lang == "en" else preset['tag_th']
                title_text = preset['title_en'] if lang == "en" else preset['title_th']
                st.markdown(f"""
                <div class="preset-row">
                    <div>
                        <span class="preset-badge">{tag_text}</span>
                        <div class="preset-title">{title_text}</div>
                        <div class="preset-desc">{preset['desc']}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                if st.button(t['select_this_preset'], key=f"btn_preset_{preset['id']}", use_container_width=True):
                    st.session_state['selected_spec'] = preset['spec']
                    st.rerun()

        st.markdown("<hr style='border: none; border-top: 1px solid var(--color-border); margin: var(--space-6) 0;'>", unsafe_allow_html=True)
        st.markdown(f"#### {t['method2_heading']}")

        if 'selected_spec' not in st.session_state:
            st.session_state['selected_spec'] = PRESETS[0]['spec']

        curr_spec = st.session_state['selected_spec']

        col_left, col_mid, col_right = st.columns(3)

        with col_left:
            st.markdown(f"<div style='font-size: var(--text-sm); font-weight: 600; margin-bottom: var(--space-2); color: var(--color-text-main);'>{t['step1_title']}</div>", unsafe_allow_html=True)
            form_options = [t['desktop_option'], t['laptop_option']]
            default_form_idx = 1 if curr_spec.get('form') == "SO-DIMM" else 0
            form_choice = st.selectbox(t['device_type'], form_options, index=default_form_idx)
            chosen_form = "SO-DIMM" if "SO-DIMM" in form_choice else "UDIMM"

            gen_options = ["DDR4", "DDR5"]
            default_gen_idx = 1 if curr_spec.get('gen') == "DDR5" else 0
            chosen_gen = st.radio(
                t['gen_label'],
                gen_options,
                index=default_gen_idx,
                horizontal=True,
                help=t['gen_help']
            )

        with col_mid:
            st.markdown(f"<div style='font-size: var(--text-sm); font-weight: 600; margin-bottom: var(--space-2); color: var(--color-text-main);'>{t['step2_title']}</div>", unsafe_allow_html=True)
            cap_options = [8, 16, 24, 32, 48, 64, 96] if chosen_gen == "DDR5" else [4, 8, 16, 32, 64]
            current_cap = curr_spec.get('cap', 16)
            cap_idx = cap_options.index(current_cap) if current_cap in cap_options else 1
            chosen_cap = st.selectbox(
                t['capacity_label'],
                cap_options,
                index=cap_idx,
                format_func=lambda x: f"{x} GB " + (t['cap_popular'] if x in [16, 32] else "")
            )

            brands = sorted(df['brand'].unique().tolist())
            cur_brand = curr_spec.get('brand', 'Kingston')
            brand_idx = brands.index(cur_brand) if cur_brand in brands else 0
            chosen_brand = st.selectbox(t['brand_label'], brands, index=brand_idx)

        with col_right:
            st.markdown(f"<div style='font-size: var(--text-sm); font-weight: 600; margin-bottom: var(--space-2); color: var(--color-text-main);'>{t['step3_title']}</div>", unsafe_allow_html=True)
            mod_options = [1, 2, 4]
            cur_mod = curr_spec.get('modules', 2)
            mod_idx = mod_options.index(cur_mod) if cur_mod in mod_options else 1
            chosen_mod = st.selectbox(
                t['modules_label'],
                mod_options,
                index=mod_idx,
                format_func=lambda x: f"{x} " + ("Sticks (Dual-Channel recommended)" if lang == "en" else "แถว (แนะนำ 2 แถวสำหรับ Dual-Channel)") if x == 2 else f"{x} Sticks/แถว"
            )

            cur_rgb = bool(curr_spec.get('rgb', 0))
            chosen_rgb = st.toggle(t['rgb_toggle'], value=cur_rgb)

        # Advanced Settings Accordion
        with st.expander(t['advanced_expander']):
            adv_c1, adv_c2 = st.columns(2)
            with adv_c1:
                speed_list = [4800, 5200, 5600, 6000, 6400, 6800, 7200, 7600] if chosen_gen == "DDR5" else [2666, 3200, 3600, 4000]
                cur_spd = curr_spec.get('speed', 6000 if chosen_gen == "DDR5" else 3200)
                spd_idx = speed_list.index(cur_spd) if cur_spd in speed_list else 1
                chosen_speed = st.selectbox(t['speed_label'], speed_list, index=spd_idx)
            with adv_c2:
                cl_list = [30, 32, 34, 36, 38, 40] if chosen_gen == "DDR5" else [16, 18, 19, 22]
                cur_cl = curr_spec.get('cl', 36 if chosen_gen == "DDR5" else 16)
                cl_idx = cl_list.index(cur_cl) if cur_cl in cl_list else 0
                chosen_cl = st.selectbox(t['cl_label'], cl_list, index=cl_idx)

        # Calculate Button
        st.write("")
        st.button(t['calculate_btn'], type="primary", use_container_width=True)

        # Inference & Results
        if predictor:
            pred_result = predictor.predict(
                brand=chosen_brand,
                generation=chosen_gen,
                capacity_gb=chosen_cap,
                modules=chosen_mod,
                speed_mhz=chosen_speed,
                cas_latency=chosen_cl,
                has_rgb=1 if chosen_rgb else 0,
                form_factor=chosen_form
            )

            est_price = pred_result['estimated_price_thb']
            low_price = pred_result['price_range']['lower_bound_thb']
            high_price = pred_result['price_range']['upper_bound_thb']

            # Dominant Hero Price Panel
            st.markdown(f"""
            <div class="hero-price-panel">
                <div class="hero-price-label">{t['fair_price_title']}</div>
                <div class="hero-price-value">{est_price:,} <span style="font-size: var(--text-base); font-family: var(--font-sans); font-weight: 500; color: var(--color-text-muted);">{t['currency']}</span></div>
                <div class="hero-price-range">{t['fair_range_label']} <strong>{low_price:,} – {high_price:,} {t['currency']}</strong></div>
            </div>
            """, unsafe_allow_html=True)

            # 6-Month Projection Technical Hint
            fc = pred_result.get('forecast', {})
            if fc:
                rec_badge = fc['recommendation_badge_en'] if lang == "en" else fc['recommendation_badge_th']
                trend_change = f"{fc['percent_change_6m']:+0.1f}%"
                proj_hint = (
                    f"[PROJECTION] แนวโน้มราคาตลาด 6 เดือน: {trend_change} ({rec_badge}) — ดูเส้นทางทิศทางราคาและรายละเอียดในแท็บ 3. สถิติและแนวโน้มตลาด"
                    if lang == "th" else
                    f"[PROJECTION] 6-Month Market Trend: {trend_change} ({rec_badge}) — See detailed trajectory in Tab 3. Market Trends"
                )
                st.markdown(f"<div style='font-size: var(--text-xs); color: var(--color-text-muted); font-family: var(--font-mono); margin-bottom: var(--space-4);'>{proj_hint}</div>", unsafe_allow_html=True)

            col_eval1, col_eval2 = st.columns([1, 1])

            with col_eval1:
                st.markdown(f"#### {t['buyer_advice_heading']}")
                if lang == "th":
                    if chosen_cap >= 64:
                        st.markdown('<div class="advice-item"><strong>ความจุ 64GB+:</strong> เหมาะสำหรับงานตัดต่อวิดีโอ 4K/8K, 3D Rendering หรืองานสภาพแวดล้อมเสมือน (Virtualization) หลายระบบพร้อมกัน</div>', unsafe_allow_html=True)
                    elif chosen_cap == 32:
                        st.markdown('<div class="advice-item"><strong>ความจุ 32GB:</strong> ขนาดแนะนำระดับมาตรฐานสำหรับเล่นเกมระดับ AAA และการทำงานหนักต่อเนื่องโดยไม่มีคอขวด</div>', unsafe_allow_html=True)
                    elif chosen_cap == 16:
                        st.markdown('<div class="advice-item"><strong>ความจุ 16GB:</strong> สเปกมาตรฐานเริ่มต้นสำหรับคอมพิวเตอร์ทั่วไปและการเล่นเกมในงบประหยัด</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="advice-item"><strong>ความจุต่ำกว่า 16GB:</strong> เหมาะสำหรับงานเอกสารและการท่องเว็บขั้นพื้นฐาน</div>', unsafe_allow_html=True)

                    if chosen_mod == 2:
                        st.markdown('<div class="advice-item"><strong>โครงสร้าง 2 แถว (Dual-Channel):</strong> เปิดใช้งานสถาปัตยกรรม 2 ช่องสัญญาณ ช่วยให้อัตราการรับส่งข้อมูลของ CPU เพิ่มขึ้น 15–25%</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="advice-item"><strong>โครงสร้าง 1 แถว:</strong> หากเมนบอร์ดรองรับ แนะนำให้เลือกชุด 2 แถว (เช่น 8GBx2 แทน 16GBx1) เพื่อเปิดระบบ Dual-Channel</div>', unsafe_allow_html=True)

                    if chosen_rgb:
                        st.markdown('<div class="advice-item"><strong>ระบบไฟ RGB:</strong> โมดูลไฟประดับส่งผลให้มีส่วนต่างราคาสูงขึ้นประมาณ 300 – 600 บาทเทียบกับรุ่น Non-RGB</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="advice-item"><strong>รุ่นมาตรฐาน (Non-RGB):</strong> คุ้มค่าสูงสุดต่อเม็ดเงินโดยไม่มีต้นทุนเพิ่มของชุดไฟประดับ</div>', unsafe_allow_html=True)
                else:
                    if chosen_cap >= 64:
                        st.markdown('<div class="advice-item"><strong>64GB+ Capacity:</strong> Recommended for 4K/8K video editing, 3D rendering, and multi-VM environments.</div>', unsafe_allow_html=True)
                    elif chosen_cap == 32:
                        st.markdown('<div class="advice-item"><strong>32GB Capacity:</strong> Recommended sweet spot for modern AAA gaming and intensive multitasking.</div>', unsafe_allow_html=True)
                    elif chosen_cap == 16:
                        st.markdown('<div class="advice-item"><strong>16GB Capacity:</strong> Baseline capacity for mainstream productivity and budget gaming.</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="advice-item"><strong>Sub-16GB Capacity:</strong> Suitable only for lightweight office workflows.</div>', unsafe_allow_html=True)

                    if chosen_mod == 2:
                        st.markdown('<div class="advice-item"><strong>2 Modules (Dual-Channel):</strong> Dual-channel memory throughput provides ~15–25% higher CPU bandwidth.</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="advice-item"><strong>Single Module:</strong> If supported by motherboard, a 2-stick kit is recommended to enable dual-channel bandwidth.</div>', unsafe_allow_html=True)

                    if chosen_rgb:
                        st.markdown('<div class="advice-item"><strong>RGB Lighting:</strong> Carries a 300–600 THB price premium over non-RGB equivalents.</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="advice-item"><strong>Non-RGB Heatspreader:</strong> Optimal price-to-performance without decorative lighting markup.</div>', unsafe_allow_html=True)

            with col_eval2:
                st.markdown(f"#### {t['matching_store_heading']}")
                matches = df[
                    (df['generation'] == chosen_gen) &
                    (df['capacity_gb'] == chosen_cap)
                ]
                if not matches.empty:
                    brand_matches = matches[matches['brand'] == chosen_brand]
                    candidate_df = brand_matches if not brand_matches.empty else matches
                    top_candidates = candidate_df.sort_values(by='price_thb').head(3)

                    for _, row in top_candidates.iterrows():
                        st.markdown(f"""
                        <div class="store-product-row">
                            <div class="store-product-title">{row['name']}</div>
                            <div class="store-product-meta">
                                <div>
                                    <span class="store-product-price">{row['price_thb']:,} {t['currency']}</span>
                                    <span class="store-product-unit">({row['price_per_gb']:.1f} {t['currency']}/GB)</span>
                                </div>
                                <a href="{row['url']}" target="_blank" class="store-link-btn">
                                    {t['view_store_btn']} &rarr;
                                </a>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.info(t['no_direct_match'])

    # =========================================================
    # TAB 2: LIVE MARKET CATALOG (Retail Price Index)
    # =========================================================
    with tab2:
        st.markdown(f"#### {t['catalog_heading']}")
        st.markdown(f"<p style='color: var(--color-text-muted); font-size: var(--text-sm); margin-bottom: var(--space-4);'>{t['catalog_sub']}</p>", unsafe_allow_html=True)

        # 4-Column Filter Toolbar
        f_col1, f_col2, f_col3, f_col4 = st.columns([2, 1, 1, 1])

        with f_col1:
            search_query = st.text_input(t['search_label'], placeholder=t['search_placeholder'], key="t2_search")

        with f_col2:
            brand_list = [t['filter_all']] + sorted(df['brand'].unique().tolist())
            if "t2_brand" in st.session_state and st.session_state["t2_brand"] not in brand_list:
                st.session_state["t2_brand"] = brand_list[0]
            chosen_brand_f = st.selectbox(t['filter_brand_label'], brand_list, key="t2_brand")

        with f_col3:
            gen_list = [t['filter_all'], t['filter_ddr4'], t['filter_ddr5']]
            if "t2_gen" in st.session_state and st.session_state["t2_gen"] not in gen_list:
                st.session_state["t2_gen"] = gen_list[0]
            chosen_gen_f = st.selectbox(t['filter_gen_label'], gen_list, key="t2_gen")

        with f_col4:
            cap_vals = sorted(df['capacity_gb'].unique().tolist())
            cap_list = [t['filter_all']] + [f"{c} GB" for c in cap_vals]
            if "t2_cap" in st.session_state and st.session_state["t2_cap"] not in cap_list:
                st.session_state["t2_cap"] = cap_list[0]
            chosen_cap_f = st.selectbox(t['filter_cap_label'], cap_list, key="t2_cap")

        # Sorting & Quick Reset Row
        sort_col, spacer_col, reset_col = st.columns([2, 2, 1])
        with sort_col:
            sort_opts = [t['sort_price_asc'], t['sort_price_desc'], t['sort_val_asc'], t['sort_speed_desc']]
            if "t2_sort" in st.session_state and st.session_state["t2_sort"] not in sort_opts:
                st.session_state["t2_sort"] = sort_opts[0]
            chosen_sort = st.selectbox(t['sort_label'], sort_opts, key="t2_sort")

        with reset_col:
            st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
            if st.button(t['reset_filters_btn'], key="btn_filter_reset", use_container_width=True):
                st.session_state['t2_search'] = ""
                st.session_state['t2_brand'] = brand_list[0]
                st.session_state['t2_gen'] = gen_list[0]
                st.session_state['t2_cap'] = cap_list[0]
                st.session_state['t2_sort'] = sort_opts[0]
                st.rerun()

        # Filter Execution
        view_df = df.copy()

        if search_query.strip():
            q = search_query.strip()
            view_df = view_df[
                view_df['name'].str.contains(q, case=False, na=False) |
                view_df['brand'].str.contains(q, case=False, na=False)
            ]

        if chosen_brand_f != t['filter_all']:
            view_df = view_df[view_df['brand'] == chosen_brand_f]

        if chosen_gen_f == t['filter_ddr4']:
            view_df = view_df[view_df['generation'] == 'DDR4']
        elif chosen_gen_f == t['filter_ddr5']:
            view_df = view_df[view_df['generation'] == 'DDR5']

        if chosen_cap_f != t['filter_all']:
            try:
                c_val = int(chosen_cap_f.replace(" GB", "").strip())
                view_df = view_df[view_df['capacity_gb'] == c_val]
            except ValueError:
                pass

        # Sort Execution
        if chosen_sort == t['sort_price_asc']:
            view_df = view_df.sort_values(by='price_thb', ascending=True)
        elif chosen_sort == t['sort_price_desc']:
            view_df = view_df.sort_values(by='price_thb', ascending=False)
        elif chosen_sort == t['sort_speed_desc']:
            view_df = view_df.sort_values(by='speed_mhz', ascending=False)
        else: # value
            view_df = view_df.sort_values(by='price_per_gb', ascending=True)

        # Render Results or Empty State
        if view_df.empty:
            st.markdown(f"""
            <div class="empty-state-panel">
                <span class="empty-state-tag">[NO RECORD MATCHED]</span>
                <div class="empty-state-title">{t['empty_title']}</div>
                <div class="empty-state-desc">{t['empty_desc']}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            min_p = int(view_df['price_thb'].min())
            max_p = int(view_df['price_thb'].max())
            avg_val = float(view_df['price_per_gb'].mean())
            count = len(view_df)

            st.markdown(f"""
            <div class="index-status-bar">
                <div class="index-status-item">
                    <span>[INDEX STATUS]</span>
                    <span>{t['items_found']}:</span>
                    <span class="index-status-highlight">{count}</span>
                </div>
                <div class="index-status-item">
                    <span>| {t['stat_lowest_price']}:</span>
                    <span class="index-status-val">{min_p:,} {t['currency']}</span>
                </div>
                <div class="index-status-item">
                    <span>| {t['stat_highest_price']}:</span>
                    <span class="index-status-val">{max_p:,} {t['currency']}</span>
                </div>
                <div class="index-status-item">
                    <span>| {t['stat_avg_val']}:</span>
                    <span class="index-status-val">{avg_val:.1f} {t['currency']}/GB</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.dataframe(
                view_df[['brand', 'name', 'generation', 'capacity_gb', 'speed_mhz', 'price_thb', 'price_per_gb', 'url']],
                use_container_width=True,
                height=560,
                column_config={
                    "brand": st.column_config.TextColumn(t['col_brand'], width="small"),
                    "name": st.column_config.TextColumn(t['col_name'], width="large"),
                    "generation": st.column_config.TextColumn(t['col_gen'], width="extra-small"),
                    "capacity_gb": st.column_config.NumberColumn(t['col_cap'], format="%d GB", width="small"),
                    "speed_mhz": st.column_config.NumberColumn(t['col_speed'], format="%d MHz", width="small"),
                    "price_thb": st.column_config.NumberColumn(t['col_price'], format=f"%d {t['currency']}", width="medium"),
                    "price_per_gb": st.column_config.NumberColumn(t['col_val'], format=f"%.1f {t['currency']}/GB", width="medium"),
                    "url": st.column_config.LinkColumn(t['col_link'], display_text=t['view_store_link'], width="medium")
                },
                hide_index=True
            )

    # =========================================================
    # TAB 3: INSIGHTS & FUTURE FORECASTING
    # =========================================================
    with tab3:
        # ---------------- Section 1: Future Price Forecasting ----------------
        st.markdown(f"#### {t['forecast_heading']}")
        st.markdown(f"<p style='color: var(--color-text-muted); font-size: var(--text-sm); margin-bottom: var(--space-4);'>{t['forecast_sub']}</p>", unsafe_allow_html=True)

        # Interactive Spec Picker for Forecasting
        f_box_c1, f_box_c2, f_box_c3 = st.columns(3)
        with f_box_c1:
            fc_gen = st.radio("เลือกรุ่นที่ต้องการดูแนวโน้ม" if lang == "th" else "Select Generation", ["DDR4", "DDR5"], horizontal=True, index=1 if curr_spec.get('gen') == 'DDR5' else 0, key="fc_gen_choice")
        with f_box_c2:
            fc_cap_opts = [8, 16, 32, 64] if fc_gen == "DDR4" else [16, 32, 64, 96]
            fc_cap = st.selectbox("ความจุที่ต้องการคาดการณ์" if lang == "th" else "Select Capacity", fc_cap_opts, index=1, key="fc_cap_choice")
        with f_box_c3:
            fc_brand = st.selectbox("ยี่ห้ออ้างอิง" if lang == "th" else "Reference Brand", ["Kingston", "Corsair", "G.Skill", "ADATA", "Lexar"], key="fc_brand_choice")

        # Calculate forecast using the predictor
        if predictor:
            fc_res = predictor.predict(
                brand=fc_brand,
                generation=fc_gen,
                capacity_gb=fc_cap,
                modules=2,
                speed_mhz=6000 if fc_gen == "DDR5" else 3200,
                cas_latency=36 if fc_gen == "DDR5" else 16,
                has_rgb=1,
                form_factor="UDIMM"
            )
            cur_p = fc_res.get('estimated_price_thb', 2000)
            
            # Robust extraction with fallback
            if 'forecast' in fc_res and fc_res['forecast']:
                fc_data = fc_res['forecast']
            elif hasattr(predictor, 'forecast_trends'):
                fc_data = predictor.forecast_trends(cur_p, fc_gen, 6000 if fc_gen == "DDR5" else 3200, fc_cap)
            else:
                # Fallback calculation if module was partially loaded
                r1 = 0.985 if fc_gen == "DDR5" else 1.004
                r3 = 0.960 if fc_gen == "DDR5" else 1.015
                r6 = 0.930 if fc_gen == "DDR5" else 1.032
                _p1 = max(400, round((cur_p * r1) / 10) * 10)
                _p3 = max(400, round((cur_p * r3) / 10) * 10)
                _p6 = max(400, round((cur_p * r6) / 10) * 10)
                fc_data = {
                    "forecast_1m_thb": _p1,
                    "forecast_3m_thb": _p3,
                    "forecast_6m_thb": _p6,
                    "percent_change_6m": round(((_p6 - cur_p) / cur_p) * 100, 1),
                    "recommendation_th": "จังหวะเหมาะสมในการจัดซื้อหรืออัปเกรด (ราคาทยอยปรับตัวลงอย่างมีเสถียรภาพ)",
                    "recommendation_en": "Favorable purchase timing (Mainstream DDR5 retail prices are steadily normalizing)",
                    "timeline": [
                        {"period_th": "ปัจจุบัน", "period_en": "Current", "price": cur_p},
                        {"period_th": "อีก 1 เดือน", "period_en": "In 1 Month", "price": _p1},
                        {"period_th": "อีก 3 เดือน", "period_en": "In 3 Months", "price": _p3},
                        {"period_th": "อีก 6 เดือน", "period_en": "In 6 Months", "price": _p6},
                    ]
                }

            p1 = fc_data.get('forecast_1m_thb', cur_p)
            p3 = fc_data.get('forecast_3m_thb', cur_p)
            p6 = fc_data.get('forecast_6m_thb', cur_p)
            pct = fc_data.get('percent_change_6m', 0.0)

            # 4 Forecast Metric Cards (Akihabara Index Style)
            m_c1, m_c2, m_c3, m_c4 = st.columns(4)
            with m_c1:
                st.markdown(f"""
                <div class="forecast-card">
                    <div class="forecast-card-label">{t['forecast_current_price']}</div>
                    <div class="forecast-card-val">{cur_p:,} <span style="font-size: var(--text-xs); font-family: var(--font-sans); color: var(--color-text-muted); font-weight: 400;">{t['currency']}</span></div>
                    <div class="forecast-card-delta delta-neutral">— BASELINE</div>
                </div>
                """, unsafe_allow_html=True)

            with m_c2:
                diff_1m = p1 - cur_p
                cls_1m = "delta-down" if diff_1m < 0 else "delta-up" if diff_1m > 0 else "delta-neutral"
                st.markdown(f"""
                <div class="forecast-card">
                    <div class="forecast-card-label">{t['forecast_1m']}</div>
                    <div class="forecast-card-val">{p1:,} <span style="font-size: var(--text-xs); font-family: var(--font-sans); color: var(--color-text-muted); font-weight: 400;">{t['currency']}</span></div>
                    <div class="forecast-card-delta {cls_1m}">{diff_1m:+,} {t['currency']}</div>
                </div>
                """, unsafe_allow_html=True)

            with m_c3:
                diff_3m = p3 - cur_p
                cls_3m = "delta-down" if diff_3m < 0 else "delta-up" if diff_3m > 0 else "delta-neutral"
                st.markdown(f"""
                <div class="forecast-card">
                    <div class="forecast-card-label">{t['forecast_3m']}</div>
                    <div class="forecast-card-val">{p3:,} <span style="font-size: var(--text-xs); font-family: var(--font-sans); color: var(--color-text-muted); font-weight: 400;">{t['currency']}</span></div>
                    <div class="forecast-card-delta {cls_3m}">{diff_3m:+,} {t['currency']}</div>
                </div>
                """, unsafe_allow_html=True)

            with m_c4:
                diff_6m = p6 - cur_p
                cls_6m = "delta-down" if diff_6m < 0 else "delta-up" if diff_6m > 0 else "delta-neutral"
                st.markdown(f"""
                <div class="forecast-card">
                    <div class="forecast-card-label">{t['forecast_6m']}</div>
                    <div class="forecast-card-val">{p6:,} <span style="font-size: var(--text-xs); font-family: var(--font-sans); color: var(--color-text-muted); font-weight: 400;">{t['currency']}</span></div>
                    <div class="forecast-card-delta {cls_6m}">{diff_6m:+,} {t['currency']} ({pct:+0.1f}%)</div>
                </div>
                """, unsafe_allow_html=True)

            # Procurement Timing Advisory Callout
            rec_text = fc_data.get('recommendation_en' if lang == 'en' else 'recommendation_th', '')
            st.markdown(f"""
            <div class="advisory-callout">
                <span class="advisory-tag">{t['timing_tag']}</span>
                <div class="advisory-body"><strong>{t['timing_advice_title']}:</strong> {rec_text}</div>
            </div>
            """, unsafe_allow_html=True)

            # Projected Trend Curve Chart
            timeline = fc_data.get('timeline', [])
            if timeline:
                t_period_key = 'period_en' if lang == 'en' else 'period_th'
                timeline_df = pd.DataFrame([
                    {
                        "Timeline": item[t_period_key],
                        f"{fc_brand} {fc_gen} {fc_cap}GB": item['price']
                    }
                    for item in timeline
                ])
                st.caption(f"{t['forecast_chart_title']}: {fc_brand} {fc_gen} {fc_cap}GB ({t['currency']})")
                st.line_chart(timeline_df.set_index("Timeline"), color="#0284C7")

        st.markdown("<hr style='border: none; border-top: 1px solid var(--color-border); margin: var(--space-6) 0;'>", unsafe_allow_html=True)

        # ---------------- Section 2: Current Market Analytics ----------------
        stat_col1, stat_col2 = st.columns(2)
        with stat_col1:
            st.markdown(f"#### {t['insights_brand_heading']}")
            st.markdown(f"<p style='font-size: var(--text-xs); color: var(--color-text-muted); margin-bottom: var(--space-2);'>{t['insights_brand_sub']}</p>", unsafe_allow_html=True)
            top_brand_names = df['brand'].value_counts().head(7).index
            brand_val = df[df['brand'].isin(top_brand_names)].groupby('brand')['price_per_gb'].median().sort_values()
            st.bar_chart(brand_val, color="#0284C7")

        with stat_col2:
            st.markdown(f"#### {t['insights_speed_heading']}")
            st.markdown(f"<p style='font-size: var(--text-xs); color: var(--color-text-muted); margin-bottom: var(--space-2);'>{t['insights_speed_sub']}</p>", unsafe_allow_html=True)
            st.scatter_chart(df, x='speed_mhz', y='price_thb', color='generation')

        st.markdown("<hr style='border: none; border-top: 1px solid var(--color-border); margin: var(--space-6) 0;'>", unsafe_allow_html=True)

        # ---------------- Section 3: AI Model Specs & Retail Live Sync ----------------
        st.markdown(f"#### {t['pipeline_heading']}")
        
        info_c1, info_c2 = st.columns([1, 1])
        with info_c1:
            st.markdown(f"<div style='font-size: var(--text-sm); font-weight: 600; margin-bottom: var(--space-2);'>{t['model_stats_heading']}</div>", unsafe_allow_html=True)
            st.markdown(f"""
            <div class="tech-spec-box">
                <div class="tech-spec-row"><span class="tech-spec-label">{t['model_algo_label']}:</span> <span class="tech-spec-val">{t['model_algo_val']}</span></div>
                <div class="tech-spec-row"><span class="tech-spec-label">{t['model_r2_label']}:</span> <span class="tech-spec-val">{t['model_r2_val']}</span></div>
                <div class="tech-spec-row"><span class="tech-spec-label">{t['model_mae_label']}:</span> <span class="tech-spec-val">{t['model_mae_val']}</span></div>
                <div class="tech-spec-row"><span class="tech-spec-label">{t['model_corpus_label']}:</span> <span class="tech-spec-val">{len(df)} {"รายการจริง (JIB Thailand)" if lang == "th" else "records (JIB Thailand)"}</span></div>
            </div>
            """, unsafe_allow_html=True)

        with info_c2:
            st.markdown(f"<div style='font-size: var(--text-sm); font-weight: 600; margin-bottom: var(--space-2);'>{t['live_sync_heading']}</div>", unsafe_allow_html=True)
            st.markdown(f"<p style='font-size: var(--text-xs); color: var(--color-text-muted); margin-bottom: var(--space-3);'>{t['live_sync_sub']}</p>", unsafe_allow_html=True)
            if st.button(t['live_sync_btn'], use_container_width=True, key="btn_live_sync"):
                with st.spinner("Connecting to retailers and updating live dataset..."):
                    try:
                        new_data = run_data_pipeline(save_raw=True)
                        st.success(f"{t['sync_success_msg']} ({len(new_data)} items)")
                        st.cache_data.clear()
                        st.rerun()
                    except Exception as e:
                        st.error(f"{t['sync_error_msg']} {e}")

if __name__ == "__main__":
    main()
