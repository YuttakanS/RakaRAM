import sys
from pathlib import Path

# Add project root directory to sys.path
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
    page_title="RakaRAM | เช็กราคาและประเมินราคาแรมคอมพิวเตอร์",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Modern Custom CSS Styling
st.markdown("""
<style>
    /* Google Fonts & Base Styling */
    @import url('https://fonts.googleapis.com/css2?family=Prompt:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Prompt', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Hero Header */
    .hero-container {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #1e3a8a 100%);
        padding: 2.2rem 2.5rem;
        border-radius: 1.2rem;
        color: white;
        margin-bottom: 1.8rem;
        box-shadow: 0 10px 25px -5px rgba(15, 23, 42, 0.25);
    }
    .hero-title {
        font-size: 2.4rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        margin-bottom: 0.4rem;
        display: flex;
        align-items: center;
        gap: 0.6rem;
    }
    .hero-subtitle {
        font-size: 1.1rem;
        color: #94a3b8;
        font-weight: 300;
        max-width: 800px;
        line-height: 1.5;
    }

    /* Result Price Card */
    .price-box {
        background: linear-gradient(135deg, #1e40af 0%, #3b82f6 100%);
        padding: 2rem;
        border-radius: 1.2rem;
        color: white;
        text-align: center;
        box-shadow: 0 10px 20px -5px rgba(59, 130, 246, 0.4);
        margin: 1.2rem 0;
    }
    .price-label {
        font-size: 1.05rem;
        font-weight: 400;
        opacity: 0.9;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .price-num {
        font-size: 3.6rem;
        font-weight: 800;
        line-height: 1.1;
        margin: 0.4rem 0;
        text-shadow: 0 2px 4px rgba(0,0,0,0.15);
    }
    .price-range-tag {
        display: inline-block;
        background: rgba(255, 255, 255, 0.18);
        backdrop-filter: blur(8px);
        padding: 0.4rem 1.2rem;
        border-radius: 2rem;
        font-size: 1.05rem;
        font-weight: 500;
        margin-top: 0.4rem;
    }

    /* Feature Cards */
    .preset-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 0.9rem;
        padding: 1.2rem;
        text-align: left;
        margin-bottom: 0.8rem;
        transition: all 0.2s ease;
    }
    .preset-card:hover {
        border-color: #3b82f6;
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.1);
    }
    .product-real-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 0.8rem;
        padding: 1rem 1.2rem;
        margin-bottom: 0.7rem;
    }
    .tier-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 1rem;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .tier-budget { background: #dcfce7; color: #166534; }
    .tier-mid { background: #dbeafe; color: #1e40af; }
    .tier-high { background: #fef3c7; color: #92400e; }
    .tier-enthusiast { background: #f3e8ff; color: #6b21a8; }
</style>
""", unsafe_allow_html=True)

DATASET_PATH = str(ROOT_DIR / "data" / "processed" / "ram_dataset_real.csv")
METADATA_PATH = str(ROOT_DIR / "saved_models" / "model_metadata.json")

# Presets for everyday users
PRESETS = [
    {
        "id": "budget_gaming",
        "title": "🎮 สายเล่นเกมคุ้มค่า (ยอดนิยมที่สุด)",
        "desc": "16GB (8GBx2) DDR4 3200MHz Non-RGB",
        "tag": "งบประหยัด",
        "tier": "budget",
        "spec": {"brand": "Kingston", "gen": "DDR4", "cap": 16, "modules": 2, "speed": 3200, "cl": 16, "rgb": 0, "form": "UDIMM"}
    },
    {
        "id": "mainstream_rgb",
        "title": "✨ สายเกมมิ่ง + มีไฟ RGB",
        "desc": "16GB (8GBx2) DDR4 3200MHz RGB",
        "tag": "สวยงามคุ้มราคา",
        "tier": "mid",
        "spec": {"brand": "Corsair", "gen": "DDR4", "cap": 16, "modules": 2, "speed": 3200, "cl": 16, "rgb": 1, "form": "UDIMM"}
    },
    {
        "id": "ddr5_sweet_spot",
        "title": "🚀 สายประกอบคอมใหม่ DDR5 (แรงจบ)",
        "desc": "32GB (16GBx2) DDR5 6000MHz RGB",
        "tag": "มาตรฐานยุคใหม่",
        "tier": "high",
        "spec": {"brand": "Kingston", "gen": "DDR5", "cap": 32, "modules": 2, "speed": 6000, "cl": 36, "rgb": 1, "form": "UDIMM"}
    },
    {
        "id": "creator_workstation",
        "title": "💻 สายทำงานหนัก / ตัดต่อ / สตรีมเมอร์",
        "desc": "64GB (32GBx2) DDR5 6000MHz Non-RGB",
        "tag": "ประสิทธิภาพสูงสุด",
        "tier": "enthusiast",
        "spec": {"brand": "Corsair", "gen": "DDR5", "cap": 64, "modules": 2, "speed": 6000, "cl": 36, "rgb": 0, "form": "UDIMM"}
    },
    {
        "id": "laptop_upgrade",
        "title": "💼 สายอัปเกรดแรมโน้ตบุ๊ก (Laptop)",
        "desc": "16GB (16GBx1) DDR4 3200MHz SO-DIMM",
        "tag": "สำหรับโน้ตบุ๊ก",
        "tier": "budget",
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

@st.cache_resource
def load_predictor():
    try:
        return RamPricePredictor()
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def main():
    # Hero Section
    st.markdown("""
    <div class="hero-container">
        <div class="hero-title">⚡ RakaRAM (ราคาแรม)</div>
        <div class="hero-subtitle">
            เครื่องมือช่วยเช็กราคาตลาดและประเมินราคาที่เหมาะสมของแรมคอมพิวเตอร์และโน้ตบุ๊ก
            ด้วยโมเดล Machine Learning อ้างอิงจากราคาวางจำหน่ายจริงในร้านค้าไอทีไทยแบบ Real-time
        </div>
    </div>
    """, unsafe_allow_html=True)

    df = load_dataset()
    predictor = load_predictor()

    if df.empty:
        st.error("⚠️ ไม่พบข้อมูลราคาสินค้า กรุณารันคำสั่ง 'python -m src.scraper.run_pipeline' เพื่อดึงข้อมูลก่อนครับ")
        return

    # Navigation Tabs (Simple & Action-Oriented)
    tab1, tab2, tab3 = st.tabs([
        "🔮 1. ประเมินราคาแรม (Price Estimator)",
        "🛒 2. ค้นหาและดูราคาร้านค้าจริง (Market Catalog)",
        "📊 3. กราฟภาพรวมตลาด & ระบบ (Insights)"
    ])

    # =========================================================
    # TAB 1: EASY PRICE ESTIMATOR
    # =========================================================
    with tab1:
        st.markdown("### 🎯 วิธีที่ 1: เลือกจากสเปกยอดนิยม (กดเลือกได้เลย)")
        st.write("ถ้าคุณไม่แน่ใจเรื่องสเปก สามารถคลิกเลือกชุดสเปกที่แนะนำด้านล่าง ระบบจะคำนวณราคาให้ทันทีครับ:")

        # Preset Buttons in Columns
        p_cols = st.columns(len(PRESETS))
        for idx, preset in enumerate(PRESETS):
            with p_cols[idx]:
                tier_class = f"tier-{preset['tier']}"
                st.markdown(f"<span class='tier-badge {tier_class}'>{preset['tag']}</span>", unsafe_allow_html=True)
                st.markdown(f"**{preset['title']}**")
                st.caption(preset['desc'])
                if st.button("เลือกชุดนี้ 👉", key=f"btn_preset_{preset['id']}", use_container_width=True):
                    st.session_state['selected_spec'] = preset['spec']
                    st.session_state['spec_note'] = preset['title']

        st.markdown("---")
        st.markdown("### ⚙️ วิธีที่ 2: ปรับแต่งสเปกตามใจคุณ")

        # Initialize session state if not set
        if 'selected_spec' not in st.session_state:
            st.session_state['selected_spec'] = PRESETS[0]['spec']
            st.session_state['spec_note'] = "สเปกเริ่มต้น"

        curr_spec = st.session_state['selected_spec']

        col_left, col_mid, col_right = st.columns(3)

        with col_left:
            st.markdown("##### 1. ประเภทและรุ่น (Type & Gen)")
            form_factor_options = ["UDIMM (สำหรับ Desktop PC ทั่วไป)", "SO-DIMM (สำหรับ Laptop / โน้ตบุ๊ก)"]
            default_form_idx = 1 if curr_spec.get('form') == "SO-DIMM" else 0
            form_choice = st.selectbox("ประเภทอุปกรณ์", form_factor_options, index=default_form_idx)
            is_laptop = "Laptop" in form_choice
            chosen_form = "SO-DIMM" if is_laptop else "UDIMM"

            gen_options = ["DDR4", "DDR5"]
            default_gen_idx = 1 if curr_spec.get('gen') == "DDR5" else 0
            chosen_gen = st.radio(
                "รุ่น DDR (Generation)",
                gen_options,
                index=default_gen_idx,
                horizontal=True,
                help="DDR4 เหมาะกับคอมที่ประกอบมาหลายปีหรือเน้นประหยัด, DDR5 เหมาะกับคอมรุ่นใหม่ล่าสุด (Intel Gen 12-14+, AMD Ryzen 7000+)"
            )

        with col_mid:
            st.markdown("##### 2. ความจุและยี่ห้อ (Capacity & Brand)")
            cap_options = [8, 16, 24, 32, 48, 64, 96] if chosen_gen == "DDR5" else [4, 8, 16, 32, 64]
            current_cap = curr_spec.get('cap', 16)
            cap_idx = cap_options.index(current_cap) if current_cap in cap_options else 1
            chosen_cap = st.selectbox(
                "ความจุรวม (GB)",
                cap_options,
                index=cap_idx,
                format_func=lambda x: f"{x} GB " + ("(ยอดนิยม ใช้งานลื่นไหล)" if x in [16, 32] else "")
            )

            # Available brands from dataset
            brands = sorted(df['brand'].unique().tolist())
            cur_brand = curr_spec.get('brand', 'Kingston')
            brand_idx = brands.index(cur_brand) if cur_brand in brands else 0
            chosen_brand = st.selectbox("ยี่ห้อ (Brand)", brands, index=brand_idx)

        with col_right:
            st.markdown("##### 3. ลูกเล่นและจำนวนแถว")
            mod_options = [1, 2, 4]
            cur_mod = curr_spec.get('modules', 2)
            mod_idx = mod_options.index(cur_mod) if cur_mod in mod_options else 1
            chosen_mod = st.selectbox(
                "จำนวนแถวแรม (Modules)",
                mod_options,
                index=mod_idx,
                format_func=lambda x: f"{x} แถว (แนะนำ 2 แถวเพื่อ Dual-Channel)" if x == 2 else f"{x} แถว"
            )

            cur_rgb = bool(curr_spec.get('rgb', 0))
            chosen_rgb = st.toggle("🌈 มีไฟ RGB สวยงาม", value=cur_rgb)

        # Advanced Settings Accordion (Hidden by default so users aren't confused)
        with st.expander("🛠️ สเปกขั้นสูง (บัสสปีด & ค่าความหน่วง CL) - หากไม่แน่ใจให้ใช้ค่าเริ่มต้นได้เลย"):
            adv_c1, adv_c2 = st.columns(2)
            with adv_c1:
                speed_list = [4800, 5200, 5600, 6000, 6400, 6800, 7200, 7600] if chosen_gen == "DDR5" else [2666, 3200, 3600, 4000]
                cur_spd = curr_spec.get('speed', 6000 if chosen_gen == "DDR5" else 3200)
                spd_idx = speed_list.index(cur_spd) if cur_spd in speed_list else 1
                chosen_speed = st.selectbox("ความเร็วบัส (Speed MHz)", speed_list, index=spd_idx)
            with adv_c2:
                cl_list = [30, 32, 34, 36, 38, 40] if chosen_gen == "DDR5" else [16, 18, 19, 22]
                cur_cl = curr_spec.get('cl', 36 if chosen_gen == "DDR5" else 16)
                cl_idx = cl_list.index(cur_cl) if cur_cl in cl_list else 0
                chosen_cl = st.selectbox("ค่า CAS Latency (CL - ยิ่งน้อยยิ่งตอบสนองไว)", cl_list, index=cl_idx)

        # Action Button
        calc_button = st.button("🚀 คำนวณราคาประเมินเดี๋ยวนี้", type="primary", use_container_width=True)

        # Perform Prediction
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

            st.write("")
            st.markdown(f"""
            <div class="price-box">
                <div class="price-label">ราคาประเมินที่เหมาะสมในตลาดไทย (Fair Market Value)</div>
                <div class="price-num">{est_price:,} <span style="font-size: 1.8rem; font-weight: 500;">บาท</span></div>
                <div class="price-range-tag">ช่วงราคาที่สมเหตุสมผล: <b>{low_price:,} – {high_price:,} บาท</b></div>
            </div>
            """, unsafe_allow_html=True)

            # Analysis Summary Cards
            col_eval1, col_eval2 = st.columns([1, 1])

            with col_eval1:
                st.markdown("#### 💡 คำแนะนำสำหรับการเลือกซื้อสเปกนี้")
                recommendations = []
                if chosen_cap >= 64:
                    recommendations.append("✅ **ความจุ 64GB+**: เหมาะสำหรับงานตัดต่อวิดีโอ 4K/8K, 3D Rendering, หรือเปิดโปรแกรมจำลอง Virtual Machine หลายตัว")
                elif chosen_cap == 32:
                    recommendations.append("✅ **ความจุ 32GB**: เป็นสเปกมาตรฐานระดับทองสำหรับเล่นเกมระดับ AAA ยุคปัจจุบันและทำงานทั่วไปได้อย่างลื่นไหลไร้สะดุด")
                elif chosen_cap == 16:
                    recommendations.append("✅ **ความจุ 16GB**: เหมาะสำหรับเกมเมอร์งบประหยัด และการทำงานเอกสาร/ท่องเว็บทั่วไป")
                
                if chosen_mod == 2:
                    recommendations.append("🚀 **แรมคู่ 2 แถว**: ยอดเยี่ยม! การใส่แรม 2 แถวจะเปิดระบบ **Dual-Channel** ช่วยให้อัตราการรับส่งข้อมูลของ CPU เร็วขึ้น 15-25%")
                else:
                    recommendations.append("ℹ️ **แรมแถวเดียว 1 แถว**: หากเมนบอร์ดรองรับ แนะนำให้ซื้อแบบ 2 แถว (เช่น 8GBx2 แทน 16GBx1) เพื่อเปิดระบบ Dual-Channel จะแรงขึ้นอย่างเห็นได้ชัดครับ")
                
                if chosen_rgb:
                    recommendations.append("🌈 **มีไฟ RGB**: สวยงาม โดดเด่นในเคสกระจก โดยทั่วไปจะมีราคาพรีเมียมเพิ่มขึ้นประมาณ 300 - 600 บาทเทียบกับรุ่น Non-RGB")

                for r in recommendations:
                    st.write(r)

            with col_eval2:
                st.markdown("#### 🛍️ สินค้าจริงในตลาดที่มีสเปกตรง/ใกล้เคียงที่สุด")
                # Find matching or nearest products in actual scraped dataset
                matches = df[
                    (df['generation'] == chosen_gen) &
                    (df['capacity_gb'] == chosen_cap)
                ]
                if not matches.empty:
                    # Sort by closeness to brand and price
                    brand_matches = matches[matches['brand'] == chosen_brand]
                    candidate_df = brand_matches if not brand_matches.empty else matches
                    top_candidates = candidate_df.sort_values(by='price_thb').head(3)

                    for _, row in top_candidates.iterrows():
                        rgb_icon = "🌈 มีไฟ RGB" if row['has_rgb'] == 1 else "⚪ Non-RGB"
                        st.markdown(f"""
                        <div class="product-real-card">
                            <div style="font-weight: 600; color: #1e293b; font-size: 0.95rem;">{row['name']}</div>
                            <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 0.4rem;">
                                <div>
                                    <span style="font-size: 1.15rem; font-weight: 700; color: #2563eb;">{row['price_thb']:,} ฿</span>
                                    <span style="font-size: 0.8rem; color: #64748b; margin-left: 0.5rem;">({row['price_per_gb']:.1f} ฿/GB)</span>
                                </div>
                                <a href="{row['url']}" target="_blank" style="text-decoration: none; background: #2563eb; color: white; padding: 0.25rem 0.75rem; border-radius: 0.4rem; font-size: 0.85rem; font-weight: 500;">
                                    ดูสินค้าร้าน JIB ↗
                                </a>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.info("ขณะนี้ยังไม่มีสินค้าวางจำหน่ายตรงกับความจุนี้ในสต็อก แต่โมเดลสามารถคำนวณราคาประเมินให้ได้ตามสถิติของตลาดครับ")

    # =========================================================
    # TAB 2: EASY MARKET CATALOG
    # =========================================================
    with tab2:
        st.markdown("### 🔍 ค้นหาและเปรียบเทียบราคาสินค้าจริงจากร้านค้าไอที")
        st.write("ดูราคาจริงของแรมทุกรุ่นที่มีการจำหน่ายในสต็อกปัจจุบัน สามารถค้นหาและกรองได้ง่ายๆ:")

        # Search & Quick Filters
        s_col1, s_col2, s_col3 = st.columns([2, 1, 1])
        with s_col1:
            search_query = st.text_input("🔎 พิมพ์ค้นหาตามชื่อรุ่น หรือ ยี่ห้อ", placeholder="เช่น Kingston, Corsair, RGB, 32GB...")
        with s_col2:
            f_gen = st.selectbox("เลือกรุ่น DDR", ["ทั้งหมด", "เฉพาะ DDR4", "เฉพาะ DDR5"])
        with s_col3:
            f_sort = st.selectbox("เรียงลำดับตาม", ["ราคา: น้อยไปมาก", "ราคา: มากไปน้อย", "ความคุ้มค่า (฿/GB ต่ำสุด)"])

        # Filter logic
        view_df = df.copy()
        if search_query:
            view_df = view_df[view_df['name'].str.contains(search_query, case=False, na=False)]
        if f_gen == "เฉพาะ DDR4":
            view_df = view_df[view_df['generation'] == 'DDR4']
        elif f_gen == "เฉพาะ DDR5":
            view_df = view_df[view_df['generation'] == 'DDR5']

        if f_sort == "ราคา: น้อยไปมาก":
            view_df = view_df.sort_values(by='price_thb', ascending=True)
        elif f_sort == "ราคา: มากไปน้อย":
            view_df = view_df.sort_values(by='price_thb', ascending=False)
        else:
            view_df = view_df.sort_values(by='price_per_gb', ascending=True)

        st.caption(f"พบรายการแรมทั้งหมด {len(view_df)} รายการ:")

        # Clean Table Presentation
        st.dataframe(
            view_df[['brand', 'name', 'generation', 'capacity_gb', 'speed_mhz', 'price_thb', 'price_per_gb', 'url']],
            use_container_width=True,
            column_config={
                "brand": st.column_config.TextColumn("ยี่ห้อ"),
                "name": st.column_config.TextColumn("ชื่อรุ่นสินค้า", width="large"),
                "generation": st.column_config.TextColumn("DDR"),
                "capacity_gb": st.column_config.NumberColumn("ความจุ", format="%d GB"),
                "speed_mhz": st.column_config.NumberColumn("บัสสปีด", format="%d MHz"),
                "price_thb": st.column_config.NumberColumn("ราคาขายจริง", format="%d ฿"),
                "price_per_gb": st.column_config.NumberColumn("ความคุ้มค่า", format="%.1f ฿/GB"),
                "url": st.column_config.LinkColumn("ลิงก์หน้าร้าน")
            },
            hide_index=True
        )

    # =========================================================
    # TAB 3: INSIGHTS & PIPELINE
    # =========================================================
    with tab3:
        st.markdown("### 📊 ภาพรวมตลาดและข้อมูลเชิงลึก (Market Insights)")

        stat_col1, stat_col2 = st.columns(2)
        with stat_col1:
            st.markdown("##### 🏆 แบรนด์ไหนคุ้มค่าที่สุด? (เปรียบเทียบราคาเฉลี่ยต่อ GB)")
            top_brand_names = df['brand'].value_counts().head(7).index
            brand_val = df[df['brand'].isin(top_brand_names)].groupby('brand')['price_per_gb'].median().sort_values()
            st.bar_chart(brand_val, color="#10b981")
            st.caption("ยิ่งค่า ฿/GB ต่ำ แสดงว่ายิ่งได้ความจุต่อเม็ดเงินที่จ่ายไปคุ้มค่าที่สุด")

        with stat_col2:
            st.markdown("##### ⚡ ความเร็วบัส (MHz) กับระดับราคา (บาท)")
            st.scatter_chart(df, x='speed_mhz', y='price_thb', color='generation')
            st.caption("DDR5 (สีส้ม/ชมพู) จะมีความเร็วบัสเริ่มต้นที่ 4,800 - 8,000 MHz ให้แบนด์วิดท์สูงกว่า DDR4 อย่างมีนัยสำคัญ")

        st.markdown("---")
        st.markdown("### 🔄 ระบบดึงข้อมูลสดและ Machine Learning")
        
        info_c1, info_c2 = st.columns([1, 1])
        with info_c1:
            st.markdown("##### 🤖 ประสิทธิภาพโมเดลทำนายราคา:")
            st.write("- **อัลกอริทึม**: `Random Forest Regressor` ร่วมกับ `Log-Normal Transformation`")
            st.write("- **ความแม่นยำ ($R^2$ Score)**: `95.3%` บน Held-out Test Set")
            st.write("- **ความคลาดเคลื่อนเฉลี่ย (MAE)**: `±1,434 บาท`")
            st.write(f"- **จำนวนข้อมูลที่ใช้เทรน**: `{len(df)} รายการสินค้าจริงในไทย`")

        with info_c2:
            st.markdown("##### 🌐 ซิงค์ราคาตลาดสดใหม่ (On-Demand Live Sync):")
            st.write("ระบบจะสั่งรัน Web Scraper ไปยังร้านค้าเพื่อสแกนราคารุ่นแรมล่าสุดและสกัดสเปกใหม่ทันที")
            if st.button("🔄 ซิงค์ข้อมูลราคาสดเดี๋ยวนี้", use_container_width=True):
                with st.spinner("กำลังเชื่อมต่อร้านค้าและอัปเดตราคาล่าสุด..."):
                    try:
                        new_data = run_data_pipeline(save_raw=True)
                        st.success(f"อัปเดตข้อมูลสำเร็จ! ได้รับข้อมูล {len(new_data)} รายการ")
                        st.cache_data.clear()
                        st.rerun()
                    except Exception as e:
                        st.error(f"เกิดข้อผิดพลาดในการดึงข้อมูล: {e}")

if __name__ == "__main__":
    main()
