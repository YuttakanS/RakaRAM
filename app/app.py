import streamlit as st
import pandas as pd
import numpy as np
import os
import json
from datetime import datetime
from src.models.predict import RamPricePredictor
from src.scraper.run_pipeline import run_data_pipeline

# Page Configuration
st.set_page_config(
    page_title="RakaRAM | ระบบวิเคราะห์และทำนายราคาแรมคอมพิวเตอร์",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.3rem;
        font-weight: 800;
        color: #1e293b;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #64748b;
        margin-bottom: 1.5rem;
    }
    .price-card {
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
        padding: 1.8rem;
        border-radius: 1rem;
        color: white;
        text-align: center;
        margin-bottom: 1rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .price-value {
        font-size: 3rem;
        font-weight: 800;
        margin: 0.3rem 0;
    }
    .price-range {
        font-size: 1.1rem;
        opacity: 0.9;
    }
    .stat-badge {
        padding: 0.2rem 0.6rem;
        border-radius: 0.4rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

DATASET_PATH = "data/processed/ram_dataset_real.csv"
METADATA_PATH = "saved_models/model_metadata.json"

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
        st.error(f"Error loading prediction model: {e}")
        return None

def main():
    st.markdown('<div class="main-title">⚡ RakaRAM (ราคาแรม)</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">ระบบวิเคราะห์ตลาดและทำนายราคาแรมคอมพิวเตอร์จากราคาตลาดจริงในไทยแบบ Real-time</div>',
        unsafe_allow_html=True
    )

    df = load_dataset()
    predictor = load_predictor()

    if df.empty:
        st.warning("⚠️ ไม่พบชุดข้อมูลราคาจริง กรุณากดปุ่ม Sync ข้อมูลในแท็บ Live Data Ingestion")
        return

    # Top Metrics Bar
    ddr4_df = df[df['generation'] == 'DDR4']
    ddr5_df = df[df['generation'] == 'DDR5']

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("📦 รายการแรมในระบบ", f"{len(df):,} รุ่น", f"{df['brand'].nunique()} แบรนด์")
    with m2:
        st.metric("⚡ ราคาเฉลี่ย DDR4", f"{ddr4_df['price_thb'].median():,.0f} ฿", f"{ddr4_df['price_per_gb'].median():.1f} ฿/GB")
    with m3:
        st.metric("🚀 ราคาเฉลี่ย DDR5", f"{ddr5_df['price_thb'].median():,.0f} ฿", f"{ddr5_df['price_per_gb'].median():.1f} ฿/GB")
    with m4:
        st.metric("🎯 ความแม่นยำโมเดล (R²)", "95.3%", "MAE: ±1,434 ฿")

    st.write("")

    # Main Tabs
    tab1, tab2, tab3 = st.tabs([
        "🔮 Real-time Price Predictor",
        "📊 Live Market Analytics",
        "⚙️ Pipeline & Model Info"
    ])

    # ---------------- TAB 1: PREDICTOR ----------------
    with tab1:
        st.subheader("💡 คำนวณราคาประเมินจากสเปกแรม")
        st.write("เลือกสเปกแรมที่คุณต้องการ ระบบ Machine Learning จะวิเคราะห์ราคาขายที่เหมาะสมในตลาดไทยแบบ Real-time")

        c1, c2, c3 = st.columns([1, 1, 1])

        with c1:
            brand_list = sorted(df['brand'].unique().tolist())
            brand = st.selectbox("ยี่ห้อ (Brand)", brand_list, index=brand_list.index("Kingston") if "Kingston" in brand_list else 0)
            generation = st.radio("รุ่นหน่วยความจำ (Generation)", ["DDR4", "DDR5"], horizontal=True)
            form_factor = st.selectbox("ประเภทการใช้งาน (Form Factor)", ["UDIMM (สำหรับ Desktop PC)", "SO-DIMM (สำหรับ Laptop/Mini-PC)"])
            form_val = "SO-DIMM" if "SO-DIMM" in form_factor else "UDIMM"

        with c2:
            capacities = [8, 16, 24, 32, 48, 64, 96, 128] if generation == "DDR5" else [4, 8, 16, 32, 64, 128]
            default_cap = 32 if 32 in capacities else capacities[1]
            capacity = st.selectbox("ความจุรวม (Total Capacity)", capacities, index=capacities.index(default_cap), format_func=lambda x: f"{x} GB")
            modules = st.selectbox("จำนวนแถวแรม (Modules)", [1, 2, 4], index=1, format_func=lambda x: f"{x} แถว ({capacity//x}GB x {x})")

        with c3:
            speeds = [4800, 5200, 5600, 6000, 6400, 6800, 7200, 7600, 8000] if generation == "DDR5" else [2400, 2666, 3000, 3200, 3600, 4000]
            default_spd = 6000 if 6000 in speeds else 3200
            speed = st.selectbox("ความเร็วบัส (Speed)", speeds, index=speeds.index(default_spd) if default_spd in speeds else 0, format_func=lambda x: f"{x} MHz")
            
            cl_options = [28, 30, 32, 34, 36, 38, 40, 42] if generation == "DDR5" else [14, 15, 16, 17, 18, 19, 22]
            default_cl = 36 if generation == "DDR5" else 16
            cas = st.selectbox("CAS Latency (CL)", cl_options, index=cl_options.index(default_cl) if default_cl in cl_options else 0, format_func=lambda x: f"CL {x}")
            
            has_rgb = st.checkbox("🌈 มีไฟ RGB (Lighting)", value=True if generation == "DDR5" else False)

        st.write("")
        predict_btn = st.button("🚀 คำนวณราคาประเมิน (Predict Price)", type="primary", use_container_width=True)

        if predict_btn or 'last_pred' not in st.session_state:
            if predictor:
                res = predictor.predict(
                    brand=brand,
                    generation=generation,
                    capacity_gb=capacity,
                    modules=modules,
                    speed_mhz=speed,
                    cas_latency=cas,
                    has_rgb=1 if has_rgb else 0,
                    form_factor=form_val
                )
                st.session_state['last_pred'] = res

        if 'last_pred' in st.session_state:
            pred = st.session_state['last_pred']
            est = pred['estimated_price_thb']
            low = pred['price_range']['lower_bound_thb']
            high = pred['price_range']['upper_bound_thb']

            st.markdown(f"""
            <div class="price-card">
                <div>ราคาประเมินที่เหมาะสมในตลาดปัจจุบัน</div>
                <div class="price-value">{est:,} บาท</div>
                <div class="price-range">ช่วงราคาที่แนะนำ (Fair Price Range): <b>{low:,} - {high:,} บาท</b></div>
            </div>
            """, unsafe_allow_html=True)

            # Contextual Comparison
            similar_df = df[
                (df['generation'] == generation) &
                (df['capacity_gb'] == capacity)
            ]
            if not similar_df.empty:
                avg_similar = similar_df['price_thb'].median()
                diff = est - avg_similar
                diff_text = f"{'+' if diff >= 0 else ''}{diff:,.0f} บาท ({diff/avg_similar*100:+.1f}%)"
                st.info(
                    f"💡 แรม {generation} ขนาด {capacity}GB ในตลาดมีราคาค่ากลางอยู่ที่ **{avg_similar:,.0f} บาท** "
                    f"(สเปกที่คุณเลือกมีราคาต่างจากค่ากลาง {diff_text})"
                )

    # ---------------- TAB 2: ANALYTICS ----------------
    with tab2:
        st.subheader("📈 วิเคราะห์และเปรียบเทียบตลาดแรมในไทย")

        col_f1, col_f2, col_f3 = st.columns(3)
        with col_f1:
            gen_filter = st.multiselect("กรองตาม Generation", ["DDR4", "DDR5"], default=["DDR4", "DDR5"])
        with col_f2:
            brand_filter = st.multiselect("กรองตามแบรนด์", sorted(df['brand'].unique()), default=df['brand'].value_counts().head(6).index.tolist())
        with col_f3:
            rgb_filter = st.selectbox("ไฟ RGB", ["ทั้งหมด", "เฉพาะที่มี RGB", "เฉพาะที่ไม่มี RGB"])

        filtered_df = df[df['generation'].isin(gen_filter) & df['brand'].isin(brand_filter)].copy()
        if rgb_filter == "เฉพาะที่มี RGB":
            filtered_df = filtered_df[filtered_df['has_rgb'] == 1]
        elif rgb_filter == "เฉพาะที่ไม่มี RGB":
            filtered_df = filtered_df[filtered_df['has_rgb'] == 0]

        st.write(f"พบข้อมูลสินค้าตามเงื่อนไข: **{len(filtered_df)} รายการ**")

        chart_c1, chart_c2 = st.columns(2)
        with chart_c1:
            st.markdown("##### ความคุ้มค่าต่อ GB (Price per GB) แยกตามแบรนด์")
            brand_val = filtered_df.groupby('brand')['price_per_gb'].median().reset_index()
            brand_val.columns = ['Brand', 'THB_per_GB']
            st.bar_chart(brand_val.set_index('Brand'), color='#3b82f6')

        with chart_c2:
            st.markdown("##### ความเร็วบัส (MHz) กับราคาขาย (บาท)")
            st.scatter_chart(
                filtered_df,
                x='speed_mhz',
                y='price_thb',
                color='generation',
                size='capacity_gb'
            )

        st.markdown("##### 🔍 คลังรายการแรมและราคาสินค้าจริง")
        display_cols = ['name', 'brand', 'generation', 'capacity_gb', 'speed_mhz', 'cas_latency', 'price_thb', 'url']
        st.dataframe(
            filtered_df[display_cols].sort_values(by='price_thb'),
            use_container_width=True,
            column_config={
                "url": st.column_config.LinkColumn("หน้าร้านค้า JIB"),
                "price_thb": st.column_config.NumberColumn("ราคา (บาท)", format="%d ฿")
            }
        )

    # ---------------- TAB 3: PIPELINE & MODEL ----------------
    with tab3:
        st.subheader("⚙️ ระบบ Data Pipeline และโมเดล Machine Learning")

        meta = {}
        if os.path.exists(METADATA_PATH):
            with open(METADATA_PATH, "r", encoding="utf-8") as f:
                meta = json.load(f)

        p_col1, p_col2 = st.columns([1, 1])

        with p_col1:
            st.markdown("##### 🤖 ข้อมูลโมเดลปัจจุบัน")
            st.write(f"- **Algorithm**: `{meta.get('model_name', 'Random Forest')}`")
            st.write(f"- **Target Transform**: `Log-Normal (np.log1p / np.expm1)`")
            st.write(f"- **จำนวนข้อมูลที่ใช้เทรน**: `{meta.get('total_training_samples', len(df))} รายการ`")
            st.write(f"- **อัปเดตโมเดลล่าสุดเมื่อ**: `{meta.get('saved_at', 'N/A')}`")

            metrics = meta.get('metrics', {}).get('test_set', {})
            st.markdown("##### 🎯 ผลการทดสอบ (Held-out Test Set):")
            st.write(f"- **R² Score**: `{metrics.get('r2', 0.9533):.4f}` (อธิบายความผันแปรของราคาได้กว่า 95%)")
            st.write(f"- **MAE (Mean Absolute Error)**: `±{metrics.get('mae', 1434.1):,.1f} บาท`")
            st.write(f"- **RMSE**: `{metrics.get('rmse', 3446.8):,.1f} บาท`")

        with p_col2:
            st.markdown("##### 🔄 ซิงค์ข้อมูลราคาตลาดแบบสด (On-Demand Live Sync)")
            st.write("กดปุ่มด้านล่างเพื่อรัน Web Scraper ไปดึงราคารุ่นแรมล่าสุดจาก JIB และสกัดสเปกใหม่ทันที")

            if st.button("🌐 เริ่มต้นซิงค์ข้อมูลราคาตลาดเดี๋ยวนี้", type="secondary"):
                with st.spinner("กำลังดึงข้อมูลราคาแรมสดจากร้านค้า..."):
                    try:
                        new_df = run_data_pipeline(save_raw=True)
                        st.success(f"ซิงค์ข้อมูลสำเร็จ! ได้รับข้อมูล {len(new_df)} รายการ")
                        st.cache_data.clear()
                        st.rerun()
                    except Exception as e:
                        st.error(f"เกิดข้อผิดพลาดในการซิงค์: {e}")

if __name__ == "__main__":
    main()
