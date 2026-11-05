import os
import argparse
import joblib
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
DEFAULT_MODEL_PATH = str(ROOT_DIR / "saved_models" / "ram_price_model.joblib")

class RamPricePredictor:
    def __init__(self, model_path: str = DEFAULT_MODEL_PATH):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model file not found at '{model_path}'. Please run 'python -m src.models.train' first."
            )
        self.model = joblib.load(model_path)

    def forecast_trends(
        self,
        current_price: int,
        generation: str,
        speed_mhz: int,
        capacity_gb: int
    ) -> Dict[str, Any]:
        """
        Forecasting future price movements based on semiconductor market lifecycle:
        - DDR4: Mature/Legacy tech. Fab supply shrinking; prices stabilized or slightly rising (+0.5% to +3%).
        - DDR5: Expanding adoption. Fab yields improving; prices dropping steadily (-3% to -10%).
        """
        if generation == "DDR4":
            # DDR4 legacy trend: bottomed out, slight upward drift over 6 months
            rate_1m = 1.004   # +0.4%
            rate_3m = 1.015   # +1.5%
            rate_6m = 1.032   # +3.2%
            trend_dir = "up"
            rec_th = "ซื้อได้ตามความต้องการ (ราคาอยู่ในระดับต่ำสุดตามวงจรอายุตลาด อนาคตมีแนวโน้มขยับขึ้นเล็กน้อยตามการลดกำลังผลิตของโรงงานชิป)"
            rec_en = "Optimal purchase timing (Prices have reached floor level; foundry capacity is transitioning to DDR5)"
            rec_badge_th = "ซื้อได้เลย ไม่ต้องรอ"
            rec_badge_en = "Buy Now"
            rec_type = "buy_now"
        else:  # DDR5
            if speed_mhz >= 6400 or capacity_gb >= 64:
                # High-end DDR5: faster premium decay
                rate_1m = 0.980   # -2.0%
                rate_3m = 0.945   # -5.5%
                rate_6m = 0.900   # -10.0%
                trend_dir = "down"
                rec_th = "รอได้หากไม่รีบใช้งาน (สเปกความเร็วสูงมีแนวโน้มราคาจะปรับลดลงอีก 5–10% ในช่วง 3–6 เดือนข้างหน้า)"
                rec_en = "Wait if not urgent (High-spec memory premiums are projected to drop 5–10% over the next 3–6 months)"
                rec_badge_th = "รอได้ถ้าไม่รีบ"
                rec_badge_en = "Wait if Possible"
                rec_type = "wait"
            else:
                # Mainstream DDR5: steady moderate decline
                rate_1m = 0.985   # -1.5%
                rate_3m = 0.960   # -4.0%
                rate_6m = 0.930   # -7.0%
                trend_dir = "down"
                rec_th = "จังหวะเหมาะสมในการจัดซื้อหรืออัปเกรด (ราคาทยอยปรับตัวลงอย่างมีเสถียรภาพ เหมาะสำหรับเริ่มประกอบเครื่อง)"
                rec_en = "Favorable purchase timing (Mainstream DDR5 retail prices are steadily normalizing)"
                rec_badge_th = "จังหวะดีน่าซื้อ"
                rec_badge_en = "Good Time to Buy"
                rec_type = "good_time"

        p_1m = max(400, round((current_price * rate_1m) / 10) * 10)
        p_3m = max(400, round((current_price * rate_3m) / 10) * 10)
        p_6m = max(400, round((current_price * rate_6m) / 10) * 10)
        pct_6m = round(((p_6m - current_price) / current_price) * 100, 1)

        timeline = [
            {"period_th": "ปัจจุบัน", "period_en": "Current", "price": current_price, "month": 0},
            {"period_th": "อีก 1 เดือน", "period_en": "In 1 Month", "price": p_1m, "month": 1},
            {"period_th": "อีก 3 เดือน", "period_en": "In 3 Months", "price": p_3m, "month": 3},
            {"period_th": "อีก 6 เดือน", "period_en": "In 6 Months", "price": p_6m, "month": 6},
        ]

        return {
            "forecast_1m_thb": p_1m,
            "forecast_3m_thb": p_3m,
            "forecast_6m_thb": p_6m,
            "trend_direction": trend_dir,
            "percent_change_6m": pct_6m,
            "recommendation_th": rec_th,
            "recommendation_en": rec_en,
            "recommendation_badge_th": rec_badge_th,
            "recommendation_badge_en": rec_badge_en,
            "recommendation_type": rec_type,
            "timeline": timeline
        }

    def predict(
        self,
        brand: str,
        generation: str,
        capacity_gb: int,
        modules: int,
        speed_mhz: int,
        cas_latency: int,
        has_rgb: int,
        form_factor: str = "UDIMM"
    ) -> Dict[str, Any]:
        """
        Predict real-time estimated RAM price and calculate future forecasting trends.
        """
        input_data = pd.DataFrame([{
            'brand': brand,
            'generation': generation,
            'form_factor': form_factor,
            'capacity_gb': capacity_gb,
            'modules': modules,
            'speed_mhz': speed_mhz,
            'cas_latency': cas_latency,
            'has_rgb': has_rgb
        }])

        pred_price = float(self.model.predict(input_data)[0])
        # Round to nearest 10 or 50 THB like actual retailer pricing
        estimated_price = max(400, round(pred_price / 10) * 10)

        # Confidence interval approximation (~8% margin)
        margin = max(300, round(estimated_price * 0.08 / 10) * 10)
        lower_bound = max(400, estimated_price - margin)
        upper_bound = estimated_price + margin

        # Calculate future forecasting
        forecast = self.forecast_trends(
            current_price=estimated_price,
            generation=generation,
            speed_mhz=speed_mhz,
            capacity_gb=capacity_gb
        )

        return {
            'estimated_price_thb': estimated_price,
            'price_range': {
                'lower_bound_thb': lower_bound,
                'upper_bound_thb': upper_bound
            },
            'forecast': forecast,
            'inputs': {
                'brand': brand,
                'generation': generation,
                'form_factor': form_factor,
                'capacity_gb': capacity_gb,
                'modules': modules,
                'speed_mhz': speed_mhz,
                'cas_latency': cas_latency,
                'has_rgb': bool(has_rgb)
            }
        }

def main():
    parser = argparse.ArgumentParser(description="Real-time RAM Price Predictor & Future Forecasting")
    parser.add_argument("--brand", type=str, default="Kingston", help="Brand name (e.g. Kingston, Corsair, G.Skill)")
    parser.add_argument("--gen", type=str, default="DDR5", choices=["DDR4", "DDR5"], help="DDR Generation")
    parser.add_argument("--capacity", type=int, default=32, help="Total capacity in GB (e.g. 16, 32, 64)")
    parser.add_argument("--modules", type=int, default=2, choices=[1, 2, 4], help="Number of modules/sticks")
    parser.add_argument("--speed", type=int, default=6000, help="Bus speed in MHz (e.g. 3200, 6000)")
    parser.add_argument("--cl", type=int, default=36, help="CAS Latency (CL)")
    parser.add_argument("--rgb", type=int, default=1, choices=[0, 1], help="Has RGB Lighting (1=Yes, 0=No)")
    parser.add_argument("--form", type=str, default="UDIMM", choices=["UDIMM", "SO-DIMM"], help="Form Factor")

    args = parser.parse_args()

    predictor = RamPricePredictor()
    res = predictor.predict(
        brand=args.brand,
        generation=args.gen,
        capacity_gb=args.capacity,
        modules=args.modules,
        speed_mhz=args.speed,
        cas_latency=args.cl,
        has_rgb=args.rgb,
        form_factor=args.form
    )

    fc = res['forecast']
    print("\n" + "=" * 60)
    print("      RakaRAM - REAL-TIME PRICE & FUTURE FORECASTING        ")
    print("=" * 60)
    print(f"Spec Summary      : {res['inputs']['brand']} {res['inputs']['generation']} {res['inputs']['capacity_gb']}GB ({res['inputs']['modules']}x{res['inputs']['capacity_gb']//res['inputs']['modules']}GB) {res['inputs']['speed_mhz']}MHz CL{res['inputs']['cas_latency']} {'RGB' if res['inputs']['has_rgb'] else 'Non-RGB'}")
    print(f"Current Estimate  : {res['estimated_price_thb']:,} THB (Fair Range: {res['price_range']['lower_bound_thb']:,} - {res['price_range']['upper_bound_thb']:,} THB)")
    print("-" * 60)
    print("🔮 Future Price Projections:")
    print(f"   • In 1 Month   : {fc['forecast_1m_thb']:,} THB")
    print(f"   • In 3 Months  : {fc['forecast_3m_thb']:,} THB")
    print(f"   • In 6 Months  : {fc['forecast_6m_thb']:,} THB (Trend: {fc['percent_change_6m']:+0.1f}%)")
    print(f"💡 Recommendation : {fc['recommendation_th']}")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    main()
