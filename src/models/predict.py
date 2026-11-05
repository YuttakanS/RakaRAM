import os
import argparse
import joblib
import pandas as pd
from typing import Dict, Any

MODEL_PATH = "saved_models/ram_price_model.joblib"

class RamPricePredictor:
    def __init__(self, model_path: str = MODEL_PATH):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model file not found at '{model_path}'. Please run 'python -m src.models.train' first."
            )
        self.model = joblib.load(model_path)

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
        Predict real-time estimated RAM price from hardware specifications.
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

        return {
            'estimated_price_thb': estimated_price,
            'price_range': {
                'lower_bound_thb': lower_bound,
                'upper_bound_thb': upper_bound
            },
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
    parser = argparse.ArgumentParser(description="Real-time RAM Price Predictor")
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

    print("\n" + "=" * 55)
    print("        RakaRAM - REAL-TIME PRICE PREDICTION        ")
    print("=" * 55)
    print(f"Spec Summary      : {res['inputs']['brand']} {res['inputs']['generation']} {res['inputs']['capacity_gb']}GB ({res['inputs']['modules']}x{res['inputs']['capacity_gb']//res['inputs']['modules']}GB) {res['inputs']['speed_mhz']}MHz CL{res['inputs']['cas_latency']} {'RGB' if res['inputs']['has_rgb'] else 'Non-RGB'}")
    print(f"Estimated Price   : {res['estimated_price_thb']:,} THB")
    print(f"Fair Price Range  : {res['price_range']['lower_bound_thb']:,} - {res['price_range']['upper_bound_thb']:,} THB")
    print("=" * 55 + "\n")

if __name__ == "__main__":
    main()
