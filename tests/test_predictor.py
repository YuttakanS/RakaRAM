import unittest
from src.models.predict import RamPricePredictor

class TestRamPricePredictor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.predictor = RamPricePredictor()

    def test_prediction_output_format(self):
        res = self.predictor.predict(
            brand="Kingston",
            generation="DDR5",
            capacity_gb=32,
            modules=2,
            speed_mhz=6000,
            cas_latency=36,
            has_rgb=1
        )
        self.assertIn("estimated_price_thb", res)
        self.assertIn("price_range", res)
        self.assertIn("forecast", res)
        self.assertGreater(res["estimated_price_thb"], 1000)

    def test_forecast_trends(self):
        # DDR5 trend should decrease over 6 months
        fc_ddr5 = self.predictor.forecast_trends(
            current_price=20000,
            generation="DDR5",
            speed_mhz=6000,
            capacity_gb=32
        )
        self.assertEqual(fc_ddr5["trend_direction"], "down")
        self.assertLess(fc_ddr5["forecast_6m_thb"], 20000)
        self.assertEqual(len(fc_ddr5["timeline"]), 4)

        # DDR4 trend should stay stable or slight upward drift
        fc_ddr4 = self.predictor.forecast_trends(
            current_price=4500,
            generation="DDR4",
            speed_mhz=3200,
            capacity_gb=16
        )
        self.assertEqual(fc_ddr4["trend_direction"], "up")
        self.assertGreaterEqual(fc_ddr4["forecast_6m_thb"], 4500)

if __name__ == "__main__":
    unittest.main()
