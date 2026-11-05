"""
RakaRAM - Serverless API Entrypoint Unit Tests
==============================================
Verifies that main.py properly exports ASGI application instances
and responds to root, health, prediction, and news queries.
"""

import unittest
import main

class TestApiEntrypoint(unittest.TestCase):
    def test_top_level_exports(self):
        """Ensure standard serverless variables are exported at module level."""
        self.assertTrue(hasattr(main, "app"), "main.py must export 'app'")
        self.assertTrue(hasattr(main, "application"), "main.py must export 'application'")
        self.assertTrue(hasattr(main, "handler"), "main.py must export 'handler'")
        self.assertIsNotNone(main.app)

    def test_root_endpoint(self):
        res = main.root()
        self.assertEqual(res["status"], "online")
        self.assertEqual(res["service"], "RakaRAM API")
        self.assertIn("docs", res)

    def test_health_endpoint(self):
        res = main.health_check()
        self.assertIn("status", res)
        self.assertTrue(res["model_loaded"])

    def test_predict_endpoints(self):
        # Test GET function
        res_get = main.predict_price_get(
            brand="Kingston",
            generation="DDR5",
            capacity_gb=32,
            modules=2,
            speed_mhz=6000,
            cas_latency=36,
            has_rgb=1,
            form_factor="UDIMM"
        )
        self.assertGreater(res_get["estimated_price_thb"], 0)
        self.assertIn("forecast", res_get)

        # Test POST function
        payload = main.PredictRequest(
            brand="Corsair",
            generation="DDR4",
            capacity_gb=16,
            modules=2,
            speed_mhz=3200,
            cas_latency=16,
            has_rgb=0,
            form_factor="UDIMM"
        )
        res_post = main.predict_price_post(payload)
        self.assertGreater(res_post["estimated_price_thb"], 0)
        self.assertIn("forecast", res_post)

    def test_market_timeline_endpoint(self):
        res = main.get_market_timeline()
        self.assertIn("months", res)
        self.assertIn("ddr4", res)
        self.assertIn("ddr5", res)

if __name__ == "__main__":
    unittest.main()
