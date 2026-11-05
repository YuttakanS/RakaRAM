import unittest
from src.parser.spec_extractor import (
    extract_brand,
    extract_generation,
    extract_capacity_and_modules,
    extract_speed,
    extract_cas_latency,
    extract_rgb,
    parse_ram_product
)

class TestSpecExtractor(unittest.TestCase):
    def test_brand_extraction(self):
        self.assertEqual(extract_brand("16GB DDR4 CORSAIR VENGEANCE LPX"), "Corsair")
        self.assertEqual(extract_brand("32GB DDR5 KINGSTON FURY BEAST"), "Kingston")
        self.assertEqual(extract_brand("G.SKILL RIPJAWS V 16GB"), "G.Skill")
        self.assertEqual(extract_brand("TEAMGROUP T-FORCE DELTA RGB 32GB"), "TeamGroup")

    def test_generation_extraction(self):
        self.assertEqual(extract_generation("CORSAIR DDR4 3200MHz"), "DDR4")
        self.assertEqual(extract_generation("KINGSTON DDR5 6000MHz"), "DDR5")

    def test_capacity_and_modules(self):
        cap, mods = extract_capacity_and_modules("16GB (8GBx2) DDR4 3200MHz")
        self.assertEqual(cap, 16)
        self.assertEqual(mods, 2)

        cap, mods = extract_capacity_and_modules("32GB (16GBx2) DDR5 6000MHz")
        self.assertEqual(cap, 32)
        self.assertEqual(mods, 2)

        cap, mods = extract_capacity_and_modules("16GB (16GBx1) DDR4 3200Mbps")
        self.assertEqual(cap, 16)
        self.assertEqual(mods, 1)

    def test_speed_extraction(self):
        self.assertEqual(extract_speed("DDR4 3200MHz", "DDR4"), 3200)
        self.assertEqual(extract_speed("DDR5 6000MT/s", "DDR5"), 6000)

    def test_rgb_extraction(self):
        self.assertEqual(extract_rgb("KINGSTON FURY BEAST RGB"), 1)
        self.assertEqual(extract_rgb("CORSAIR VENGEANCE LPX BLACK"), 0)

    def test_full_parse(self):
        raw = {
            "title": "16GB (8GBx2) DDR4 3200MHz RAM CORSAIR VENGEANCE LPX CL16",
            "price": 4690,
            "url": "https://example.com/item"
        }
        res = parse_ram_product(raw)
        self.assertEqual(res["brand"], "Corsair")
        self.assertEqual(res["generation"], "DDR4")
        self.assertEqual(res["capacity_gb"], 16)
        self.assertEqual(res["modules"], 2)
        self.assertEqual(res["speed_mhz"], 3200)
        self.assertEqual(res["cas_latency"], 16)
        self.assertEqual(res["has_rgb"], 0)
        self.assertEqual(res["price_thb"], 4690)

if __name__ == "__main__":
    unittest.main()
