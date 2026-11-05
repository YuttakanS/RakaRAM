import numpy as np
import pandas as pd

np.random.seed(42)
n_samples = 1200

brands = ['Corsair', 'Kingston', 'G.Skill', 'TeamGroup', 'KLEVV', 'Crucial']
generations = ['DDR4', 'DDR5']

data = []

for _ in range(n_samples):
  gen = np.random.choice(generations, p=[0.45, 0.55])
  brand = np.random.choice(brands)
  has_rgb = np.random.choice([0, 1], p=[0.4, 0.6])

  if gen == 'DDR4':
    capacity = np.random.choice([8, 16, 32, 64], p=[0.15, 0.50, 0.30, 0.05])
    speed = np.random.choice([2666, 3200, 3600, 4000], p=[0.1, 0.5, 0.35, 0.05])
    cas = int(np.random.choice([16, 18, 19, 22]))
    voltage = 1.35 if speed >= 3200 else 1.20
    base_price = capacity * 65 + (speed - 2666) * 0.4 - (cas - 16) * 25
  else:  # DDR5
    capacity = np.random.choice([16, 32, 48, 64, 96], p=[0.15, 0.55, 0.10, 0.15, 0.05])
    speed = np.random.choice(
        [4800, 5200, 5600, 6000, 6400, 7200, 7600],
        p=[0.1, 0.1, 0.2, 0.35, 0.15, 0.07, 0.03],
    )
    cas = int(np.random.choice([30, 32, 36, 38, 40]))
    voltage = 1.40 if speed >= 6400 else (1.35 if speed >= 6000 else 1.25)
    base_price = (
        capacity * 105
        + (speed - 4800) * 0.75
        - (cas - 30) * 45
        + (500 if gen == 'DDR5' else 0)
    )

  modules = 2 if capacity in [16, 32, 64] and np.random.rand() > 0.3 else 1
  if has_rgb:
    base_price += 350

  brand_multiplier = {
      'Corsair': 1.12,
      'G.Skill': 1.10,
      'Kingston': 1.00,
      'TeamGroup': 0.95,
      'KLEVV': 0.93,
      'Crucial': 0.90,
  }
  noise = np.random.normal(0, 120)
  final_price = max(
      500, round((base_price * brand_multiplier[brand] + noise) / 10) * 10
  )

  name = f'{brand} {gen} {capacity}GB ({modules}x{capacity//modules}GB) {speed}MHz CL{cas}'
  if has_rgb:
    name += ' RGB'

  data.append({
      'name': name,
      'brand': brand,
      'generation': gen,
      'capacity_gb': capacity,
      'modules': modules,
      'speed_mhz': speed,
      'cas_latency': cas,
      'has_rgb': has_rgb,
      'voltage': round(voltage, 2),
      'price_thb': int(final_price),
  })

df = pd.DataFrame(data)
df.to_csv('ram_dataset.csv', index=False)
print(f'สร้าง Dataset สำเร็จ จำนวน {len(df)} แถว บันทึกลง ram_dataset.csv')