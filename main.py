"""
RakaRAM - Serverless API & Mobile Web Dashboard Entrypoint
==========================================================
FastAPI application providing:
1. Responsive Mobile Web Dashboard (HTML/CSS/JS) for smartphone & desktop browsers
2. High-performance RESTful API for RAM price prediction & market intelligence
3. Seamless zero-config deployment on Vercel, AWS Lambda, or local ASGI uvicorn.

The top-level ASGI framework instance is exposed globally as `app`, `application`, and `handler`.
"""

from datetime import datetime, timezone
from typing import Optional, Dict, Any
from fastapi import FastAPI, Query, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Core domain modules
from src.models.predict import RamPricePredictor
from src.news.news_analyzer import get_latest_news_analysis, analyze_news_articles

# ---------------------------------------------------------------------------
# Global Application Instance (Required for Vercel / Serverless ASGI handlers)
# ---------------------------------------------------------------------------
app = FastAPI(
    title="RakaRAM API & Web Dashboard",
    description="Real-time Computer RAM Price Predictor & Global Semiconductor Market Intelligence",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Standard Serverless WSGI/ASGI Aliases (Vercel, AWS Lambda, etc.)
application = app
handler = app

# Enable Cross-Origin Resource Sharing (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lazy singleton model loader
_predictor_instance: Optional[RamPricePredictor] = None

def get_predictor() -> RamPricePredictor:
    global _predictor_instance
    if _predictor_instance is None:
        try:
            _predictor_instance = RamPricePredictor()
        except Exception as e:
            raise RuntimeError(f"Could not load ML price model: {e}")
    return _predictor_instance


# ---------------------------------------------------------------------------
# Pydantic Schemas
# ---------------------------------------------------------------------------
class PredictRequest(BaseModel):
    brand: str = Field(default="Kingston", description="RAM Brand (e.g. Kingston, Corsair, G.Skill)")
    generation: str = Field(default="DDR5", description="Memory generation: 'DDR4' or 'DDR5'")
    capacity_gb: int = Field(default=32, description="Total capacity in GB (e.g. 8, 16, 32, 64)")
    modules: int = Field(default=2, description="Number of modules/sticks (1 or 2)")
    speed_mhz: int = Field(default=6000, description="Memory clock speed in MHz (e.g. 3200, 5600, 6000)")
    cas_latency: int = Field(default=36, description="CAS latency timing (e.g. 16, 30, 36, 40)")
    has_rgb: int = Field(default=1, description="RGB lighting indicator: 1 for Yes, 0 for No")
    form_factor: str = Field(default="UDIMM", description="Form factor: 'UDIMM' (Desktop) or 'SO-DIMM' (Laptop)")


class PredictResponse(BaseModel):
    estimated_price_thb: int
    price_range: Dict[str, int]
    forecast: Dict[str, Any]
    inputs: Dict[str, Any]


# ---------------------------------------------------------------------------
# Mobile-First Web Dashboard HTML Template
# ---------------------------------------------------------------------------
MOBILE_WEB_DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="th">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
  <title>RakaRAM - ประเมินราคาแรม & เรดาร์ตลาดชิปโลก</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Prompt:wght@300;400;500;600;700&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg: #090D16;
      --card: #111726;
      --card-border: #1E293B;
      --pcb: #0B5D46;
      --pcb-light: #10B981;
      --gold: #D97706;
      --gold-light: #FBBF24;
      --ink: #F8FAFC;
      --muted: #94A3B8;
      --pill-bg: #1A2234;
      --pill-active: #0B5D46;
      --down: #10B981;
      --up: #EF4444;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; font-family: 'Prompt', -apple-system, sans-serif; }
    body { background-color: var(--bg); color: var(--ink); padding-bottom: 50px; line-height: 1.5; -webkit-tap-highlight-color: transparent; }
    .container { max-width: 680px; margin: 0 auto; padding: 16px; }

    /* Header */
    header { display: flex; align-items: center; justify-content: space-between; padding: 12px 0 16px; border-bottom: 1px solid var(--card-border); margin-bottom: 18px; }
    .logo { display: flex; align-items: center; gap: 8px; font-weight: 700; font-size: 20px; color: var(--ink); }
    .logo-icon { background: linear-gradient(135deg, #10B981, #0B5D46); width: 34px; height: 34px; border-radius: 8px; display: flex; align-items: center; justify-content: center; font-size: 18px; }
    .badge-accuracy { font-size: 11px; font-weight: 600; background: rgba(16, 185, 129, 0.15); color: var(--pcb-light); border: 1px solid rgba(16, 185, 129, 0.3); border-radius: 20px; padding: 3px 10px; }

    /* RAM PCB Result Card */
    .ram-stick { background: linear-gradient(180deg, #094735 0%, #06382A 100%); border: 1px solid #13775A; border-radius: 12px; padding: 18px 16px 14px; position: relative; box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.5), inset 0 1px 0 rgba(255, 255, 255, 0.1); margin-bottom: 20px; overflow: hidden; }
    .ram-header { display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 10px; }
    .ram-tag { font-size: 11px; text-transform: uppercase; letter-spacing: 0.5px; font-weight: 600; background: rgba(0,0,0,0.3); padding: 2px 8px; border-radius: 6px; color: #A7F3D0; }
    .rec-badge { font-size: 12px; font-weight: 600; padding: 3px 10px; border-radius: 20px; background: #FEF3C7; color: #92400E; display: inline-flex; align-items: center; gap: 4px; }
    .rec-badge.good_time, .rec-badge.buy_now { background: #D1FAE5; color: #065F46; }
    .rec-badge.wait { background: #FEF3C7; color: #92400E; }
    
    .price-main { font-size: 38px; font-weight: 700; color: #FFFFFF; line-height: 1.1; margin: 6px 0 4px; display: flex; align-items: baseline; gap: 6px; }
    .price-main span { font-size: 18px; font-weight: 500; color: #A7F3D0; }
    .price-range { font-size: 12px; color: #D1FAE5; opacity: 0.85; margin-bottom: 14px; }
    
    /* 6M Mini Forecast Cards */
    .fc-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; margin-top: 10px; padding-top: 12px; border-top: 1px solid rgba(255,255,255,0.1); }
    .fc-box { background: rgba(0, 0, 0, 0.25); border-radius: 8px; padding: 8px 6px; text-align: center; }
    .fc-box-label { font-size: 10px; color: #A7F3D0; margin-bottom: 2px; }
    .fc-box-val { font-size: 13px; font-weight: 600; color: #FFF; }
    .fc-box-diff { font-size: 10px; font-weight: 600; }
    .fc-box-diff.down { color: #6EE7B7; }
    .fc-box-diff.up { color: #FCA5A5; }

    /* Gold contact pins decorative */
    .gold-contacts { display: flex; justify-content: space-between; gap: 3px; height: 7px; margin: 12px -16px -14px -16px; padding: 0 12px; }
    .gold-pin { flex: 1; background: linear-gradient(180deg, #F59E0B, #B45309); border-radius: 1px 1px 0 0; }

    /* Form Section */
    .card { background: var(--card); border: 1px solid var(--card-border); border-radius: 12px; padding: 18px 16px; margin-bottom: 16px; }
    .card-title { font-size: 14px; font-weight: 600; color: var(--ink); margin-bottom: 12px; display: flex; justify-content: space-between; align-items: center; }
    
    .field-group { margin-bottom: 14px; }
    .field-label { font-size: 12px; font-weight: 500; color: var(--muted); margin-bottom: 6px; display: block; }
    
    /* Segmented Controls & Pills */
    .pills { display: flex; gap: 6px; flex-wrap: wrap; }
    .pill { flex: 1; min-width: 60px; text-align: center; padding: 8px 10px; font-size: 12px; font-weight: 600; background: var(--pill-bg); border: 1px solid var(--card-border); border-radius: 8px; color: var(--muted); cursor: pointer; transition: all 0.15s; user-select: none; }
    .pill.active { background: var(--pcb); border-color: var(--pcb-light); color: #FFF; box-shadow: 0 0 10px rgba(16, 185, 129, 0.25); }

    .grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
    select, input { width: 100%; background: var(--pill-bg); border: 1px solid var(--card-border); border-radius: 8px; color: var(--ink); padding: 9px 12px; font-size: 13px; font-weight: 500; outline: none; }
    select:focus, input:focus { border-color: var(--pcb-light); }

    /* News Radar Section */
    .news-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px; }
    .news-radar-tag { font-size: 11px; font-weight: 600; padding: 3px 8px; border-radius: 12px; }
    .news-radar-tag.bullish { background: rgba(239, 68, 68, 0.15); color: #F87171; border: 1px solid rgba(239, 68, 68, 0.3); }
    .news-radar-tag.bearish { background: rgba(16, 185, 129, 0.15); color: #34D399; border: 1px solid rgba(16, 185, 129, 0.3); }
    
    .news-item { background: rgba(255, 255, 255, 0.02); border: 1px solid var(--card-border); border-radius: 8px; padding: 10px 12px; margin-bottom: 8px; }
    .news-item-title { font-size: 12px; font-weight: 600; color: var(--ink); margin-bottom: 4px; line-height: 1.4; }
    .news-item-sub { font-size: 11px; color: var(--muted); line-height: 1.4; }

    footer { text-align: center; font-size: 11px; color: var(--muted); margin-top: 24px; padding-top: 14px; border-top: 1px solid var(--card-border); }
    footer a { color: var(--pcb-light); text-decoration: none; }
  </style>
</head>
<body>
  <div class="container">
    <header>
      <div class="logo">
        <div class="logo-icon">⚡</div>
        <span>RakaRAM</span>
      </div>
      <div class="badge-accuracy">ความแม่นยำ R² 95.3%</div>
    </header>

    <!-- RAM Stick Result Card -->
    <div class="ram-stick" id="resultCard">
      <div class="ram-header">
        <span class="ram-tag" id="dispSpecSummary">Kingston DDR5 32GB 6000MHz</span>
        <span class="rec-badge good_time" id="dispRecBadge">🟢 ซื้อได้เลย</span>
      </div>
      
      <div class="price-main">
        <span style="font-size: 26px;">฿</span><span id="dispPrice">--</span>
      </div>
      <div class="price-range" id="dispRange">ช่วงราคาที่เหมาะสม: -- ถึง -- บาท</div>

      <div class="fc-grid">
        <div class="fc-box">
          <div class="fc-box-label">อีก 1 เดือน</div>
          <div class="fc-box-val" id="fc1Val">--</div>
          <div class="fc-box-diff down" id="fc1Diff">--</div>
        </div>
        <div class="fc-box">
          <div class="fc-box-label">อีก 3 เดือน</div>
          <div class="fc-box-val" id="fc3Val">--</div>
          <div class="fc-box-diff down" id="fc3Diff">--</div>
        </div>
        <div class="fc-box">
          <div class="fc-box-label">อีก 6 เดือน</div>
          <div class="fc-box-val" id="fc6Val">--</div>
          <div class="fc-box-diff down" id="fc6Diff">--</div>
        </div>
      </div>

      <div class="gold-contacts">
        <div class="gold-pin"></div><div class="gold-pin"></div><div class="gold-pin"></div><div class="gold-pin"></div>
        <div class="gold-pin"></div><div class="gold-pin"></div><div class="gold-pin"></div><div class="gold-pin"></div>
        <div class="gold-pin"></div><div class="gold-pin"></div><div class="gold-pin"></div><div class="gold-pin"></div>
        <div class="gold-pin"></div><div class="gold-pin"></div><div class="gold-pin"></div><div class="gold-pin"></div>
      </div>
    </div>

    <!-- Specs Input Form -->
    <div class="card">
      <div class="card-title">
        <span>⚙️ กำหนดสเปกที่ต้องการ</span>
        <span style="font-size: 11px; font-weight: normal; color: var(--muted);">คำนวณแบบสดทันที</span>
      </div>

      <!-- Generation -->
      <div class="field-group">
        <label class="field-label">เทคโนโลยีแรม (Generation)</label>
        <div class="pills" id="genPills">
          <div class="pill active" data-val="DDR5" onclick="selectPill('gen', 'DDR5', this)">DDR5 (กระแสหลัก)</div>
          <div class="pill" data-val="DDR4" onclick="selectPill('gen', 'DDR4', this)">DDR4 (รุ่นเก่า)</div>
        </div>
      </div>

      <!-- Capacity -->
      <div class="field-group">
        <label class="field-label">ความจุรวม (Capacity)</label>
        <div class="pills" id="capPills">
          <div class="pill" data-val="8" onclick="selectPill('cap', 8, this)">8 GB</div>
          <div class="pill" data-val="16" onclick="selectPill('cap', 16, this)">16 GB</div>
          <div class="pill active" data-val="32" onclick="selectPill('cap', 32, this)">32 GB</div>
          <div class="pill" data-val="64" onclick="selectPill('cap', 64, this)">64 GB</div>
        </div>
      </div>

      <!-- Brand & Device Type -->
      <div class="grid-2 field-group">
        <div>
          <label class="field-label">ยี่ห้อ (Brand)</label>
          <select id="selBrand" onchange="updateCalc()">
            <option value="Kingston" selected>Kingston</option>
            <option value="Corsair">Corsair</option>
            <option value="G.Skill">G.Skill</option>
            <option value="TeamGroup">TeamGroup</option>
            <option value="KLEVV">KLEVV</option>
            <option value="Crucial">Crucial</option>
            <option value="ADATA">ADATA</option>
          </select>
        </div>
        <div>
          <label class="field-label">ประเภทอุปกรณ์</label>
          <select id="selForm" onchange="updateCalc()">
            <option value="UDIMM" selected>PC (UDIMM)</option>
            <option value="SO-DIMM">โน้ตบุ๊ก (SO-DIMM)</option>
          </select>
        </div>
      </div>

      <!-- Speed & CAS Latency -->
      <div class="grid-2 field-group">
        <div>
          <label class="field-label">ความเร็วบัส (MHz)</label>
          <select id="selSpeed" onchange="updateCalc()">
            <!-- Options dynamically populated by JS based on Gen -->
          </select>
        </div>
        <div>
          <label class="field-label">ค่าความหน่วง (CL)</label>
          <select id="selCl" onchange="updateCalc()">
            <!-- Options dynamically populated by JS based on Gen -->
          </select>
        </div>
      </div>

      <!-- Modules & RGB -->
      <div class="grid-2">
        <div>
          <label class="field-label">จำนวนแถว (Sticks)</label>
          <select id="selMod" onchange="updateCalc()">
            <option value="2" selected>2 แถว (Kit of 2)</option>
            <option value="1">1 แถวเดี่ยว</option>
          </select>
        </div>
        <div>
          <label class="field-label">ไฟประดับ RGB</label>
          <select id="selRgb" onchange="updateCalc()">
            <option value="1" selected>มีไฟ RGB (+350฿)</option>
            <option value="0">ไม่มีไฟ (Non-RGB)</option>
          </select>
        </div>
      </div>
    </div>

    <!-- Global DRAM News Radar Widget -->
    <div class="card">
      <div class="news-header">
        <span class="card-title" style="margin-bottom:0;">📡 เรดาร์ข่าวตลาดชิปโลก</span>
        <span class="news-radar-tag bearish" id="newsTagD5">DDR5 -1.40%</span>
      </div>
      <div style="font-size: 12px; color: var(--muted); margin-bottom: 12px;" id="newsSentimentText">
        กำลังโหลดสัญญาณข่าวอุตสาหกรรมชิป...
      </div>

      <div id="newsList">
        <!-- News dynamically injected -->
      </div>
    </div>

    <footer>
      RakaRAM &copy; 2026 &middot; ข้อมูลราคาจริงจากตลาดไอทีไทย &middot; 
      <a href="/docs" target="_blank">Swagger API</a>
    </footer>
  </div>

  <script>
    const SPECS = {
      gen: 'DDR5',
      cap: 32,
      brand: 'Kingston',
      form: 'UDIMM',
      speed: 6000,
      cl: 36,
      mod: 2,
      rgb: 1
    };

    const SPEED_PRESETS = {
      DDR4: [2666, 3200, 3600, 4000],
      DDR5: [4800, 5200, 5600, 6000, 6400, 7200]
    };

    const CL_PRESETS = {
      DDR4: [16, 18, 19, 22],
      DDR5: [30, 32, 36, 38, 40]
    };

    function populateDropdowns() {
      const speedSel = document.getElementById('selSpeed');
      const clSel = document.getElementById('selCl');
      
      speedSel.innerHTML = SPEED_PRESETS[SPECS.gen].map(s => 
        `<option value="${s}" ${s === SPECS.speed ? 'selected' : ''}>${s} MHz</option>`
      ).join('');
      
      clSel.innerHTML = CL_PRESETS[SPECS.gen].map(c => 
        `<option value="${c}" ${c === SPECS.cl ? 'selected' : ''}>CL ${c}</option>`
      ).join('');
    }

    function selectPill(type, val, elem) {
      SPECS[type] = val;
      const parent = elem.parentElement;
      parent.querySelectorAll('.pill').forEach(p => p.classList.remove('active'));
      elem.classList.add('active');

      if (type === 'gen') {
        SPECS.speed = val === 'DDR5' ? 6000 : 3200;
        SPECS.cl = val === 'DDR5' ? 36 : 16;
        populateDropdowns();
      }
      updateCalc();
    }

    async function updateCalc() {
      SPECS.brand = document.getElementById('selBrand').value;
      SPECS.form = document.getElementById('selForm').value;
      SPECS.speed = parseInt(document.getElementById('selSpeed').value, 10);
      SPECS.cl = parseInt(document.getElementById('selCl').value, 10);
      SPECS.mod = parseInt(document.getElementById('selMod').value, 10);
      SPECS.rgb = parseInt(document.getElementById('selRgb').value, 10);

      document.getElementById('dispSpecSummary').innerText = 
        `${SPECS.brand} ${SPECS.gen} ${SPECS.cap}GB ${SPECS.speed}MHz`;

      try {
        const query = new URLSearchParams({
          brand: SPECS.brand,
          generation: SPECS.gen,
          capacity_gb: SPECS.cap,
          modules: SPECS.mod,
          speed_mhz: SPECS.speed,
          cas_latency: SPECS.cl,
          has_rgb: SPECS.rgb,
          form_factor: SPECS.form
        });

        const res = await fetch(`/predict?${query.toString()}`);
        if (!res.ok) throw new Error('API error');
        const data = await res.json();

        // Update Price Card
        document.getElementById('dispPrice').innerText = data.estimated_price_thb.toLocaleString();
        document.getElementById('dispRange').innerText = 
          `ช่วงราคาที่เหมาะสม: ${data.price_range.lower_bound_thb.toLocaleString()} - ${data.price_range.upper_bound_thb.toLocaleString()} บาท`;

        const badge = document.getElementById('dispRecBadge');
        badge.innerText = data.forecast.recommendation_badge_th || 'ซื้อได้เลย';
        badge.className = `rec-badge ${data.forecast.recommendation_type || 'buy_now'}`;

        // 6M Forecast
        const est = data.estimated_price_thb;
        const p1 = data.forecast.forecast_1m_thb;
        const p3 = data.forecast.forecast_3m_thb;
        const p6 = data.forecast.forecast_6m_thb;

        document.getElementById('fc1Val').innerText = p1.toLocaleString() + ' ฿';
        document.getElementById('fc3Val').innerText = p3.toLocaleString() + ' ฿';
        document.getElementById('fc6Val').innerText = p6.toLocaleString() + ' ฿';

        const d1 = p1 - est;
        const d3 = p3 - est;
        const d6 = p6 - est;

        document.getElementById('fc1Diff').innerText = (d1 >= 0 ? '+' : '') + d1 + ' ฿';
        document.getElementById('fc1Diff').className = `fc-box-diff ${d1 <= 0 ? 'down' : 'up'}`;

        document.getElementById('fc3Diff').innerText = (d3 >= 0 ? '+' : '') + d3 + ' ฿';
        document.getElementById('fc3Diff').className = `fc-box-diff ${d3 <= 0 ? 'down' : 'up'}`;

        document.getElementById('fc6Diff').innerText = (d6 >= 0 ? '+' : '') + d6 + ' ฿';
        document.getElementById('fc6Diff').className = `fc-box-diff ${d6 <= 0 ? 'down' : 'up'}`;

      } catch (err) {
        console.error(err);
      }
    }

    async function loadNewsRadar() {
      try {
        const res = await fetch('/news');
        const data = await res.json();
        
        document.getElementById('newsSentimentText').innerText = data.market_sentiment_th || '';
        const d4 = data.net_impact_ddr4_pct || 0;
        const d5 = data.net_impact_ddr5_pct || 0;
        
        const tag = document.getElementById('newsTagD5');
        tag.innerText = `DDR4 ${d4 > 0 ? '+' : ''}${d4.toFixed(1)}% | DDR5 ${d5 > 0 ? '+' : ''}${d5.toFixed(1)}%`;
        tag.className = `news-radar-tag ${d5 < 0 ? 'bearish' : 'bullish'}`;

        const list = document.getElementById('newsList');
        list.innerHTML = (data.articles || []).slice(0, 3).map(art => `
          <div class="news-item">
            <div style="display:flex; justify-content:space-between; margin-bottom:4px;">
              <span style="font-size:10px; color:var(--muted);">${art.source}</span>
              <span style="font-size:10px; font-weight:600; color:${art.impact_multiplier_pct < 0 ? '#34D399' : (art.impact_multiplier_pct > 0 ? '#F87171' : 'var(--muted)')}">
                ${art.target_generation} ${art.impact_multiplier_pct > 0 ? '+' : ''}${art.impact_multiplier_pct.toFixed(1)}%
              </span>
            </div>
            <div class="news-item-title">${art.headline_th || art.title}</div>
            <div class="news-item-sub">${art.summary_th || art.summary}</div>
          </div>
        `).join('');

      } catch (e) {
        console.error(e);
      }
    }

    // Init on page load
    populateDropdowns();
    updateCalc();
    loadNewsRadar();
  </script>
</body>
</html>
"""

# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------
@app.get("/", tags=["Dashboard"], response_class=HTMLResponse)
def index_page(request: Request) -> Any:
    """
    Renders the interactive Mobile Web Dashboard for browser users,
    or returns API JSON metadata if requested by programmatic clients (curl / Accept: application/json).
    """
    accept_header = request.headers.get("accept", "")
    if "application/json" in accept_header and "text/html" not in accept_header:
        return JSONResponse(content={
            "service": "RakaRAM API",
            "version": "1.0.0",
            "docs": "/docs",
            "endpoints": ["/predict", "/news", "/market/timeline", "/health"]
        })
    return HTMLResponse(content=MOBILE_WEB_DASHBOARD_HTML)


@app.get("/api", tags=["General"])
def api_info() -> Dict[str, Any]:
    """API overview and service endpoints."""
    return {
        "service": "RakaRAM API",
        "description": "RAM Price Prediction & Global DRAM Market Intelligence",
        "version": "1.0.0",
        "status": "online",
        "docs": "/docs",
        "endpoints": {
            "web_ui": "/",
            "predict_get": "/predict?brand=Kingston&generation=DDR5&capacity_gb=32&modules=2&speed_mhz=6000&cas_latency=36&has_rgb=1",
            "predict_post": "/predict",
            "news": "/news",
            "market_timeline": "/market/timeline",
            "health": "/health"
        },
        "timestamp": datetime.now(timezone.utc).isoformat()
    }


@app.get("/health", tags=["General"])
def health_check() -> Dict[str, Any]:
    """Health check endpoint: Verifies model accessibility and service readiness."""
    try:
        predictor = get_predictor()
        model_ready = predictor is not None
    except Exception:
        model_ready = False

    return {
        "status": "healthy" if model_ready else "degraded",
        "model_loaded": model_ready,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }


@app.get("/predict", response_model=PredictResponse, tags=["Inference"])
def predict_price_get(
    brand: str = Query("Kingston", description="Brand name (e.g. Kingston, Corsair, G.Skill, KLEVV)"),
    generation: str = Query("DDR5", pattern="^(DDR4|DDR5)$", description="Memory generation: DDR4 or DDR5"),
    capacity_gb: int = Query(32, ge=4, le=256, description="Total capacity in GB"),
    modules: int = Query(2, ge=1, le=4, description="Number of modules (1, 2, 4)"),
    speed_mhz: int = Query(6000, ge=2133, le=10000, description="Speed in MHz"),
    cas_latency: int = Query(36, ge=10, le=50, description="CAS latency"),
    has_rgb: int = Query(1, ge=0, le=1, description="RGB flag: 1 for Yes, 0 for No"),
    form_factor: str = Query("UDIMM", pattern="^(UDIMM|SO-DIMM)$", description="Form factor")
) -> Dict[str, Any]:
    """
    Real-time price prediction via GET query parameters.
    Returns fair market price in Thai Baht (THB), confidence interval, and 6-month forecast.
    """
    try:
        predictor = get_predictor()
        result = predictor.predict(
            brand=brand,
            generation=generation,
            capacity_gb=capacity_gb,
            modules=modules,
            speed_mhz=speed_mhz,
            cas_latency=cas_latency,
            has_rgb=has_rgb,
            form_factor=form_factor
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/predict", response_model=PredictResponse, tags=["Inference"])
def predict_price_post(payload: PredictRequest) -> Dict[str, Any]:
    """
    Real-time price prediction via POST JSON payload.
    Returns fair market price in Thai Baht (THB), confidence interval, and 6-month forecast.
    """
    try:
        predictor = get_predictor()
        result = predictor.predict(
            brand=payload.brand,
            generation=payload.generation,
            capacity_gb=payload.capacity_gb,
            modules=payload.modules,
            speed_mhz=payload.speed_mhz,
            cas_latency=payload.cas_latency,
            has_rgb=payload.has_rgb,
            form_factor=payload.form_factor
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.get("/news", tags=["Market Intelligence"])
def get_news(force_refresh: bool = Query(False, description="Whether to trigger fresh news sync")) -> Dict[str, Any]:
    """
    Retrieves latest global DRAM news intelligence, sentiment classification,
    and net price multipliers for DDR4 and DDR5.
    """
    try:
        if force_refresh:
            return analyze_news_articles(force_refresh=True)
        return get_latest_news_analysis()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"News analysis error: {str(e)}")


@app.get("/market/timeline", tags=["Market Intelligence"])
def get_market_timeline() -> Dict[str, Any]:
    """
    Returns the projected 6-month price movement comparison for DDR4 vs DDR5,
    dynamically adjusted by global semiconductor intelligence.
    """
    months = ["Nov 26", "Dec 26", "Jan 27", "Feb 27", "Mar 27", "Apr 27", "May 27"]
    d4_base = [100.0, 100.3, 100.8, 101.5, 102.1, 102.7, 103.2]
    d5_base = [100.0, 98.7, 97.3, 96.0, 94.8, 93.8, 93.0]

    try:
        news_intel = get_latest_news_analysis()
        d4_impact = news_intel.get("net_impact_ddr4_pct", 0.0)
        d5_impact = news_intel.get("net_impact_ddr5_pct", 0.0)
    except Exception:
        d4_impact, d5_impact = 0.0, 0.0

    adj4 = (d4_impact / 100.0) * 0.4 if d4_impact else 0.0
    adj5 = (d5_impact / 100.0) * 0.4 if d5_impact else 0.0

    d4_timeline = [round(r + (adj4 * (i / 6.0) * 100.0), 1) for i, r in enumerate(d4_base)]
    d5_timeline = [round(r + (adj5 * (i / 6.0) * 100.0), 1) for i, r in enumerate(d5_base)]

    return {
        "months": months,
        "ddr4": {
            "name": "DDR4 (Legacy · Foundry supply shrinking)",
            "index_timeline": d4_timeline,
            "net_news_impact_pct": d4_impact,
            "trend": "upward_drift"
        },
        "ddr5": {
            "name": "DDR5 (Mainstream · Die yield expanding)",
            "index_timeline": d5_timeline,
            "net_news_impact_pct": d5_impact,
            "trend": "downward_easing"
        },
        "updated_at": datetime.now(timezone.utc).isoformat()
    }


# ---------------------------------------------------------------------------
# Local Development Execution
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
