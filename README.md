# ⚡ RakaRAM (ราคาแรม)

**RakaRAM** คือระบบวิเคราะห์และทำนายราคาแรมคอมพิวเตอร์ (RAM Price Analysis & Real-time Prediction) ที่สร้างขึ้นเพื่อดึงข้อมูลราคาและสเปกจริงจากตลาดในไทย นำมาผ่านกระบวนการ Data Pipeline, ทำการวิเคราะห์ทางสถิติ (EDA), และเทรนโมเดล Machine Learning (Regression) สำหรับประเมินราคาตามสเปกแบบ Real-time พร้อมแสดงผลผ่าน Interactive Web Dashboard

---

## 🚀 ฟีเจอร์หลัก (Key Features)

- **Real-time / Live Data Pipeline**: ระบบดึงข้อมูลราคาและสเปกแรมจากร้านค้าไอทีชั้นนำในไทย
- **Automated Feature Extractor**: สกัดสเปกละเอียดจากชื่อสินค้า (แบรนด์, DDR4/DDR5, บัสสปีด, ความจุ, จำนวนแถว, ค่า CL, ไฟ RGB)
- **Exploratory Data Analysis (EDA)**: วิเคราะห์แนวโน้มราคา ความคุ้มค่าต่อ GB (Price/GB) และเปรียบเทียบระหว่างแบรนด์
- **Machine Learning Price Predictor**: โมเดล Regression ที่ทำนายราคาแรมจากสเปกอย่างแม่นยำ พร้อมช่วงราคาประเมิน (Price Range)
- **Interactive Streamlit Web Dashboard**: เว็บแอปพลิเคชันสำหรับสำรวจตลาดและทดลองคำนวณราคาแบบ Real-time

---

## 📁 โครงสร้างโปรเจกต์ (Project Structure)

```text
RakaRAM/
├── data/
│   ├── raw/                  # ข้อมูลดิบที่ดึงมาจากแหล่งข้อมูล
│   └── processed/            # ชุดข้อมูลที่ผ่านการทำความสะอาดและสกัดฟีเจอร์แล้ว
├── src/
│   ├── scraper/              # สคริปต์ดึงข้อมูลราคาจากร้านค้า
│   ├── parser/               # ตัวแยกสเปกแรมด้วย Regex
│   ├── models/               # การเทรนและฟังก์ชันทำนายราคาด้วย ML
│   └── utils/                # ยูทิลิตี้และระบบบันทึก Log
├── notebooks/                # สคริปต์วิเคราะห์ข้อมูลและกราฟิกสรุป
├── saved_models/             # โมเดล ML ที่บันทึกไว้ (.joblib)
├── app/                      # Streamlit Web Application
├── requirements.txt          # รายการไลบรารีที่จำเป็น
└── README.md
```

---

## 🛠️ การติดตั้งและเริ่มต้นใช้งาน

1. โคลนคลังข้อมูล:
```bash
git clone https://github.com/YuttakanS/RakaRAM.git
cd RakaRAM
```

2. ติดตั้ง Dependencies:
```bash
pip install -r requirements.txt
```

---

*พัฒนาและดูแลโดย [YuttakanS](https://github.com/YuttakanS)*
