---
title: VoltBrain AI EV Range Prediction Engine
emoji: ⚡
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: 1.38.0
app_file: app.py
pinned: false
license: mit
---

# VoltBrain — Electric Vehicle Range Prediction Engine

VoltBrain is a machine learning application designed to estimate real-world electric vehicle (EV) driving range based on physical battery and vehicle specifications. It uses an ensemble Random Forest Regressor pipeline to provide reliable estimates beyond standard WLTP/EPA laboratory benchmarks.

---

## 📌 Project Overview
Range anxiety remains a key challenge in electric vehicle adoption. Standard WLTP and EPA estimates often differ from real-world driving conditions due to weather, aerodynamics, and driving style. VoltBrain addresses this by modeling key physical parameters (Battery Capacity, Efficiency, Top Speed, Drivetrain, Brand) to deliver objective range estimates.

---

## ⚡ Features
- **Interactive Range Predictor:** Real-time range estimation interface with scenario breakdowns (City, Highway, Winter).
- **Model Performance Dashboard:** Interactive visualization of key metrics ($R^2$, MAE, RMSE, Dataset Rows).
- **Exploratory Data Analytics:** Interactive Plotly visualizations for correlation analysis and range distribution histograms.
- **Dataset Reference:** Raw specification data table and interactive data dictionary.

---

## 🛠️ Requirements & Tech Stack
- **Python 3.8+**
- **Streamlit** (Web UI Framework)
- **Scikit-Learn** (ML Pipeline & Random Forest Engine)
- **Pandas & NumPy** (Data processing & manipulation)
- **Plotly** (Interactive data visualizations)

---

## 📥 Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/VoltBrain.git
   cd VoltBrain
   ```

2. **Create and activate a virtual environment (optional but recommended):**
   ```bash
   python -m venv .venv
   # Windows:
   .venv\Scripts\activate
   # Linux/macOS:
   source .venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## 🚀 Run Instructions

### 1. Launch the Streamlit Web Application
```bash
streamlit run app.py
```

### 2. Retrain the Machine Learning Model (Optional)
To retrain the model and regenerate training diagnostics:
```bash
python src/train.py
```

---

## 📂 Project Structure
```text
VoltBrain/
├── .streamlit/
│   └── config.toml             # Custom theme settings
├── assets/                     # Diagnostic charts and documentation visuals
├── data/
│   └── EV_Specs_Dataset.csv    # EV Specifications Dataset
├── docs/                       # Additional documentation
├── models/
│   ├── voltbrain_pipeline.pkl  # Serialized Scikit-Learn ML Pipeline
│   ├── training_report.json    # Model evaluation metrics report
│   └── model_comparison.csv    # Cross-validation comparison summary
├── src/
│   ├── preprocess.py           # Feature engineering & preprocessing pipeline
│   └── train.py                # Model training & serialization script
├── .env.example                # Environment configuration template
├── .gitignore                  # Git tracking rules
├── app.py                      # Streamlit application entry point
├── CHANGELOG.md                # Project release history
├── LICENSE                     # MIT License
├── PROJECT_STRUCTURE.md        # Detailed directory documentation
├── README.md                   # Main GitHub documentation
└── requirements.txt            # Python package dependencies
```

---

## 📊 Model Information
- **Algorithm:** Random Forest Regressor Pipeline
- **Preprocessing:** Standard Scaling (Numerical) & One-Hot Encoding (Categorical) with Median/Mode Imputation
- **Target Variable:** Real-world driving range (`Range (km)`)

---

## 🗄️ Dataset
- **Source:** EV Specifications Dataset (`data/EV_Specs_Dataset.csv`)
- **Key Features:** Battery Capacity (kWh), Energy Efficiency (Wh/km), Top Speed (km/h), Drivetrain Type (AWD/FWD/RWD), Brand

---

## 🔮 Future Improvements
- Integration of real-time temperature and elevation APIs.
- Expansion to support battery degradation profiles over vehicle age.
- Docker containerization for automated cloud deployment.

---

## 📄 License
Distributed under the MIT License. See `LICENSE` for details.

---

## 👨‍💻 Author
Developed as an ML and software engineering portfolio project.
