# Project Structure

```text
VoltBrain/
├── .streamlit/
│   └── config.toml             # Streamlit UI configuration
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

## Directory Descriptions
- **`src/`**: Machine learning backend pipelines (`preprocess.py`, `train.py`).
- **`data/`**: Raw dataset storage directory (`EV_Specs_Dataset.csv`).
- **`models/`**: Serialized model artifacts (`voltbrain_pipeline.pkl`) and training metric reports.
- **`assets/`**: Static image assets and diagnostic charts.
- **`.streamlit/`**: UI configuration settings.
