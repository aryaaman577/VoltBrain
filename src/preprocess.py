import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def prepare_data(filepath):
    df = pd.read_csv(filepath)
    print("Cleaning dataset...")
    
    df = df.drop_duplicates()
    
    mapping = {
        'Range_Km': 'Range (km)', 'Range': 'Range (km)',
        'Battery_Capacity_kWh': 'Battery Capacity (kWh)', 'Battery': 'Battery Capacity (kWh)',
        'Efficiency_WhKm': 'Efficiency (Wh/km)', 'Efficiency': 'Efficiency (Wh/km)',
        'TopSpeed_KmH': 'Top Speed (km/h)', 'TopSpeed': 'Top Speed (km/h)',
        'PowerTrain': 'Drive Type', 'Make': 'Brand'
    }
    df = df.rename(columns=mapping)
    
    target = 'Range (km)'
    num_cols = ['Battery Capacity (kWh)', 'Efficiency (Wh/km)', 'Top Speed (km/h)']
    cat_cols = ['Brand', 'Drive Type']
    
    required = [target] + num_cols + cat_cols
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
        
    df = df.dropna(subset=[target])
    
    X = df[num_cols + cat_cols]
    y = df[target]
    
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_pipeline, num_cols),
            ('cat', cat_pipeline, cat_cols)
        ],
        remainder='drop'
    )
    
    return X, y, preprocessor
