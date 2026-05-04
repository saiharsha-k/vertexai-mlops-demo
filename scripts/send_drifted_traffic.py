import os
import yaml
import logging
import pandas as pd
from google.cloud import aiplatform
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # Load Configuration
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    config_path = os.path.join(base_dir, "config", "pipeline_config.yaml")
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    project_id = config.get("project", {}).get("id")
    location = config.get("project", {}).get("region")
    v1_uri = config.get("data", {}).get("raw_data_gcs_uri")
    v2_uri = config.get("data", {}).get("drifted_data_gcs_uri")

    aiplatform.init(project=project_id, location=location)

    # 1. Fit Preprocessor on V1 to ensure exact column mapping
    logger.info("Loading V1 data to construct preprocessing schema...")
    df_v1 = pd.read_csv(v1_uri)
    df_v1['TotalCharges'] = pd.to_numeric(df_v1['TotalCharges'], errors='coerce').fillna(0.0)
    
    drop_cols = ['customerID', 'Churn']
    numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    categorical_cols = [col for col in df_v1.columns if col not in numeric_cols + drop_cols]
    
    X_v1 = df_v1.drop(columns=[c for c in drop_cols if c in df_v1.columns])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_cols),
            ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), categorical_cols)
        ]
    )
    preprocessor.fit(X_v1)

    # 2. Process V2 Drifted Data
    logger.info(f"Loading and processing Drifted V2 data from {v2_uri}...")
    df_v2 = pd.read_csv(v2_uri)
    df_v2['TotalCharges'] = pd.to_numeric(df_v2['TotalCharges'], errors='coerce').fillna(0.0)
    X_v2 = df_v2.drop(columns=[c for c in drop_cols if c in df_v2.columns])
    
    # Transform V2 into XGBoost-ready floats
    X_v2_processed = preprocessor.transform(X_v2)

    # 3. Locate Endpoint and Send Traffic
    logger.info("Locating Vertex AI Endpoint...")
    endpoints = aiplatform.Endpoint.list(filter='display_name="telco-churn-endpoint"')
    if not endpoints:
        logger.error("Could not find endpoint.")
        return
    endpoint = endpoints[0]

    instances = X_v2_processed.tolist()
    batch_size = 100
    
    logger.info(f"Sending {len(instances)} drifted records to the Endpoint in batches of {batch_size}...")
    for i in range(0, len(instances), batch_size):
        batch = instances[i:i+batch_size]
        endpoint.predict(instances=batch)
        logger.info(f"  -> Sent instances {i} to {i + len(batch)}")

    logger.info("Successfully fired all drifted traffic to the endpoint! Model Monitoring will log these payloads.")

if __name__ == "__main__":
    main()