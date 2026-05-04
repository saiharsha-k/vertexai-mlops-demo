import argparse
import logging
from kfp import dsl

# Set up local script logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dsl.component(
    base_image="python:3.10",
    packages_to_install=["pandas", "scikit-learn", "fsspec", "gcsfs", "pyarrow"]
)
def data_loader(
    input_csv_path: str,
    test_size: float,
    random_state: int,
    train_dataset: dsl.Output[dsl.Dataset],
    test_dataset: dsl.Output[dsl.Dataset]
) -> dict:
    """
    KFP v2 component to load data from GCS, preprocess, and output stratified Parquet splits.
    """
    import pandas as pd
    import numpy as np
    import logging
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.compose import ColumnTransformer

    # Component-level logging setup
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("kfp_data_loader")

    try:
        logger.info(f"Loading data from {input_csv_path}")
        df = pd.read_csv(input_csv_path)

        # Handle typical Telco Churn issue: 'TotalCharges' often contains blank spaces
        logger.info("Cleaning data: Coercing TotalCharges to numeric")
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0.0)

        target = 'Churn'
        drop_cols = ['customerID']
        numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
        categorical_cols = [col for col in df.columns if col not in numeric_cols + drop_cols + [target]]

        logger.info(f"Dropping columns: {drop_cols}")
        df = df.drop(columns=drop_cols)

        logger.info("Encoding target variable (Yes=1, No=0)")
        df[target] = df[target].map({'Yes': 1, 'No': 0})

        X = df.drop(columns=[target])
        y = df[target]

        logger.info(f"Performing stratified train/test split (test_size={test_size})")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        logger.info("Setting up ColumnTransformer preprocessing pipeline")
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numeric_cols),
                ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_cols)
            ]
        )

        logger.info("Fitting and transforming data")
        X_train_processed = preprocessor.fit_transform(X_train)
        X_test_processed = preprocessor.transform(X_test)

        # Extract feature names to rebuild dataframes
        num_features = numeric_cols
        cat_features = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_cols)
        feature_names = num_features + list(cat_features)

        train_df = pd.DataFrame(X_train_processed, columns=feature_names)
        train_df[target] = y_train.values

        test_df = pd.DataFrame(X_test_processed, columns=feature_names)
        test_df[target] = y_test.values

        logger.info(f"Saving training data to {train_dataset.path}")
        train_df.to_parquet(train_dataset.path, index=False)

        logger.info(f"Saving testing data to {test_dataset.path}")
        test_df.to_parquet(test_dataset.path, index=False)

        # Return metadata dict
        return {
            "train_uri": train_dataset.uri,
            "test_uri": test_dataset.uri,
            "train_shape": str(train_df.shape),
            "test_shape": str(test_df.shape)
        }

    except Exception as e:
        logger.error(f"Pipeline execution failed in data_loader component: {str(e)}")
        raise e

if __name__ == "__main__":
    # Local execution implementation using argparse for debugging and compliance
    parser = argparse.ArgumentParser(description="Data Loader KFP Component (Local Test)")
    parser.add_argument("--input_csv_path", type=str, required=True, help="GCS or local path to input CSV")
    parser.add_argument("--test_size", type=float, default=0.2, help="Test split size")
    parser.add_argument("--random_state", type=int, default=42, help="Random state seed")
    parser.add_argument("--train_out", type=str, required=True, help="Local path to save train.parquet")
    parser.add_argument("--test_out", type=str, required=True, help="Local path to save test.parquet")

    args = parser.parse_args()
    logger.info("Running Data Loader locally via argparse...")
    
    # Mock KFP Dataset objects for local execution
    class MockDataset:
        def __init__(self, path):
            self.path = path
            self.uri = f"file://{path}"
            
    mock_train = MockDataset(args.train_out)
    mock_test = MockDataset(args.test_out)

    # Call the core function directly (KFP ignores the decorator when called this way)
    result = data_loader.python_func(
        input_csv_path=args.input_csv_path,
        test_size=args.test_size,
        random_state=args.random_state,
        train_dataset=mock_train,
        test_dataset=mock_test
    )
    
    logger.info(f"Local Execution Complete. Metadata: {result}")