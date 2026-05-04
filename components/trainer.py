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
    packages_to_install=[
        "pandas", 
        "xgboost", 
        "scikit-learn", 
        "pyarrow", 
        "fsspec", 
        "gcsfs", 
        "google-cloud-aiplatform"
    ]
)
def trainer(
    train_dataset: dsl.Input[dsl.Dataset],
    test_dataset: dsl.Input[dsl.Dataset],
    learning_rate: float,
    max_depth: int,
    n_estimators: int,
    project_id: str,
    location: str,
    experiment_name: str,
    run_name: str,
    serving_container_image_uri: str,
    model_artifact: dsl.Output[dsl.Artifact]
):
    """
    KFP v2 component to train an XGBoost model and log to Vertex AI Experiments.
    """
    import pandas as pd
    import logging
    from xgboost import XGBClassifier
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    from google.cloud import aiplatform

    # Component-level logging setup
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("kfp_trainer")

    try:
        logger.info("Initializing Vertex AI Experiments...")
        aiplatform.init(project=project_id, location=location, experiment=experiment_name)
        
        # Start the experiment run
        aiplatform.start_run(run=f"{run_name}-trainer")

        logger.info(f"Loading training data from {train_dataset.path}")
        train_df = pd.read_parquet(train_dataset.path)
        
        logger.info(f"Loading testing data from {test_dataset.path}")
        test_df = pd.read_parquet(test_dataset.path)

        target = 'Churn'
        X_train = train_df.drop(columns=[target])
        y_train = train_df[target]
        X_test = test_df.drop(columns=[target])
        y_test = test_df[target]

        # Log parameters to Vertex AI
        params = {
            "learning_rate": learning_rate,
            "max_depth": max_depth,
            "n_estimators": n_estimators,
            "objective": "binary:logistic"
        }
        logger.info(f"Logging parameters: {params}")
        aiplatform.log_params(params)

        logger.info("Initializing and training XGBoost model...")
        model = XGBClassifier(**params, random_state=42)
        model.fit(X_train, y_train)

        logger.info("Generating predictions on test set...")
        y_pred = model.predict(X_test)

        logger.info("Calculating metrics...")
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1_score": f1_score(y_test, y_pred, zero_division=0)
        }
        
        logger.info(f"Logging metrics: {metrics}")
        aiplatform.log_metrics(metrics)

        logger.info(f"Saving model artifact to {model_artifact.path}")
        # XGBoost handles saving directly to the specified path
        import os
        # Create a directory at the artifact path
        os.makedirs(model_artifact.path, exist_ok=True)
        # Save the model specifically as 'model.bst' inside that directory
        model.save_model(os.path.join(model_artifact.path, "model.bst"))
        model_artifact.metadata["containerSpec"] = {
            "imageUri": serving_container_image_uri
        }

        logger.info("Ending Vertex AI Experiment run...")
        aiplatform.end_run()

    except Exception as e:
        logger.error(f"Pipeline execution failed in trainer component: {str(e)}")
        # Ensure experiment run is ended even on failure to avoid zombie runs
        aiplatform.end_run()
        raise e

if __name__ == "__main__":
    # Local execution implementation using argparse for debugging
    parser = argparse.ArgumentParser(description="Trainer KFP Component (Local Test)")
    parser.add_argument("--train_in", type=str, required=True, help="Local path to train.parquet")
    parser.add_argument("--test_in", type=str, required=True, help="Local path to test.parquet")
    parser.add_argument("--learning_rate", type=float, default=0.1)
    parser.add_argument("--max_depth", type=int, default=5)
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--project_id", type=str, required=True)
    parser.add_argument("--location", type=str, default="us-central1")
    parser.add_argument("--experiment_name", type=str, default="local-churn-experiment")
    parser.add_argument("--run_name", type=str, default="local-run-1")
    parser.add_argument("--model_out", type=str, required=True, help="Local path to save model.bst")

    args = parser.parse_args()
    logger.info("Running Trainer locally via argparse...")
    
    # Mock KFP Dataset and Model objects for local execution
    class MockArtifact:
        def __init__(self, path):
            self.path = path
            self.uri = f"file://{path}"
            
    mock_train = MockArtifact(args.train_in)
    mock_test = MockArtifact(args.test_in)
    mock_model = MockArtifact(args.model_out)

    # Call the core function directly
    trainer.python_func(
        train_dataset=mock_train,
        test_dataset=mock_test,
        learning_rate=args.learning_rate,
        max_depth=args.max_depth,
        n_estimators=args.n_estimators,
        project_id=args.project_id,
        location=args.location,
        experiment_name=args.experiment_name,
        run_name=args.run_name,
        model_artifact=mock_model
    )
    
    logger.info("Local Execution Complete.")