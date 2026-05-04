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
        "gcsfs"
    ]
)
def evaluator(
    test_dataset: dsl.Input[dsl.Dataset],
    model_artifact: dsl.Input[dsl.Artifact],
    auc_threshold: float
) -> bool:
    """
    KFP v2 component to evaluate model performance and determine if it meets production thresholds.
    """
    import pandas as pd
    import logging
    from xgboost import XGBClassifier
    from sklearn.metrics import roc_auc_score, f1_score

    # Component-level logging setup
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("kfp_evaluator")

    try:
        logger.info(f"Loading testing data from {test_dataset.path}")
        test_df = pd.read_parquet(test_dataset.path)

        target = 'Churn'
        X_test = test_df.drop(columns=[target])
        y_test = test_df[target]

        logger.info(f"Loading model from {model_artifact.path}")
        import os
        model = XGBClassifier()
        model.load_model(os.path.join(model_artifact.path, "model.bst"))

        logger.info("Generating predictions and probabilities...")
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] # Probabilities for the positive class

        logger.info("Calculating evaluation metrics...")
        auc_roc = roc_auc_score(y_test, y_prob)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        logger.info(f"Model Performance - AUC-ROC: {auc_roc:.4f}, F1-Score: {f1:.4f}")
        logger.info(f"Required AUC-ROC Threshold: {auc_threshold}")

        is_approved = bool(auc_roc > auc_threshold)
        
        if is_approved:
            logger.info("Model PASSED the evaluation threshold.")
        else:
            logger.warning("Model FAILED the evaluation threshold.")

        return is_approved

    except Exception as e:
        logger.error(f"Pipeline execution failed in evaluator component: {str(e)}")
        raise e

if __name__ == "__main__":
    # Local execution implementation using argparse for debugging
    parser = argparse.ArgumentParser(description="Evaluator KFP Component (Local Test)")
    parser.add_argument("--test_in", type=str, required=True, help="Local path to test.parquet")
    parser.add_argument("--model_in", type=str, required=True, help="Local path to model.bst")
    parser.add_argument("--auc_threshold", type=float, default=0.85, help="Minimum AUC-ROC to approve")

    args = parser.parse_args()
    logger.info("Running Evaluator locally via argparse...")
    
    # Mock KFP Artifact objects for local execution
    class MockArtifact:
        def __init__(self, path):
            self.path = path
            self.uri = f"file://{path}"
            
    mock_test = MockArtifact(args.test_in)
    mock_model = MockArtifact(args.model_in)

    # Call the core function directly
    is_approved = evaluator.python_func(
        test_dataset=mock_test,
        model_artifact=mock_model,
        auc_threshold=args.auc_threshold
    )
    
    logger.info(f"Local Execution Complete. Model Approved: {is_approved}")