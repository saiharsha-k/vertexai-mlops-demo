import os
import yaml
import logging
from kfp import dsl
from kfp import compiler

# Import custom components
from components.data_loader import data_loader
from components.trainer import trainer
from components.evaluator import evaluator

# Import Google Cloud Pipeline Components for Model Registry & Deployment (Components 4 & 5)
from google_cloud_pipeline_components.v1.model import ModelUploadOp
from google_cloud_pipeline_components.v1.endpoint import EndpointCreateOp
from google_cloud_pipeline_components.types import artifact_types
# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load pipeline configuration
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config", "pipeline_config.yaml")
try:
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)
    logger.info("Successfully loaded pipeline_config.yaml")
except FileNotFoundError:
    logger.warning(f"Config file not found at {CONFIG_PATH}. Proceeding with pipeline definition using empty defaults. Ensure config is passed at runtime.")
    config = {}

# Extract defaults from config (fallback to sensible defaults if missing)
PROJECT_ID = config.get("project_id", "garvaman-ai-poc")
LOCATION = config.get("location", "us-central1")
PIPELINE_ROOT = config.get("pipeline_root", "gs://vertex-mlops-demo/pipeline_root")
SERVING_IMAGE = config.get("serving_image", "us-docker.pkg.dev/vertex-ai/prediction/xgboost-cpu.1-6:latest")

@dsl.pipeline(
    name="telco-churn-production-pipeline",
    description="End-to-end MLOps pipeline for Telco Churn prediction including drift awareness",
    pipeline_root=PIPELINE_ROOT
)
def churn_pipeline(
    input_csv_path: str,
    project_id: str = PROJECT_ID,
    location: str = LOCATION,
    experiment_name: str = "telco-churn-experiment",
    run_name: str = "pipeline-run",
    test_size: float = 0.2,
    random_state: int = 42,
    learning_rate: float = 0.1,
    max_depth: int = 5,
    n_estimators: int = 100,
    auc_threshold: float = 0.85,
    serving_container_image_uri: str = SERVING_IMAGE
):
    # ========================================================================
    # Component 1: Data Loader
    # ========================================================================
    data_loader_task = data_loader(
        input_csv_path=input_csv_path,
        test_size=test_size,
        random_state=random_state
    )
    data_loader_task.set_caching_options(True)
    data_loader_task.set_display_name("Load & Preprocess Data")

    # ========================================================================
    # Component 2: Trainer
    # ========================================================================
    trainer_task = trainer(
        train_dataset=data_loader_task.outputs["train_dataset"],
        test_dataset=data_loader_task.outputs["test_dataset"],
        learning_rate=learning_rate,
        max_depth=max_depth,
        n_estimators=n_estimators,
        project_id=project_id,
        location=location,
        experiment_name=experiment_name,
        run_name=run_name,
        serving_container_image_uri=serving_container_image_uri
    )
    trainer_task.set_caching_options(True)
    trainer_task.set_display_name("Train XGBoost Model")

    # ========================================================================
    # Component 3: Evaluator
    # ========================================================================
    evaluator_task = evaluator(
        test_dataset=data_loader_task.outputs["test_dataset"],
        model_artifact=trainer_task.outputs["model_artifact"],
        auc_threshold=auc_threshold
    )
    evaluator_task.set_caching_options(False) # Often disabled for evaluation to ensure fresh metric checks
    evaluator_task.set_display_name("Evaluate Model Metrics")

    # ========================================================================
    # Conditional Execution (Threshold Check)
    # ========================================================================
    # ========================================================================
    # Conditional Execution (Threshold Check)
    # ========================================================================
    with dsl.Condition(evaluator_task.output == True, name="approve_model_for_production"):
        
        # ========================================================================
        # Component 4: Model Registry Upload (GCPC)
        # ========================================================================
        model_upload_task = ModelUploadOp(
            project=project_id,
            location=location,
            display_name="telco-churn-xgboost",
            description="XGBoost model predicting customer churn",
            unmanaged_container_model=trainer_task.outputs["model_artifact"] # <--- Direct pass
        )
        model_upload_task.set_display_name("Upload to Model Registry")

        # ========================================================================
        # Component 5: Create Endpoint (GCPC)
        # ========================================================================
        endpoint_create_task = EndpointCreateOp(
            project=project_id,
            location=location,
            display_name="telco-churn-endpoint",
        )
        endpoint_create_task.set_display_name("Create Vertex AI Endpoint")
        endpoint_create_task.after(model_upload_task)

if __name__ == "__main__":
    # Local compilation step for testing
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="churn_pipeline.json", help="Path to save compiled pipeline")
    args = parser.parse_args()

    logger.info(f"Compiling pipeline to {args.output}...")
    compiler.Compiler().compile(
        pipeline_func=churn_pipeline,
        package_path=args.output
    )
    logger.info("Pipeline compilation complete.")