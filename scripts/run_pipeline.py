import os
import sys
import yaml
import argparse
import logging
from datetime import datetime
from google.cloud import aiplatform
from kfp import compiler

# Add the parent directory to the path so we can import the pipeline definition
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from pipeline.churn_pipeline import churn_pipeline

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_config(config_path: str) -> dict:
    """Loads pipeline configuration from a YAML file."""
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded configuration from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Failed to load config file: {e}")
        raise

def main(args):
    # Determine paths
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    config_path = os.path.join(base_dir, "config", "pipeline_config.yaml")
    compiled_pipeline_path = os.path.join(base_dir, "churn_pipeline.json")
    
    # Load configuration
    config = load_config(config_path)
    
    # Extract variables from the NESTED config structure
    project_id = args.project_id or config.get("project", {}).get("id")
    location = args.location or config.get("project", {}).get("region")
    pipeline_root = args.pipeline_root or config.get("pipeline", {}).get("pipeline_root")
    input_csv = args.input_csv or config.get("data", {}).get("raw_data_gcs_uri")
    experiment_name = args.experiment_name or config.get("pipeline", {}).get("experiment_name", "telco-churn-experiment")
    auc_threshold = config.get("model", {}).get("eval_threshold_auc", 0.85)
    
    if not all([project_id, location, pipeline_root, input_csv]):
        logger.error(f"Missing required configuration. Extracted values: Project={project_id}, Location={location}, Root={pipeline_root}, CSV={input_csv}")
        sys.exit(1)

    # Generate a unique run name based on timestamp
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = f"churn-pipeline-run-{timestamp}"

    try:
        logger.info(f"Compiling pipeline to {compiled_pipeline_path}...")
        compiler.Compiler().compile(
            pipeline_func=churn_pipeline,
            package_path=compiled_pipeline_path
        )
        logger.info("Pipeline compiled successfully.")

        logger.info(f"Initializing Vertex AI SDK for project {project_id} in {location}...")
        aiplatform.init(
            project=project_id,
            location=location,
            experiment=experiment_name
        )

        # Define pipeline parameters based on config/args
        pipeline_params = {
            "input_csv_path": input_csv,
            "project_id": project_id,
            "location": location,
            "experiment_name": experiment_name,
            "run_name": run_name,
            "test_size": 0.2,            # Hardcoded default or add to config
            "learning_rate": 0.1,        # Hardcoded default or add to config
            "max_depth": 5,              # Hardcoded default or add to config
            "n_estimators": 100,         # Hardcoded default or add to config
            "auc_threshold": auc_threshold,
            "serving_container_image_uri": "us-docker.pkg.dev/vertex-ai/prediction/xgboost-cpu.1-6:latest"
        }

        logger.info("Preparing Vertex AI PipelineJob...")
        job = aiplatform.PipelineJob(
            display_name=f"telco-churn-job-{timestamp}",
            template_path=compiled_pipeline_path,
            pipeline_root=pipeline_root,
            parameter_values=pipeline_params,
            enable_caching=True
        )

        logger.info(f"Submitting PipelineJob to Vertex AI. Experiment: {experiment_name}")
        job.submit(experiment=experiment_name)

        logger.info("Pipeline submitted successfully!")
        logger.info("======================================================")
        logger.info(f"Monitor your pipeline run at the following URL:")
        logger.info(job._dashboard_uri())
        logger.info("======================================================")

    except Exception as e:
        logger.error(f"Failed to execute pipeline submission: {e}")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Submit KFP Pipeline to Vertex AI")
    parser.add_argument("--project_id", type=str, help="GCP Project ID")
    parser.add_argument("--location", type=str, help="GCP Region (e.g., us-central1)")
    parser.add_argument("--pipeline_root", type=str, help="GCS URI for pipeline artifacts")
    parser.add_argument("--input_csv", type=str, help="GCS path to the input dataset (e.g., gs://vertex-mlops-demo-2026/data/churn_v1.csv)")
    parser.add_argument("--experiment_name", type=str, help="Vertex AI Experiment name")

    args = parser.parse_args()
    main(args)