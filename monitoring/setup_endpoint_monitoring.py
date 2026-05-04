import os
import yaml
import logging
from google.cloud import aiplatform
from google.cloud.aiplatform import model_monitoring

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
    alert_email = config.get("monitoring", {}).get("alert_email")
    skew_threshold = config.get("monitoring", {}).get("skew_threshold", 0.3)

    logger.info(f"Initializing Vertex AI for project {project_id}...")
    aiplatform.init(project=project_id, location=location)

    # 1. Locate the deployed endpoint
    endpoints = aiplatform.Endpoint.list(filter='display_name="telco-churn-endpoint"')
    if not endpoints:
        logger.error("Could not find 'telco-churn-endpoint'. Ensure the pipeline created it.")
        return
    endpoint = endpoints[0]
    logger.info(f"Found Endpoint: {endpoint.resource_name}")

    # NEW: Check if the endpoint is empty, and deploy the model if necessary
    if not endpoint.list_models():
        logger.warning("Endpoint is currently empty! Deploying the model now...")
        logger.info("(Note: Vertex AI Model Deployment takes ~10-15 minutes. Grab a coffee!)")
        
        # Find the model we uploaded in the pipeline
        models = aiplatform.Model.list(filter='display_name="telco-churn-xgboost"')
        if not models:
            logger.error("Could not find the 'telco-churn-xgboost' model in the registry.")
            return
        
        latest_model = models[0]
        
        # Deploy the model to the endpoint
        latest_model.deploy(
            endpoint=endpoint,
            machine_type="n1-standard-4",
            min_replica_count=1,
            max_replica_count=1
        )
        logger.info("Model deployment complete!")
    else:
        logger.info("Model is already deployed to the endpoint.")

    # 2. Configure Sampling and Email Alerts
    sample_config = model_monitoring.RandomSampleConfig(sample_rate=1.0) # Log 100% of traffic
    alert_config = model_monitoring.EmailAlertConfig(
        user_emails=[alert_email], 
        enable_logging=True
    )
    # Run monitoring analysis every 1 hour
    schedule_config = model_monitoring.ScheduleConfig(monitor_interval=1)

    # 3. Configure Prediction Drift Detection
    # Compares recent incoming traffic distributions against historical traffic
    drift_config = model_monitoring.DriftDetectionConfig(
        drift_thresholds={
            "MonthlyCharges": skew_threshold,
            "tenure": skew_threshold
        }
    )
    
    objective_config = model_monitoring.ObjectiveConfig(
        drift_detection_config=drift_config,
        explanation_config=None
    )

    # 4. Create the Monitoring Job
    logger.info("Creating Model Deployment Monitoring Job (This takes a few seconds)...")
    monitoring_job = aiplatform.ModelDeploymentMonitoringJob.create(
        display_name="churn-drift-monitor",
        endpoint=endpoint,
        logging_sampling_strategy=sample_config,
        schedule_config=schedule_config,
        alert_config=alert_config,
        objective_configs=objective_config,
    )
    logger.info(f"Success! Monitoring Job created: {monitoring_job.resource_name}")

if __name__ == "__main__":
    main()