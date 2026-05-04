import os
from google.cloud import storage
import google.cloud.aiplatform as aiplatform

PROJECT_ID = os.getenv("PROJECT_ID", "garvaman-ai-poc")
REGION = os.getenv("REGION", "us-central1")

aiplatform.init(project=PROJECT_ID, location=REGION)

client = storage.Client(project=PROJECT_ID)
print("Vertex AI SDK initialized")
print("GCS client ready")
print("Project:", PROJECT_ID)
print("Region:", REGION)