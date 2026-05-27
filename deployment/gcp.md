# GCP Deployment (Cloud Run)

1) Build and push to Artifact Registry
- gcloud artifacts repositories create cyberbully --repository-format=docker --location=<REGION>
- gcloud auth configure-docker <REGION>-docker.pkg.dev
- docker build -t <REGION>-docker.pkg.dev/<PROJECT_ID>/cyberbully/api:latest .
- docker push <REGION>-docker.pkg.dev/<PROJECT_ID>/cyberbully/api:latest

2) Deploy to Cloud Run
- gcloud run deploy cyberbully-api --image <REGION>-docker.pkg.dev/<PROJECT_ID>/cyberbully/api:latest --region <REGION> --platform managed --allow-unauthenticated --port 8000 --memory=1Gi --cpu=1

3) Set env vars: APP_ENV=prod, ENABLE_TRANSLATION=1

4) Configure min/max instances for auto-scaling.
