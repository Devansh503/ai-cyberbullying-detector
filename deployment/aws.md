# AWS Deployment (ECS Fargate)

1) Build and push image to ECR
- aws ecr create-repository --repository-name cyberbully-api
- docker build -t cyberbully-api .
- docker tag cyberbully-api:latest <AWS_ACCOUNT>.dkr.ecr.<REGION>.amazonaws.com/cyberbully-api:latest
- aws ecr get-login-password --region <REGION> | docker login --username AWS --password-stdin <AWS_ACCOUNT>.dkr.ecr.<REGION>.amazonaws.com
- docker push <AWS_ACCOUNT>.dkr.ecr.<REGION>.amazonaws.com/cyberbully-api:latest

2) Create ECS Fargate service
- Task definition with 512MB/1vCPU, port 8000, health check GET /docs
- Service behind Application Load Balancer with listener 80 → target group
- Auto Scaling: target CPU 50%, min 1, max 5 tasks

3) Env vars: APP_ENV=prod; ENABLE_TRANSLATION=1

4) Security: Restrict ALB ingress; add WAF if needed.
