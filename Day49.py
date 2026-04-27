"""
Day 49: Dockerizing FastAPI AI Application

What I did:
- Created Dockerfile for FastAPI app
- Used Python base image
- Installed dependencies
- Exposed port 8000
- Ran app using Uvicorn

Command to build:
docker build -t ai-app .

Command to run:
docker run -p 8000:8000 ai-app
"""