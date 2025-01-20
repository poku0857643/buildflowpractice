# Stage 1: Development Environment
FROM python3.11-slim-buster AS development

WORKDIR /app

# Install system dependencies if needed (e.g., for specific libraries)
# RUN apt-get update && apt-get install -y --no-install-recommends \
#   libpq-dev \ # Example for psycopg2
#   && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements-dev.txt ./
RUN pip install -r requirements.txt -r requrements-dev.txt

COPY . .
#COPY the rest of the application code

# For development, you might want to specify a cmmand that starts a development server or run tests
CMD ["python", "main.py"]
# Or "pytest, or any other dev command

# Stage 2:
FROM python:3.11-slim-buster AS production

WORKDIR /app

# Install system dependencies needed for production
# RUN apt-get update && apt-get install -y --no-install-recommends \
#   libpq-dev \ # Example for psycopg2
#   && rm -rf /var/lib/apt/lists/*

COPY --from-develpment /app/requirements.txt /app/requirements_prod.txt ./
RUN pip install --no-cache-dir -r requirements.txt -r requirements-prod.txt

COPY --from=development /app/src /app/src
COPY --from=development /app/main.py /app/main.py
COPY --from=development /app/config.yaml /app/config.yaml #Copy config file

# Set environment variables (best to do this at runtime, see below)
ENV ENVIRONMENT="prodiction" # Set production environment
# ENV DATABASE_URL=... #Set database URL (do not hardcode in Dockerfile)

# Expose the port your app listens on
EXPOSE 8000

# Use a production-ready WSGI server
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "main:app"]
# Assuming i have a Flask/FastAPI app in main.py

