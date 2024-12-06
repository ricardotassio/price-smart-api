```markdown
# Apache Airflow with Astro CLI - Local Setup Guide

This guide provides the steps to set up and run Apache Airflow locally using the Astro CLI.

---
## Prerequisites

Before starting, ensure that you have the following installed on your system:

- **Docker Desktop**: [Install Docker](https://docs.docker.com/get-docker/)
- **Homebrew**: [Install Homebrew](https://brew.sh/)

---

## Installation Steps

### 1. Install the Astro CLI

To install Astro CLI, run the following command in your terminal:
```bash
brew install astro
```

---

## Running Apache Airflow Locally

Once the Astro CLI is installed, you can start your Airflow instance.

1. Navigate to the directory where your Airflow project is located.
2. Start Airflow by running:
   ```bash
   astro dev start
   ```

3. Once the environment is up, you can access the following:

   - **Airflow Webserver**: [http://localhost:8080](http://localhost:8080)
   - **Postgres Database**: `localhost:5432/postgres`

---

## Default Credentials

Use these credentials for initial access:

- **Airflow UI**:
  - Username: `admin`
  - Password: `admin`

- **Postgres Database**:
  - Username: `postgres`
  - Password: `postgres`

---

## Stopping Airflow

To stop the Airflow environment, use the following command:
```bash
astro dev stop
```

---

## Troubleshooting

If you encounter any issues, please refer to the [Astro CLI documentation](https://docs.astronomer.io/astro/cli).

---
```