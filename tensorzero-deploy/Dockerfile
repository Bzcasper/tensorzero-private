FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install uv

# Copy project files
COPY pyproject.toml .
COPY tensorzero_client.py media_server_client.py agent.py ./

# Install dependencies
RUN uv sync --no-install-project

# Run the agent
CMD ["uv", "run", "python", "agent.py"]