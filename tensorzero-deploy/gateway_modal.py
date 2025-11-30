import modal

app = modal.App("tensorzero-gateway")

# Use the existing Dockerfile
gateway_image = modal.Image.from_dockerfile("Dockerfile.gateway")

# Secrets for API keys
secrets = modal.Secret.from_dict(
    {
        "TENSORZERO_CLICKHOUSE_URL": "https://your-clickhouse-instance.clickhouse.cloud:8443/tensorzero",
        "CEREBRAS_API_KEY": "your_key",
        "GROQ_API_KEY": "your_key",
        "SAMBANOVA_API_KEY": "your_key",
        "MISTRAL_API_KEY": "your_key",
        "DEEPSEEK_API_KEY": "your_key",
        "GEMINI_API_KEY": "your_key",
        "GROK_API_KEY": "your_key",
        "OPENROUTER_API_KEY": "your_key",
        "TOGETHER_API_KEY": "your_key",
    }
)


@app.function(
    image=gateway_image,
    secrets=[secrets],
    cpu=0.5,
    memory=512,
    keep_warm=1,
)
@modal.web_server(3000, startup_timeout=60)
def serve_gateway():
    import subprocess
    import time

    # Start the TensorZero gateway
    process = subprocess.Popen(
        ["tensorzero-gateway", "--config-file", "/app/config/tensorzero.toml"]
    )

    # Keep the function running
    try:
        while True:
            time.sleep(1)
            if process.poll() is not None:
                break
    except KeyboardInterrupt:
        process.terminate()
        process.wait()


@app.local_entrypoint()
def main():
    print("TensorZero Gateway Modal App")
    print("Run: modal deploy gateway_modal.py")
