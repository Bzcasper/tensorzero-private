import modal
import os
import json
import numpy as np
import torch
import torch.nn as nn
from typing import List

# ====================================================================
# CONFIG & EXACT VARIANTS FROM YOUR tensorzero.toml
# ====================================================================
VARIANTS = [
    "grok_creative",  # 0
    "cerebras_fast",  # 1
    "creative_mode",  # 2
    "structured_mode",  # 3
    "claude_sonnet",  # 4
    "gpt4o",  # 5
]

VOCAB_SIZE = 5000
MAX_SEQ_LEN = 64
MODEL_PATH = "/data/router.onnx"
VARIANT_MAP_PATH = "/data/variant_map.json"

# ====================================================================
# MODAL SETUP
# ====================================================================
app = modal.App("video-generation-router")

volume = modal.Volume.from_name("tensorzero-models", create_if_missing=True)
secrets = [modal.Secret.from_name("tensorzero-secrets")]

# Training image — full PyTorch stack
train_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "torch==2.4.1",
        "torchvision",
        "torchaudio",
        extra_options="--index-url https://download.pytorch.org/whl/cu124",
    )
    .pip_install("clickhouse-connect", "pandas", "scikit-learn", "numpy", "onnx", "faker")
)

# Serving image — lightweight ONNX Runtime
serve_image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "onnxruntime==1.19.2", "fastapi", "uvicorn", "pydantic", "numpy"
)


# ====================================================================
# TRAINING: Real ClickHouse Data → ONNX Model
# ====================================================================
@app.function(image=train_image, volumes={"/data": volume}, secrets=secrets, gpu="T4", timeout=3600)
def train_and_save():
    import clickhouse_connect
    import pandas as pd
    from sklearn.preprocessing import LabelEncoder

    print("Fetching real training data from ClickHouse...")

    client = clickhouse_connect.get_client(
        host=os.getenv("CLICKHOUSE_HOST", "clickhouse"),
        port=8123,
        username="default",
        password=os.getenv("CLICKHOUSE_PASSWORD", "tensorzero"),
    )

    query = """
    SELECT
        i.input,
        i.variant_name,
        f.value as quality_score,
        m.response_time_ms
    FROM JsonInference i
    JOIN FloatMetricFeedback f ON i.id = f.target_id
    JOIN ModelInference m ON i.id = m.inference_id
    WHERE i.function_name = 'script_generator'
      AND f.metric_name = 'video_quality_score'
      AND f.value >= 0
    LIMIT 10000
    """

    try:
        df = client.query_df(query)
    except:
        print("No real data found. Using synthetic fallback...")
        df = None

    if df is None or len(df) < 50:
        print("Generating synthetic training data...")
        from faker import Faker

        fake = Faker()
        data = []
        for _ in range(5000):
            text = fake.sentence(nb_words=12)
            mode = "kid" if "kid" in text.lower() or "fun" in text.lower() else "adult"
            duration = np.random.randint(60, 600)
            label = np.random.choice(VARIANTS)
            data.append({"text": text, "mode": mode, "duration": duration, "variant": label})
        texts = [d["text"] for d in data]
        modes = [1.0 if d["mode"] == "kid" else 0.0 for d in data]
        durations = [d["duration"] / 600.0 for d in data]
        labels = [VARIANTS.index(d["variant"]) for d in data]
    else:
        print(f"Loaded {len(df)} real samples")
        df["input"] = df["input"].apply(json.loads)
        texts = df["input"].apply(lambda x: x.get("project_description", "")).tolist()
        modes = df["input"].apply(lambda x: 1.0 if x.get("content_mode") == "kid" else 0.0).tolist()
        durations = (
            df["input"].apply(lambda x: float(x.get("target_duration", 180)) / 600.0).tolist()
        )
        labels = [VARIANTS.index(v) for v in df["variant_name"] if v in VARIANTS]

    # Simple hash tokenizer
    def tokenize(text):
        return [hash(c) % VOCAB_SIZE for c in text.lower()[:MAX_SEQ_LEN]] + [0] * max(
            0, MAX_SEQ_LEN - len(text)
        )

    input_ids = torch.tensor([tokenize(t) for t in texts], dtype=torch.long)
    metadata = torch.tensor(list(zip(modes, durations)), dtype=torch.float32)
    targets = torch.tensor(labels, dtype=torch.long)

    # Model
    class Router(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(VOCAB_SIZE, 64)
            self.text_proj = nn.Linear(64 * MAX_SEQ_LEN, 128)
            self.meta_proj = nn.Linear(2, 32)
            self.classifier = nn.Sequential(
                nn.Linear(128 + 32, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, len(VARIANTS))
            )

        def forward(self, ids, meta):
            x = self.embed(ids).flatten(1)
            x = torch.relu(self.text_proj(x))
            m = torch.relu(self.meta_proj(meta))
            return torch.softmax(self.classifier(torch.cat([x, m], dim=1)), dim=-1)

    model = Router()
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()

    print("Training router...")
    for epoch in range(30):
        opt.zero_grad()
        out = model(input_ids, metadata)
        loss = loss_fn(out, targets)
        loss.backward()
        opt.step()
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

    # Export
    model.eval()
    dummy_ids = torch.zeros((1, MAX_SEQ_LEN), dtype=torch.long)
    dummy_meta = torch.zeros((1, 2), dtype=torch.float32)

    torch.onnx.export(
        model,
        (dummy_ids, dummy_meta),
        MODEL_PATH,
        input_names=["input_ids", "metadata"],
        output_names=["variant_probs"],
        dynamic_axes={
            "input_ids": {0: "batch"},
            "metadata": {0: "batch"},
            "variant_probs": {0: "batch"},
        },
        opset_version=14,
    )

    # Save variant map
    with open(VARIANT_MAP_PATH, "w") as f:
        json.dump({str(i): v for i, v in enumerate(VARIANTS)}, f)

    print("Model + variant map saved to volume!")

    volume.commit()
    return "Router trained and deployed!"


# ====================================================================
# SERVING: FastAPI + ONNX Runtime
# ====================================================================
from fastapi import FastAPI
from pydantic import BaseModel
import onnxruntime as ort

web_app = FastAPI(title="TensorZero Neural Router")


class RouteRequest(BaseModel):
    project_description: str
    content_mode: str
    target_duration: int


@web_app.post("/route")
async def route(req: RouteRequest):
    if not os.path.exists(MODEL_PATH):
        return {"best_variant": "cerebras_fast", "confidence": 0.5}

    session = ort.InferenceSession(MODEL_PATH)
    with open(VARIANT_MAP_PATH) as f:
        variant_map = json.load(f)

    # Tokenize
    tokens = [hash(c) % VOCAB_SIZE for c in req.project_description.lower()[:MAX_SEQ_LEN]]
    tokens += [0] * (MAX_SEQ_LEN - len(tokens))
    input_ids = np.array([tokens], dtype=np.int64)
    metadata = np.array(
        [[1.0 if req.content_mode == "kid" else 0.0, req.target_duration / 600.0]], dtype=np.float32
    )

    probs = session.run(None, {"input_ids": input_ids, "metadata": metadata})[0][0]
    idx = int(probs.argmax())
    return {
        "best_variant": variant_map.get(str(idx), VARIANTS[0]),
        "confidence": float(probs[idx]),
        "all_probs": {variant_map.get(str(i), f"v{i}"): float(p) for i, p in enumerate(probs)},
    }


@web_app.get("/health")
def health():
    return {"status": "ok", "model_loaded": os.path.exists(MODEL_PATH)}


@app.function(
    image=serve_image, volumes={"/data": volume}, allow_concurrent_inputs=100, keep_warm=2
)
@modal.web_server(8080)
def serve():
    return web_app


# ====================================================================
# RUN TRAINING ONCE
# ====================================================================
@app.local_entrypoint()
def deploy():
    print("Training router with real + synthetic data...")
    train_and_save.remote()
    print("\nRouter deployed at:")
    print("https://your-workspace--video-generation-router-serve.modal.run/route")
