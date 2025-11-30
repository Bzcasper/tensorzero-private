import os
import json
import torch
import torch.nn as nn
import torch.onnx
import clickhouse_connect
import pandas as pd
import numpy as np
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# --- CONFIG ---
CLICKHOUSE_HOST = os.getenv("CLICKHOUSE_HOST", "localhost")
CLICKHOUSE_PASSWORD = os.getenv("CLICKHOUSE_PASSWORD", "tensorzero")
VOCAB_SIZE = 2000
MAX_SEQ_LEN = 32
HIDDEN_DIM = 64


class ContextualRouter(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim, metadata_dim):
        super().__init__()
        # Text Branch (simplified for ONNX compatibility)
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.text_proj = nn.Linear(embed_dim * MAX_SEQ_LEN, hidden_dim)

        # Metadata Branch (content_mode, duration)
        self.meta_fc = nn.Linear(metadata_dim, 16)

        # Fusion
        self.fc_final = nn.Linear(hidden_dim + 16, output_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_ids, metadata):
        # Text processing
        embedded = self.embedding(input_ids).flatten(1)
        text_feat = torch.relu(self.text_proj(embedded))

        # Metadata processing
        meta_feat = torch.relu(self.meta_fc(metadata))

        # Concatenate
        combined = torch.cat((text_feat, meta_feat), dim=1)
        logits = self.fc_final(combined)
        return self.softmax(logits)


def fetch_training_data(clickhouse_url=None):
    # Use provided URL or construct from environment
    if clickhouse_url:
        # For ngrok URLs, we need to handle authentication differently
        # Assuming the URL includes auth or we get it from env
        client = clickhouse_connect.get_client(
            host=clickhouse_url.replace("http://", "").replace("https://", "").split(":")[0],
            username="default",
            password=os.getenv("CLICKHOUSE_PASSWORD", CLICKHOUSE_PASSWORD),
            port=int(clickhouse_url.split(":")[-1])
            if ":" in clickhouse_url.split("/")[-1]
            else 8123,
        )
    else:
        client = clickhouse_connect.get_client(
            host=CLICKHOUSE_HOST, username="default", password=CLICKHOUSE_PASSWORD, port=8123
        )

    # Query: Join Inputs (JsonInference) with Outcomes (Feedback + Usage)
    query = """
    SELECT
        i.input,
        i.variant_name,
        f.value as quality_score,
        m.output_tokens,
        m.response_time_ms
    FROM JsonInference i
    JOIN FloatMetricFeedback f ON i.id = f.target_id
    JOIN ModelInference m ON i.id = m.inference_id
    WHERE i.function_name = 'script_generator'
      AND f.metric_name = 'video_quality_score'
    """

    df = client.query_df(query)
    return df


def preprocess_data(df):
    # 1. Parse JSON Input
    df["parsed_input"] = df["input"].apply(json.loads)
    df["desc"] = df["parsed_input"].apply(lambda x: x.get("project_description", ""))
    df["mode"] = df["parsed_input"].apply(lambda x: 1.0 if x.get("content_mode") == "kid" else 0.0)
    df["duration"] = df["parsed_input"].apply(
        lambda x: float(x.get("target_duration", 180)) / 600.0
    )  # Normalize

    # 2. Calculate Utility Score (Reward)
    # Utility = Quality^2 / log(Latency) -> Penalize slow models, heavily reward high quality
    df["utility"] = (df["quality_score"] ** 2) / np.log1p(df["response_time_ms"])

    # 3. Labeling: For a given context, which variant had the highest utility?
    # This is a simplification. For true off-policy RL, we'd need counterfactual estimation.
    # Here, we treat high-utility samples as "correct" labels.
    top_performers = df[df["utility"] > df["utility"].quantile(0.7)]

    return top_performers


def create_training_dataset_from_clickhouse(clickhouse_url=None):
    """Pull comprehensive training data from TensorZero ClickHouse"""
    print("📊 Fetching training data from ClickHouse...")

    # Override environment variables if URL provided
    if clickhouse_url:
        # Parse URL to extract host and port
        # Expected format: http://host:port or https://host:port
        if clickhouse_url.startswith("http"):
            # Extract host and port from URL
            import re

            match = re.match(r"https?://([^:/]+)(?::(\d+))?", clickhouse_url)
            if match:
                host = match.group(1)
                port = match.group(2) or "8123"
                os.environ["CLICKHOUSE_HOST"] = host
                # Note: Port is hardcoded in the connection, so we'll use the URL directly

    try:
        df = fetch_training_data(clickhouse_url)
        if len(df) == 0:
            print("⚠️ No training data found")
            return []

        df_clean = preprocess_data(df)
        print(f"✅ Processed {len(df_clean)} training samples")

        # Convert to training format
        training_data = []
        for _, row in df_clean.iterrows():
            # Extract features
            features = {
                "text_length": len(row["desc"]),
                "word_count": len(row["desc"].split()),
                "is_kid_mode": row["mode"],
                "normalized_duration": row["duration"],
                "utility_score": row["utility"],
            }

            training_data.append(
                {
                    "text": row["desc"],
                    "features": features,
                    "variant": row["variant_name"],
                    "utility": row["utility"],
                }
            )

        return training_data

    except Exception as e:
        print(f"❌ Error fetching training data: {e}")
        return []


def train_contextual_router(training_data):
    """Train the contextual router model"""
    print("🤖 Training contextual router...")

    if len(training_data) < 10:
        print("⚠️ Insufficient training data")
        return None

    # Prepare data
    texts = [sample["text"] for sample in training_data]
    variants = [sample["variant"] for sample in training_data]

    # Simple tokenizer for Modal compatibility
    def simple_tokenize(text):
        # Basic character-level tokenization (simplified)
        tokens = [ord(c) % VOCAB_SIZE for c in text[:MAX_SEQ_LEN]]
        return tokens + [0] * (MAX_SEQ_LEN - len(tokens))

    # Encode variants
    le = LabelEncoder()
    labels = le.fit_transform(variants)
    num_classes = len(le.classes_)

    # Prepare tensors
    input_ids = np.array([simple_tokenize(t) for t in texts], dtype=np.int64)

    # Extract metadata features
    metadata = np.array(
        [
            [sample["features"]["is_kid_mode"], sample["features"]["normalized_duration"]]
            for sample in training_data
        ],
        dtype=np.float32,
    )

    X_text = torch.tensor(input_ids)
    X_meta = torch.tensor(metadata)
    y = torch.tensor(labels, dtype=torch.long)

    # Train model
    model = ContextualRouter(VOCAB_SIZE, 64, HIDDEN_DIM, num_classes, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    for epoch in range(50):
        optimizer.zero_grad()
        output = model(X_text, X_meta)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

    return model


def export_model(model, output_path):
    """Export trained model to ONNX"""
    print("💾 Exporting model to ONNX...")

    dummy_text = torch.randint(0, VOCAB_SIZE, (1, MAX_SEQ_LEN))
    dummy_meta = torch.randn(1, 2)

    torch.onnx.export(
        model,
        (dummy_text, dummy_meta),
        output_path,
        input_names=["input_ids", "metadata"],
        output_names=["variant_probs"],
        dynamic_axes={
            "input_ids": {0: "batch"},
            "metadata": {0: "batch"},
            "variant_probs": {0: "batch"},
        },
    )

    print(f"✅ Model exported to {output_path}")


def load_synthetic_training_data():
    """Load training data from synthetic JSON file."""
    try:
        with open("training_data.json", "r") as f:
            data = json.load(f)
        training_data = []
        for sample in data:
            features = {
                "text_length": len(sample["input"]["project_description"]),
                "word_count": len(sample["input"]["project_description"].split()),
                "is_kid_mode": 1.0 if sample["input"]["content_mode"] == "kid" else 0.0,
                "normalized_duration": sample["input"]["target_duration"] / 600.0,
                "utility_score": sample["utility_score"],
            }
            training_data.append(
                {
                    "text": sample["input"]["project_description"],
                    "features": features,
                    "variant": sample["variant_name"],
                    "utility": sample["utility_score"],
                }
            )
        return training_data
    except FileNotFoundError:
        return []


if __name__ == "__main__":
    # Try synthetic data first, then ClickHouse
    training_data = load_synthetic_training_data()
    if not training_data:
        training_data = create_training_dataset_from_clickhouse()
    if training_data:
        model = train_contextual_router(training_data)
        if model:
            export_model(model, "router.onnx")
    else:
        print("No training data available - using dummy export")
        # Fallback to dummy model for testing
        dummy_model = ContextualRouter(VOCAB_SIZE, 64, HIDDEN_DIM, 5, 2)  # 5 variants now
        export_model(dummy_model, "router.onnx")
