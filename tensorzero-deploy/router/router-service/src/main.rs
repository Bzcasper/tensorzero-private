use axum::{routing::post, Json, Router};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tract_onnx::prelude::*;
use router_core::NeuralRouter;

#[derive(Clone)]
struct AppState {
    router: Arc<NeuralRouter>,
}

#[derive(Deserialize)]
struct RouteRequest {
    input_ids: Vec<i64>, // Tokenized by client or intermediate layer
    metadata: Vec<f32>,  // [is_kid, normalized_duration]
}

#[derive(Serialize)]
struct RouteResponse {
    best_variant: String,
    confidence: f32,
}

async fn route_handler(
    axum::extract::State(state): axum::extract::State<AppState>,
    Json(payload): Json<RouteRequest>,
) -> Json<RouteResponse> {
    // 1. Prepare Tensors
    let input_ids = tract_ndarray::Array2::<i64>::from_shape_vec(
        (1, payload.input_ids.len()),
        payload.input_ids,
    ).unwrap();
    let meta = tract_ndarray::Array2::<f32>::from_shape_vec(
        (1, 2),
        payload.metadata,
    ).unwrap();

    // 2. Inference
    let result = state.router.route_with_features(&input_ids.view(), &meta.view()).unwrap();
    let scores = result.view();

    // 3. Argmax
    let mut best_idx = 0;
    let mut max_score = -1.0f32;
    for (i, &score) in scores.iter().enumerate() {
        if score > max_score {
            max_score = score;
            best_idx = i;
        }
    }

    // For now, hardcode variant names - in production, load from variant_map.json
    let variants = vec!["claude_sonnet", "gpt4o", "cerebras_fast", "grok_creative"];

    Json(RouteResponse {
        best_variant: variants[best_idx].to_string(),
        confidence: max_score,
    })
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Read paths from Env or default to local dev paths
    let model_path = std::env::var("MODEL_PATH").unwrap_or_else(|_| "router.onnx".to_string());
    let tokenizer_path = std::env::var("TOKENIZER_PATH").unwrap_or_else(|_| "tokenizer.json".to_string());

    println!("Loading model from: {}", model_path);

    let router = Arc::new(NeuralRouter::new(&model_path, &tokenizer_path)?);

    let state = AppState { router };

    let app = Router::new()
        .route("/route", post(route_handler))
        .route("/health", axum::routing::get(|| async { "OK" }))
        .with_state(state);

    // Modal web_server expects us to listen on the specific port
    let addr = "0.0.0.0:8080";
    let listener = tokio::net::TcpListener::bind(addr).await?;
    println!("Router service listening on {}", addr);

    axum::serve(listener, app).await?;
    Ok(())
}