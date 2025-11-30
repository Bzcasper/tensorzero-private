#!/bin/bash
# Helper script to get ngrok ClickHouse URL and run Modal training

echo "🔍 Getting ngrok ClickHouse URL..."

# Get the ngrok tunnel URL for ClickHouse (port 4041 is the inspection interface for the second ngrok)
CLICKHOUSE_URL=$(curl -s http://localhost:4041/api/tunnels | jq -r '.tunnels[0].public_url')

if [ -z "$CLICKHOUSE_URL" ] || [ "$CLICKHOUSE_URL" = "null" ]; then
    echo "❌ Could not get ngrok URL. Make sure ngrok-clickhouse is running."
    echo "Check: docker-compose ps ngrok-clickhouse"
    exit 1
fi

echo "✅ ClickHouse accessible at: $CLICKHOUSE_URL"

# Run Modal training with the ngrok URL
echo "🤖 Starting Modal training..."
modal run router/modal_app.py::train_router_once --clickhouse-url "$CLICKHOUSE_URL"

echo "✅ Training completed!"