#!/bin/bash
set -e

echo "🚀 Deploying TensorZero Stack to Railway (Cheapest Option)"

# Check if Railway CLI is installed
if ! command -v railway &> /dev/null; then
    echo "Installing Railway CLI..."
    npm install -g @railway/cli
fi

# Login to Railway
echo "Logging into Railway..."
railway login

# Link to project (create if doesn't exist)
echo "Setting up Railway project..."
railway project create tensorzero-stack --team personal || railway link

# Deploy services
echo "Deploying services..."

# 1. ClickHouse (external - free tier)
echo "ClickHouse: Use https://clickhouse.cloud (free tier)"
echo "Create free ClickHouse Cloud instance and get connection URL"

# 2. TensorZero Gateway
echo "Deploying TensorZero Gateway..."
railway service create tensorzero-gateway
railway service link tensorzero-gateway

# Set environment variables for gateway
railway variables set TENSORZERO_CLICKHOUSE_URL=your-clickhouse-url
railway variables set CEREBRAS_API_KEY=your-key
railway variables set GROQ_API_KEY=your-key
# ... add all API keys

# Deploy from GitHub
railway service connect github
# Select your tensorzero-private repo
# Set root directory to tensorzero-deploy
# Railway will use docker-compose.yml

# 3. UI (optional - deploy to Vercel for free)
echo "UI: Deploy to Vercel (free tier)"
echo "Run: cd tensorzero-deploy && vercel --prod"

echo "✅ Deployment initiated!"
echo "Check Railway dashboard for status"
echo "Total cost: ~$0-5/month (Railway free tier + ClickHouse free)"