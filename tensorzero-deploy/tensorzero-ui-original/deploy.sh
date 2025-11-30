#!/bin/bash
set -e

echo "🚀 Deploying Original TensorZero UI to Vercel"

cd tensorzero-ui-original

# Deploy to Vercel with Docker
vercel --prod

echo "✅ Original UI deployed!"
echo "🌐 Access at: https://tensorzero-ui-original.vercel.app"