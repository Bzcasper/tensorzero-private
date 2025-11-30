#!/bin/bash
set -e

echo "🚀 Deploying TensorZero UI to Vercel"

cd tensorzero-ui

# Deploy to Vercel
vercel --prod

echo "✅ UI deployed!"
echo "🌐 Access at: https://tensorzero-ui.vercel.app"