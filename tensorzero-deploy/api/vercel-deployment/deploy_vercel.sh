#!/bin/bash
set -e

echo "🚀 Deploying DIY Video API to Vercel..."

# Check if Vercel CLI is installed
if ! command -v vercel &> /dev/null; then
    echo "Installing Vercel CLI..."
    npm install -g vercel
fi

# Add environment variables (one-time setup)
echo "📝 Setting up environment variables..."
vercel env add TENSORZERO_URL production
vercel env add MEDIA_SERVER_URL production
vercel env add CEREBRAS_API_KEY production
vercel env add GROQ_API_KEY production
vercel env add SAMBANOVA_API_KEY production
vercel env add MISTRAL_API_KEY production
vercel env add DEEPSEEK_API_KEY production
vercel env add GEMINI_API_KEY production
vercel env add GROK_API_KEY production
vercel env add OPENROUTER_API_KEY production
vercel env add TOGETHER_API_KEY production
vercel env add IMAGE_AUTH production

# Deploy
echo "🚢 Deploying to production..."
vercel --prod

echo "✅ Deployment complete!"
echo "📍 Your API is live at: https://your-project.vercel.app"