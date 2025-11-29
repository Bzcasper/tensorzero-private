#!/bin/bash
# Vercel Deployment Script for DIY Video Production API

echo "🚀 Deploying DIY Video Production API to Vercel"
echo "==============================================="

# Check if Vercel CLI is installed
if ! command -v vercel &> /dev/null; then
    echo "❌ Vercel CLI not found. Install with: npm i -g vercel"
    exit 1
fi

# Check if logged in
if ! vercel whoami &> /dev/null; then
    echo "❌ Not logged in to Vercel. Run: vercel login"
    exit 1
fi

# Deploy to Vercel
echo "📦 Deploying to Vercel..."
vercel --prod

echo "✅ Deployment complete!"
echo "🌐 Your API will be available at the Vercel URL shown above"
echo ""
echo "📖 API Endpoints:"
echo "  POST /api/generate_video - Generate a video"
echo "  GET  /api/health - Health check"
echo ""
echo "📝 Example request:"
echo 'curl -X POST https://your-app.vercel.app/api/generate_video \''
echo '  -H "Content-Type: application/json" \''
echo '  -d "{\"project_description\": \"Make a paper airplane\", \"content_mode\": \"kid\", \"target_duration\": 180}"'