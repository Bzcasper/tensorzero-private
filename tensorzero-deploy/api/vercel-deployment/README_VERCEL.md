# Vercel Deployment Guide

## Prerequisites

- Vercel account
- All API keys configured

## Quick Deploy

### Option 1: CLI Deployment

```bash
# Install Vercel CLI
npm install -g vercel

# Login
vercel login

# Deploy
vercel --prod
```

### Option 2: GitHub Integration

1. Push code to GitHub
2. Connect repository to Vercel
3. Add environment variables in Vercel dashboard
4. Deploy automatically on push

## Environment Variables

Add these in Vercel Project Settings → Environment Variables:

```
TENSORZERO_URL=http://your-tensorzero-gateway:3000
MEDIA_SERVER_URL=https://2281a5a294754c19f8c9e2df0be013fb-bobby-casper-4235.aiagentsaz.com
CEREBRAS_API_KEY=your_key
GROQ_API_KEY=your_key
SAMBANOVA_API_KEY=your_key
MISTRAL_API_KEY=your_key
DEEPSEEK_API_KEY=your_key
GEMINI_API_KEY=your_key
GROK_API_KEY=your_key
OPENROUTER_API_KEY=your_key
TOGETHER_API_KEY=your_key
IMAGE_AUTH=Bearer 80408040
```

## Testing

```bash
# Health check
curl https://your-project.vercel.app/health

# Generate video
curl -X POST https://your-project.vercel.app/api/generate_video \
  -H "Content-Type: application/json" \
  -d '{
    "project_description": "make a paper airplane",
    "content_mode": "kid",
    "target_duration": 180
  }'
```

## Limitations

- **Function Timeout**: 60s on Hobby, 300s on Pro
- **Package Size**: Must be <250MB unzipped
- **Memory**: 1GB on Hobby, 3GB on Pro
- **Concurrency**: Limited on Hobby tier

## Production Considerations

1. **Job Queue**: Use Redis/Upstash for long-running jobs
2. **Status Tracking**: Store job status in database
3. **Webhooks**: Notify completion via webhooks
4. **Caching**: Cache TensorZero responses
