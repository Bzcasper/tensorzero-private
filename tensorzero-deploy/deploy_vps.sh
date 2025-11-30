#!/bin/bash
set -e

echo "🚀 Deploying TensorZero Stack to Cheap VPS (DigitalOcean/Linode)"

# Update system
echo "Updating system..."
sudo apt update && sudo apt upgrade -y

# Install Docker and Docker Compose
echo "Installing Docker..."
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo apt install -y docker-compose-plugin

# Clone repo (if not already)
if [ ! -d "tensorzero-deploy" ]; then
    echo "Cloning repository..."
    git clone https://github.com/Bzcasper/tensorzero-private.git
    cd tensorzero-private/tensorzero-deploy
else
    cd tensorzero-deploy
    git pull
fi

# Create .env file
echo "Creating .env file..."
cat > .env << EOF
# Database
CLICKHOUSE_PASSWORD=your_secure_password
POSTGRES_PASSWORD=your_secure_password

# API Keys
CEREBRAS_API_KEY=your_key
GROQ_API_KEY=your_key
SAMBANOVA_API_KEY=your_key
MISTRAL_API_KEY=your_key
DEEPSEEK_API_KEY=your_key
GEMINI_API_KEY=your_key
GROK_API_KEY=your_key
OPENROUTER_API_KEY=your_key
TOGETHER_API_KEY=your_key

# External Services
MEDIA_SERVER_URL=https://2281a5a294754c19f8c9e2df0be013fb-bobby-casper-4235.aiagentsaz.com
IMAGE_AUTH=Bearer 80408040
EOF

echo "Please edit .env with your actual API keys"
read -p "Press enter when .env is configured..."

# Start services
echo "Starting TensorZero stack (Gateway + ClickHouse + UI)..."
docker compose up -d tensorzero clickhouse ui

# Wait for services
echo "Waiting for services to start..."
sleep 60

# Check health
echo "Checking service health..."
curl -f http://localhost:3000/health && echo "✅ TensorZero Gateway healthy"
curl -f http://localhost:8123/ping && echo "✅ ClickHouse healthy"
curl -f http://localhost:4000 && echo "✅ UI healthy"

# Setup nginx reverse proxy
echo "Setting up Nginx reverse proxy..."
sudo apt install -y nginx certbot python3-certbot-nginx

# Create nginx config
sudo tee /etc/nginx/sites-available/tensorzero << EOF
server {
    listen 80;
    server_name your-domain.com;

    # TensorZero Gateway
    location /inference {
        proxy_pass http://localhost:3000;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
    }

    # Video API
    location /api/ {
        proxy_pass http://localhost:8000;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
    }

    # UI Dashboard
    location / {
        proxy_pass http://localhost:4000;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
    }
}
EOF

sudo ln -s /etc/nginx/sites-available/tensorzero /etc/nginx/sites-enabled/
sudo nginx -t && sudo systemctl reload nginx

# Setup SSL (optional)
echo "To add SSL: sudo certbot --nginx -d your-domain.com"

echo "✅ Deployment complete!"
echo "🌐 Access your services:"
echo "  - Gateway: http://your-domain.com/inference"
echo "  - API: http://your-domain.com/api/"
echo "  - UI: http://your-domain.com"
echo ""
echo "💰 Total cost: $5/month (Linode VPS) - includes all TensorZero components"
echo ""
echo "📋 What's running:"
echo "  - TensorZero Gateway: http://your-domain.com/inference"
echo "  - ClickHouse: http://your-domain.com:8123"
echo "  - UI Dashboard: http://your-domain.com"
echo "  - Video API: Already on Vercel"
echo "🔧 Maintenance: Minimal (Docker handles updates)"