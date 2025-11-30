#!/bin/bash
set -e

echo "🚀 Deploying TensorZero Video API to PythonAnywhere"

# This deploys only the Python video API to PythonAnywhere
# Gateway and ClickHouse remain external

# 1. Upload code to PythonAnywhere
echo "Upload tensorzero-deploy/ to PythonAnywhere"
echo "Use PythonAnywhere file manager or git clone"

# 2. Install dependencies
echo "In PythonAnywhere bash console:"
echo "cd tensorzero-deploy"
echo "python -m venv venv"
echo "source venv/bin/activate"
echo "pip install -r requirements.txt"

# 3. Configure environment
echo "Create .env file with:"
cat << 'EOF'
TENSORZERO_BASE_URL=https://tensorzero-gateway.onrender.com
MEDIA_SERVER_URL=https://2281a5a294754c19f8c9e2df0be013fb-bobby-casper-4235.aiagentsaz.com
IMAGE_AUTH=Bearer 80408040
REDIS_URL=redis://your-redis-instance
CLICKHOUSE_URL=https://your-clickhouse-instance
EOF

# 4. Configure web app
echo "In PythonAnywhere Web tab:"
echo "- Source code: /home/yourusername/tensorzero-deploy"
echo "- Working directory: /home/yourusername/tensorzero-deploy"
echo "- Virtualenv: /home/yourusername/tensorzero-deploy/venv"
echo "- WSGI file: (create) /home/yourusername/tensorzero-deploy/wsgi.py"

# 5. Create WSGI file
cat << 'EOF' > wsgi.py
import sys
import os

# Add project directory to path
sys.path.insert(0, os.path.dirname(__file__))

# Activate virtualenv
activate_this = '/home/yourusername/tensorzero-deploy/venv/bin/activate_this.py'
with open(activate_this) as file_:
    exec(file_.read(), dict(__file__=activate_this))

# Import app
from video_api import app as application
EOF

# 6. Reload web app
echo "Reload PythonAnywhere web app"

echo "✅ Deployment complete!"
echo "🌐 Your API will be at: https://yourusername.pythonanywhere.com"
echo "💰 Cost: $0/month (free tier)"