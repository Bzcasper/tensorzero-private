# DIY Video Production Agent

A complete automated video production system using **TensorZero** for AI orchestration, **Fast Agent** for workflow management, and **MCP** for Media Server integration.

## 🏗️ Architecture

- **TensorZero**: Handles all AI prompts and model routing (scripts, prompt enhancement, quality evaluation)
- **Fast Agent**: Orchestrates the workflow logic (no prompts in agent code)
- **MCP Server**: Standardized interface to Media Server API with progress notifications
- **Media Server**: Provides video/audio generation and processing APIs

## 🚀 Quick Start

### Prerequisites

- **Python 3.12+**
- **uv** package manager (`curl -LsSf https://astral.sh/uv/install.sh | sh`)
- API keys for AI providers (TensorZero models)
- Access to Media Server API

### Setup

1. **Install dependencies:**

   ```bash
   uv sync
   ```

2. **Configure environment:**

   ```bash
   cp .env.example .env
   # Edit .env with your API keys (see .env.example for required variables)
   ```

3. **Verify MCP server:**
   ```bash
   uv run python test_mcp.py
   ```

### Running the Pipeline

#### Option A: Direct Python Execution

```bash
# Run the complete pipeline
uv run python run_diy_video.py "how to make paper airplanes that fly far"

# Or run the agent directly
uv run python agent.py
```

#### Option B: Docker Compose (Full Stack)

```bash
# Start all services (TensorZero + Agent + MCP)
docker-compose up --build

# In another terminal, trigger the pipeline
curl -X POST http://localhost:8000/trigger \
  -H "Content-Type: application/json" \
  -d '{"project": "paper airplane tutorial"}'
```

#### Option C: Development Mode

```bash
# Run TensorZero gateway only
docker-compose up tensorzero

# Run agent locally with MCP server
uv run python agent.py
```

### Expected Output

```
🎬 Starting DIY Video Production Pipeline
📝 Project: how to make paper airplanes that fly far
==================================================
📝 Generating script...
✅ Script: How to Make Paper Airplanes That Fly Far
🎨 Generating images...
🏭 Generating all images...
✅ Generated 7 images total
🎬 Creating video segments...
🔗 Merging videos...
⭐ Evaluating quality...
✅ Quality Score: 8.5/10
📹 Final Video: video_12345

==================================================
🎉 PIPELINE COMPLETE
==================================================
Video ID: video_12345
Quality Score: 8.5/10
Passed: ✅
Feedback: Excellent educational value and clear instructions...
==================================================
```

### Architecture Flow

1. **TensorZero Script Generation** → Creates structured DIY script
2. **TensorZero Prompt Enhancement** → Improves image generation prompts
3. **Gemini Image Generation** → Creates visuals via API
4. **MCP Media Server** → Uploads images, generates TTS, creates video segments
5. **MCP Video Assembly** → Merges segments into final video
6. **TensorZero Quality Evaluation** → Scores final result and provides feedback

### Configuration Files

- **`config/tensorzero.toml`**: Model routing and function definitions
- **`fastagent.config.yaml`**: MCP server configuration
- **`templates/`**: MiniJinja templates for AI prompts (no prompts in code!)
- **`.env`**: API keys and environment variables

2. **Configure environment:**

   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

3. **Run with Docker:**

   ```bash
   docker-compose up
   ```

4. **Or run locally:**
   ```bash
   uv run python agent.py
   ```

## 📋 Prerequisites

- **Python 3.12+**
- **uv** package manager
- API keys for AI providers (see `.env.example`)
- Access to Media Server API

## 🔧 Configuration

### TensorZero Functions

The system uses three TensorZero functions:

1. **`script_generator`**: Creates structured DIY video scripts
   - Variants: `grok_creative`, `cerebras_fast`
   - Experimentation: Static weights (70% Grok, 30% Cerebras)

2. **`prompt_enhancer`**: Enhances basic prompts for better image generation
   - Variants: `mistral_balanced`, `deepseek_efficient`, `cerebras_creative`
   - Experimentation: Best-of-N sampling with 3 candidates

3. **`quality_evaluator`**: Evaluates final video quality
   - Variant: `gemini_pro`
   - No experimentation (consistent evaluation needed)

### Model Routing

```toml
# Fast inference (sub-10s responses)
[models.fast_llm]
routing = ["cerebras", "groq", "openrouter_fast"]

# Heavy reasoning (complex analysis)
[models.smart_llm]
routing = ["sambanova", "deepseek", "mistral", "openrouter_smart"]

# Creative tasks (prompt enhancement)
[models.creative_llm]
routing = ["xai", "google", "together"]
```

### MCP Configuration

The Media Server MCP server provides standardized tools:

```yaml
# fastagent.config.yaml
mcp:
  servers:
    media_server:
      command: "uv"
      args: ["run", "python", "media_server_mcp.py"]
      env:
        IMAGE_AUTH: "${IMAGE_AUTH}"
        MEDIA_SERVER_URL: "${MEDIA_SERVER_URL}"
```

## 🎬 Workflow

The Fast Agent orchestrates this pipeline:

1. **Script Generation** → TensorZero creates video script
2. **Prompt Enhancement** → TensorZero improves image prompts
3. **Image Generation** → Gemini API creates visuals + MCP upload
4. **Video Assembly** → MCP tools generate video segments with TTS/captions
5. **Quality Evaluation** → TensorZero assesses final result

## 📊 Monitoring & Feedback

- **Logging**: Comprehensive logging to `diy_video_production.log`
- **MCP Progress**: Real-time progress notifications during video processing
- **TensorZero Feedback**: Quality scores sent back for model improvement
- **Health Checks**: Docker health checks for service monitoring

## 🛠️ Development

### Quality Feedback System

The system includes a comprehensive quality feedback loop that improves video generation over time:

#### **TensorZero Functions**

- **`video_evaluator`**: Advanced quality assessment with vision analysis
  - Evaluates content, production, audience appropriateness, and technical execution
  - Returns detailed scores and improvement suggestions
  - Uses Gemini Pro Vision for visual analysis

#### **Evaluator-Optimizer Workflow**

```python
@fast.evaluator_optimizer(
    name="quality_assured_video",
    generator="diy_video_pipeline",
    evaluator="video_quality_judge",
    min_rating="GOOD",  # Requires score >= 7
    max_refinements=3
)
```

- Automatically regenerates videos that don't meet quality standards
- Avoids expensive full regeneration by targeting specific improvements
- Maps numeric scores to EXCELLENT/GOOD/FAIR/POOR ratings

#### **Metrics Collection**

The system tracks comprehensive metrics in TensorZero:

- `video_quality_score`: Overall quality (1-10)
- `video_generation_cost`: API costs in USD
- `video_generation_time`: Total generation time
- `user_satisfaction`: Boolean user feedback
- `video_completion_rate`: Success/failure tracking

#### **A/B Testing Strategy**

Multiple experimentation variants for continuous improvement:

**Script Generation A/B Tests:**

- `grok_creative`: High creativity, detailed scripts
- `cerebras_fast`: Fast, structured generation
- `creative_mode`: Maximum creativity
- `structured_mode`: Maximum structure

**Prompt Enhancement A/B Tests:**

- `mistral_balanced`: Balanced enhancement
- `deepseek_efficient`: Cost-effective
- `cerebras_creative`: Highly creative
- `conservative_style`: Safe, reliable
- `aggressive_style`: Bold, experimental

**Quality Evaluation A/B Tests:**

- `gemini_pro_vision`: Advanced vision analysis
- `grok_vision`: Alternative vision model

### Running Tests

```bash
uv run pytest
```

### Code Quality

```bash
uv run ruff check . --fix
uv run mypy .
```

### Analytics & Monitoring

```bash
# View quality trends
uv run python -c "
import asyncio
from analytics_queries import run_quality_dashboard
asyncio.run(run_quality_dashboard())
"

# Check A/B test results
uv run python -c "
from analytics_queries import analyze_ab_tests
analyze_ab_tests()
"
```

### Testing MCP Server

```bash
uv run python test_mcp.py
```

### Code Quality

```bash
uv run ruff check . --fix
uv run mypy .
```

### Adding New Functions

1. **Define in `config/tensorzero.toml`:**

   ```toml
   [functions.new_function]
   type = "json"
   system_template = "templates/new_function_system.minijinja"
   user_schema = "templates/new_function_user_schema.json"
   ```

2. **Create templates** in `templates/` directory

3. **Add agent method** in `agent.py`:
   ```python
   @fast.agent(name="new_function", servers=["media_server"])
   async def new_function(agent, param: str) -> Any:
       result = await agent.call_tool("tool_name", {"param": param})
       return result
   ```

## 🔒 Security

- **No prompts in agent code** - All prompts stored in TensorZero templates
- **Environment variables** for API keys
- **Input validation** via JSON schemas
- **MCP authentication** via Bearer tokens

## 📈 Performance

- **Async operations** throughout the pipeline
- **MCP job polling** with progress notifications
- **Parallel processing** where possible
- **Efficient model routing** via TensorZero

## 🤝 Contributing

1. Follow the established patterns (prompts in TensorZero, logic in agents)
2. Add comprehensive logging
3. Include error handling
4. Update documentation

## 🚀 Production Deployment

### Prerequisites

- **Docker & Docker Compose** (latest versions)
- **4GB+ RAM** (recommended 8GB+)
- **10GB+ disk space** for databases and logs
- **API Keys** for all LLM providers (see `.env.example`)

### Quick Start

```bash
# 1. Clone and setup
git clone <repository>
cd diy-video-producer
cp .env.example .env
# Edit .env with your API keys

# 2. Start all services
docker-compose up --build -d

# 3. Check health
curl http://localhost/health

# 4. Generate a video
curl -X POST http://localhost/api/generate_video \
  -H "Content-Type: application/json" \
  -d '{"project_description": "how to make paper airplanes", "content_mode": "kid"}'

# 5. Monitor status
curl http://localhost/api/status/{job_id}
```

### Service Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Nginx Proxy   │    │  Video API      │    │  Background     │
│   (Port 80)     │────│  (FastAPI)      │────│  Worker         │
│                 │    │  (Port 8000)    │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────────┐    ┌─────────────────┐
                    │  TensorZero     │    │  ClickHouse     │
                    │  Gateway        │    │  (Metrics)      │
                    │  (Port 3000)    │    │  (Port 8123)    │
                    └─────────────────┘    └─────────────────┘
                             │                       │
                    ┌─────────────────┐    ┌─────────────────┐
                    │  Postgres       │    │  Redis          │
                    │  (Observability)│    │  (Job Queue)    │
                    │  (Port 5432)    │    │  (Port 6379)    │
                    └─────────────────┘    └─────────────────┘
```

### Scaling Strategy

#### Horizontal Scaling

```bash
# Add more API workers
docker-compose up --scale video-api=3

# Add more background workers
docker-compose up --scale worker=5
```

#### Rate Limiting

- **API Requests**: 10/second (configurable in nginx.conf)
- **Video Generation**: 2/minute per user
- **Background Jobs**: Redis-based queuing

#### Caching Strategy

- **Redis TTL**: 1 hour for results
- **Media Server**: File caching for generated assets
- **TensorZero**: Model response caching

### Monitoring & Observability

#### Health Checks

```bash
# Individual service health
curl http://localhost/health                    # Nginx
curl http://localhost:8000/health              # Video API
curl http://localhost:3000/health              # TensorZero

# Database health
docker-compose exec clickhouse clickhouse-client --query "SELECT 1"
docker-compose exec redis redis-cli ping
```

#### Metrics Dashboard

```bash
# Real-time metrics (run ClickHouse queries from monitoring_queries.sql)
docker-compose exec clickhouse clickhouse-client --query "
SELECT formatDateTime(now(), '%Y-%m-%d %H:%i:%s') as time,
       (SELECT avg(value) FROM tensorzero.metrics
        WHERE metric_name = 'video_quality_score'
        AND timestamp >= now() - INTERVAL 1 HOUR) as quality_1h,
       (SELECT count(*) FROM tensorzero.metrics
        WHERE metric_name = 'video_quality_score'
        AND timestamp >= now() - INTERVAL 1 HOUR) as videos_1h
"
```

#### Alert Conditions

- **Quality Degradation**: >10% drop in average quality score
- **High Error Rate**: >15% of requests failing
- **Slow Performance**: Average generation time >15 minutes
- **Cost Spike**: >50% increase in daily costs

### Cost Analysis & Optimization

#### Current Cost Breakdown (per video)

| Component          | Model             | Cost           | Notes                  |
| ------------------ | ----------------- | -------------- | ---------------------- |
| Script Generation  | Claude-3.5-Sonnet | $0.02-0.05     | 500-1000 tokens        |
| Prompt Enhancement | Mixtral/Grok      | $0.01-0.03     | 200-400 tokens         |
| Image Generation   | Gemini API        | $0.005-0.01    | Per image              |
| Video Processing   | Media Server      | $0.10-0.20     | TTS + rendering        |
| Quality Evaluation | Gemini Pro        | $0.02-0.04     | 300-600 tokens         |
| **Total**          |                   | **$0.16-0.32** | **Per 3-minute video** |

#### Cost Optimization Strategies

1. **Model Selection via A/B Testing**

   ```sql
   -- Identify best model for quality/cost ratio
   SELECT variant_name,
          avg(quality_score) / avg(cost) as quality_per_dollar,
          avg(generation_time) as speed
   FROM experimentation_metrics
   GROUP BY variant_name
   ORDER BY quality_per_dollar DESC;
   ```

2. **Caching & Reuse**
   - Cache successful prompts for similar projects
   - Reuse generated images across similar tutorials
   - Cache TTS audio for common phrases

3. **Quality Gates**
   - Stop regeneration after 2 attempts if quality < 5.0
   - Skip expensive evaluation for obviously poor results
   - Batch similar requests to reduce API calls

4. **Dynamic Model Selection**
   ```python
   # Route based on complexity and budget
   if project_complexity == "simple":
       use_fast_model()  # $0.01-0.02
   elif budget_priority == "cost":
       use_efficient_model()  # $0.02-0.04
   else:
       use_best_model()  # $0.03-0.06
   ```

#### Projected Cost Savings

- **A/B Testing**: 15-25% cost reduction by finding optimal models
- **Smart Regeneration**: 20-30% savings by avoiding unnecessary retries
- **Caching**: 10-15% reduction through asset reuse
- **Quality Gates**: 5-10% savings by preventing over-processing

**Target Cost**: $0.10-0.20 per video (50% of current cost)

### Backup & Recovery

#### Database Backups

```bash
# ClickHouse backup
docker-compose exec clickhouse clickhouse-client --query "
BACKUP DATABASE tensorzero TO Disk('backups', 'backup_$(date +%Y%m%d_%H%M%S)')
"

# Postgres backup
docker-compose exec postgres pg_dump tensorzero > backup_$(date +%Y%m%d).sql
```

#### Configuration Backup

```bash
# Backup configs and templates
tar -czf backup_$(date +%Y%m%d).tar.gz \
  config/ templates/ docker-compose.yml nginx.conf
```

#### Disaster Recovery

1. **Stop all services**: `docker-compose down`
2. **Restore databases** from backups
3. **Restore configs**: `tar -xzf backup_*.tar.gz`
4. **Restart services**: `docker-compose up -d`

### Security Considerations

#### API Security

- **Rate Limiting**: Implemented in nginx.conf
- **Input Validation**: Pydantic models in FastAPI
- **CORS**: Configured for allowed origins
- **API Keys**: Environment variables only

#### Data Protection

- **Encryption**: TLS for external traffic
- **Access Control**: Internal network for databases
- **Audit Logging**: All API calls logged
- **PII Handling**: No user data stored

### Troubleshooting

#### Common Issues

1. **Services won't start**

   ```bash
   # Check logs
   docker-compose logs [service_name]

   # Check resource usage
   docker stats

   # Restart specific service
   docker-compose restart [service_name]
   ```

2. **High latency**

   ```bash
   # Check Redis queue length
   docker-compose exec redis redis-cli LLEN video_jobs

   # Monitor ClickHouse performance
   docker-compose exec clickhouse clickhouse-client --query "
   SELECT * FROM system.metrics WHERE metric LIKE '%Query%'
   "
   ```

3. **Out of memory**
   ```bash
   # Increase Docker memory limit
   # Or scale down concurrent workers
   docker-compose up --scale worker=2
   ```

#### Performance Tuning

```yaml
# docker-compose.yml adjustments
services:
  video-api:
    environment:
      - MAX_WORKERS=2 # Reduce for lower memory usage
    deploy:
      resources:
        limits:
          memory: 1G
        reservations:
          memory: 512M
```

### API Documentation

#### Generate Video

```bash
POST /api/generate_video
Content-Type: application/json

{
  "project_description": "how to make paper airplanes",
  "content_mode": "kid",
  "target_duration": 180,
  "quality_mode": "standard",
  "priority": "normal"
}
```

#### Check Status

```bash
GET /api/status/{job_id}
```

#### Health Check

```bash
GET /health
```

#### Metrics

```bash
GET /api/metrics
```

### Contributing

1. **Code Quality**: Run `uv run ruff check . --fix && uv run mypy .`
2. **Testing**: Add tests for new features
3. **Documentation**: Update this README for changes
4. **Security**: Review security implications of changes

## 📄 License

Apache 2.0
