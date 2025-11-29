-- ClickHouse Analytics Queries for Video Quality Feedback System
-- These queries analyze TensorZero metrics to understand quality improvements over time

-- Query 1: Which Script Generator variant produces higher Quality Scores?
SELECT
    variant_name,
    count() as attempts,
    avg(value) as avg_quality_score,
    max(value) as max_score
FROM FloatMetricFeedback
JOIN JsonInference ON FloatMetricFeedback.target_id = JsonInference.id
WHERE
    function_name = 'script_generator'
    AND metric_name = 'video_quality_score'
GROUP BY variant_name
ORDER BY avg_quality_score DESC;

-- Query 2: Cost Analysis (Assuming we log cost as a metric)
-- This helps identify if a "Fair" model is significantly cheaper than a "Good" model.
SELECT
    variant_name,
    avg(value) as avg_cost,
    (SELECT avg(value) FROM FloatMetricFeedback WHERE metric_name='video_quality_score' AND target_id IN (SELECT id FROM JsonInference WHERE variant_name=T.variant_name)) as quality_correlation
FROM FloatMetricFeedback T
JOIN JsonInference ON T.target_id = JsonInference.id
WHERE
    metric_name = 'production_cost'
GROUP BY variant_name;

-- Query 3: Human vs. AI Score Correlation
-- Validates if the AI Judge aligns with human thumbs up/down.
SELECT
    AI.target_id as episode_id,
    AI.value as ai_score,
    Human.value as human_liked
FROM FloatMetricFeedback AI
JOIN BooleanMetricFeedback Human ON AI.target_id = Human.target_id
WHERE
    AI.metric_name = 'video_quality_score'
    AND Human.metric_name = 'human_rating'
LIMIT 50;