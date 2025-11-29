-- Production Monitoring Dashboard Queries
-- Execute these ClickHouse queries to monitor system health and performance

-- =============================================================================
-- REAL-TIME DASHBOARD METRICS
-- =============================================================================

-- 1. System Health Overview
SELECT
    formatDateTime(now(), '%Y-%m-%d %H:%i:%s') as current_time,
    'Video Generation API' as service_name,
    (SELECT count(*) FROM tensorzero.metrics
     WHERE metric_name = 'video_quality_score'
     AND timestamp >= now() - INTERVAL 5 MINUTE) as requests_last_5min,
    (SELECT avg(value) FROM tensorzero.metrics
     WHERE metric_name = 'video_quality_score'
     AND timestamp >= now() - INTERVAL 5 MINUTE) as avg_quality_last_5min,
    (SELECT countIf(value >= 8.0) / count(*) FROM tensorzero.metrics
     WHERE metric_name = 'video_quality_score'
     AND timestamp >= now() - INTERVAL 5 MINUTE) * 100 as success_rate_pct,
    (SELECT avg(value) FROM tensorzero.metrics
     WHERE metric_name = 'video_generation_time'
     AND timestamp >= now() - INTERVAL 5 MINUTE) as avg_generation_time_sec;

-- 2. Job Queue Status
SELECT
    'Job Queue Status' as metric,
    (SELECT count(*) FROM tensorzero.inferences
     WHERE timestamp >= now() - INTERVAL 1 HOUR) as total_requests_1h,
    (SELECT count(*) FROM tensorzero.inferences
     WHERE function_name = 'script_generator'
     AND timestamp >= now() - INTERVAL 1 HOUR) as script_requests_1h,
    (SELECT count(*) FROM tensorzero.inferences
     WHERE function_name = 'prompt_enhancer'
     AND timestamp >= now() - INTERVAL 1 HOUR) as prompt_requests_1h,
    (SELECT count(*) FROM tensorzero.inferences
     WHERE function_name = 'video_evaluator'
     AND timestamp >= now() - INTERVAL 1 HOUR) as eval_requests_1h;

-- =============================================================================
-- PERFORMANCE MONITORING
-- =============================================================================

-- 3. Function Performance by Hour
SELECT
    toStartOfHour(timestamp) as hour,
    function_name,
    count(*) as requests,
    avg(CASE WHEN metric_name = 'video_generation_time' THEN value END) as avg_time_sec,
    avg(CASE WHEN metric_name = 'video_quality_score' THEN value END) as avg_quality,
    countIf(error_message != '') as errors
FROM tensorzero.inferences i
LEFT JOIN tensorzero.metrics m ON i.id = m.inference_id
WHERE timestamp >= now() - INTERVAL 24 HOUR
GROUP BY hour, function_name
ORDER BY hour DESC, function_name;

-- 4. Model Performance Comparison
SELECT
    model_name,
    count(*) as total_requests,
    avg(CASE WHEN m.metric_name = 'video_generation_time' THEN m.value END) as avg_time_sec,
    avg(CASE WHEN m.metric_name = 'video_quality_score' THEN m.value END) as avg_quality,
    countIf(i.error_message != '') as errors,
    countIf(i.error_message != '') / count(*) * 100 as error_rate_pct
FROM tensorzero.inferences i
LEFT JOIN tensorzero.metrics m ON i.id = m.inference_id
WHERE i.timestamp >= now() - INTERVAL 24 HOUR
GROUP BY model_name
ORDER BY avg_quality DESC, avg_time_sec ASC;

-- =============================================================================
-- QUALITY & COST ANALYSIS
-- =============================================================================

-- 5. Quality Distribution
SELECT
    CASE
        WHEN value >= 9 THEN 'Excellent (9-10)'
        WHEN value >= 7 THEN 'Good (7-8.9)'
        WHEN value >= 5 THEN 'Fair (5-6.9)'
        ELSE 'Poor (0-4.9)'
    END as quality_tier,
    count(*) as count,
    count(*) / (SELECT count(*) FROM tensorzero.metrics
                WHERE metric_name = 'video_quality_score'
                AND timestamp >= now() - INTERVAL 24 HOUR) * 100 as percentage
FROM tensorzero.metrics
WHERE metric_name = 'video_quality_score'
    AND timestamp >= now() - INTERVAL 24 HOUR
GROUP BY quality_tier
ORDER BY quality_tier DESC;

-- 6. Cost Analysis by Function
SELECT
    function_name,
    count(*) as requests,
    avg(CASE WHEN m.metric_name = 'video_generation_cost' THEN m.value END) as avg_cost_per_request,
    sum(CASE WHEN m.metric_name = 'video_generation_cost' THEN m.value END) as total_cost_24h,
    avg(CASE WHEN m.metric_name = 'video_quality_score' THEN m.value END) as avg_quality,
    avg(CASE WHEN m.metric_name = 'video_generation_time' THEN m.value END) as avg_time_sec
FROM tensorzero.inferences i
LEFT JOIN tensorzero.metrics m ON i.id = m.inference_id
WHERE i.timestamp >= now() - INTERVAL 24 HOUR
GROUP BY function_name
ORDER BY total_cost_24h DESC;

-- =============================================================================
-- A/B TESTING ANALYSIS
-- =============================================================================

-- 7. Experimentation Results
SELECT
    function_name,
    variant_name,
    count(*) as sample_size,
    avg(CASE WHEN m.metric_name = 'video_quality_score' THEN m.value END) as avg_quality,
    avg(CASE WHEN m.metric_name = 'video_generation_time' THEN m.value END) as avg_time_sec,
    avg(CASE WHEN m.metric_name = 'video_generation_cost' THEN m.value END) as avg_cost,
    stddevPop(CASE WHEN m.metric_name = 'video_quality_score' THEN m.value END) as quality_stddev
FROM tensorzero.experimentation e
LEFT JOIN tensorzero.metrics m ON e.inference_id = m.inference_id
WHERE e.timestamp >= now() - INTERVAL 7 DAY
GROUP BY function_name, variant_name
ORDER BY function_name, avg_quality DESC;

-- 8. Best Performing Variants
WITH variant_stats AS (
    SELECT
        function_name,
        variant_name,
        avg(CASE WHEN m.metric_name = 'video_quality_score' THEN m.value END) as avg_quality,
        avg(CASE WHEN m.metric_name = 'video_generation_time' THEN m.value END) as avg_time,
        avg(CASE WHEN m.metric_name = 'video_generation_cost' THEN m.value END) as avg_cost,
        count(*) as sample_size
    FROM tensorzero.experimentation e
    LEFT JOIN tensorzero.metrics m ON e.inference_id = m.inference_id
    WHERE e.timestamp >= now() - INTERVAL 7 DAY
    GROUP BY function_name, variant_name
)
SELECT
    function_name,
    variant_name as best_variant,
    avg_quality,
    avg_time,
    avg_cost,
    sample_size,
    ROW_NUMBER() OVER (PARTITION BY function_name ORDER BY avg_quality DESC, avg_time ASC) as rank
FROM variant_stats
WHERE sample_size >= 10  -- Minimum sample size
ORDER BY function_name, rank;

-- =============================================================================
-- ERROR MONITORING
-- =============================================================================

-- 9. Error Analysis
SELECT
    function_name,
    countIf(error_message != '') as total_errors,
    count(*) as total_requests,
    countIf(error_message != '') / count(*) * 100 as error_rate_pct,
    arrayJoin(arrayDistinct(arrayMap(x -> extract(error_message, '([A-Za-z]+)'), error_message))) as error_types
FROM tensorzero.inferences
WHERE timestamp >= now() - INTERVAL 24 HOUR
GROUP BY function_name
ORDER BY error_rate_pct DESC;

-- 10. Recent Errors (Last 100)
SELECT
    timestamp,
    function_name,
    model_name,
    error_message,
    input
FROM tensorzero.inferences
WHERE error_message != ''
    AND timestamp >= now() - INTERVAL 24 HOUR
ORDER BY timestamp DESC
LIMIT 100;

-- =============================================================================
-- USER SATISFACTION & BUSINESS METRICS
-- =============================================================================

-- 11. User Satisfaction Trends
SELECT
    toDate(timestamp) as date,
    avg(CASE WHEN metric_name = 'user_satisfaction' THEN value END) as avg_satisfaction,
    countIf(metric_name = 'user_satisfaction' AND value = true) as satisfied_users,
    countIf(metric_name = 'user_satisfaction') as total_ratings,
    countIf(metric_name = 'user_satisfaction' AND value = true) /
    countIf(metric_name = 'user_satisfaction') * 100 as satisfaction_rate_pct
FROM tensorzero.metrics
WHERE timestamp >= now() - INTERVAL 30 DAY
GROUP BY date
ORDER BY date DESC;

-- 12. Quality vs Satisfaction Correlation
SELECT
    round(q.value, 1) as quality_bucket,
    avg(s.value) as avg_satisfaction,
    count(*) as sample_size,
    countIf(s.value = true) / count(*) * 100 as satisfaction_rate
FROM tensorzero.metrics q
JOIN tensorzero.metrics s ON q.episode_id = s.episode_id
WHERE q.metric_name = 'video_quality_score'
    AND s.metric_name = 'user_satisfaction'
    AND q.timestamp >= now() - INTERVAL 30 DAY
GROUP BY round(q.value, 1)
ORDER BY quality_bucket DESC;

-- =============================================================================
-- SYSTEM HEALTH & ALERTS
-- =============================================================================

-- 13. System Health Check
SELECT
    'System Health' as check_type,
    (SELECT count(*) FROM tensorzero.inferences
     WHERE timestamp >= now() - INTERVAL 5 MINUTE) > 0 as api_healthy,
    (SELECT avg(value) FROM tensorzero.metrics
     WHERE metric_name = 'video_quality_score'
     AND timestamp >= now() - INTERVAL 1 HOUR) >= 5.0 as quality_healthy,
    (SELECT countIf(error_message != '') / count(*) FROM tensorzero.inferences
     WHERE timestamp >= now() - INTERVAL 1 HOUR) < 0.1 as error_rate_healthy,
    (SELECT avg(value) FROM tensorzero.metrics
     WHERE metric_name = 'video_generation_time'
     AND timestamp >= now() - INTERVAL 1 HOUR) < 600 as performance_healthy;

-- 14. Alert Conditions (Run every 5 minutes)
SELECT
    'ALERT: Quality Degradation' as alert_type,
    current_avg as current_quality,
    previous_avg as previous_quality,
    ((current_avg - previous_avg) / previous_avg) * 100 as percent_change
FROM (
    SELECT
        (SELECT avg(value) FROM tensorzero.metrics
         WHERE metric_name = 'video_quality_score'
         AND timestamp >= now() - INTERVAL 30 MINUTE) as current_avg,
        (SELECT avg(value) FROM tensorzero.metrics
         WHERE metric_name = 'video_quality_score'
         AND timestamp >= now() - INTERVAL 1 HOUR
         AND timestamp < now() - INTERVAL 30 MINUTE) as previous_avg
) subquery
WHERE current_avg < previous_avg * 0.9  -- 10% degradation

UNION ALL

SELECT
    'ALERT: High Error Rate' as alert_type,
    error_rate * 100 as error_rate_pct,
    total_requests,
    null
FROM (
    SELECT
        countIf(error_message != '') / count(*) as error_rate,
        count(*) as total_requests
    FROM tensorzero.inferences
    WHERE timestamp >= now() - INTERVAL 1 HOUR
) subquery
WHERE error_rate > 0.15  -- 15% error rate

UNION ALL

SELECT
    'ALERT: Slow Performance' as alert_type,
    avg_time,
    null,
    null
FROM (
    SELECT avg(value) as avg_time
    FROM tensorzero.metrics
    WHERE metric_name = 'video_generation_time'
        AND timestamp >= now() - INTERVAL 1 HOUR
) subquery
WHERE avg_time > 900;  -- Over 15 minutes average