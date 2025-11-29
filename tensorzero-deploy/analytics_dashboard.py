#!/usr/bin/env python3
"""
Analytics Dashboard for Video Quality Feedback System

This script provides insights into video generation quality, costs, and A/B testing results.
Run ClickHouse queries to analyze TensorZero metrics.
"""

import asyncio
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any
import httpx


class AnalyticsDashboard:
    """Dashboard for analyzing video quality metrics"""

    def __init__(self, clickhouse_url: str | None = None):
        self.clickhouse_url = clickhouse_url or os.getenv(
            "TENSORZERO_CLICKHOUSE_URL", "http://localhost:8123"
        )
        self.client = httpx.AsyncClient(timeout=30.0)

    async def run_query(self, query: str) -> List[Dict[str, Any]]:
        """Execute ClickHouse query"""
        try:
            response = await self.client.post(
                f"{self.clickhouse_url}/?database=tensorzero",
                content=query,
                headers={"Content-Type": "text/plain"},
            )
            response.raise_for_status()

            # Parse TSV response
            lines = response.text.strip().split("\n")
            if not lines:
                return []

            headers = lines[0].split("\t")
            results = []

            for line in lines[1:]:
                if line.strip():
                    values = line.split("\t")
                    row = dict(zip(headers, values))
                    results.append(row)

            return results

        except Exception as e:
            print(f"❌ Query failed: {e}")
            return []

    async def quality_trends_dashboard(self, days: int = 7) -> None:
        """Show quality trends over time"""
        print("📊 Quality Trends Dashboard")
        print("=" * 50)

        query = f"""
        SELECT
            toDate(timestamp) as date,
            round(avg(value), 2) as avg_quality,
            count(*) as evaluations,
            round(countIf(value >= 8.0) / count(*) * 100, 1) as high_quality_pct
        FROM tensorzero.metrics
        WHERE metric_name = 'video_quality_score'
            AND timestamp >= now() - INTERVAL {days} DAY
        GROUP BY date
        ORDER BY date DESC
        """

        results = await self.run_query(query)

        if not results:
            print("No quality data available")
            return

        print(f"Quality metrics for last {days} days:")
        print("Date\t\tQuality\tCount\tHigh Quality %")
        print("-" * 50)

        for row in results:
            date = row.get("date", "Unknown")
            quality = row.get("avg_quality", "0")
            count = row.get("evaluations", "0")
            pct = row.get("high_quality_pct", "0")
            print(f"{date}\t{quality}\t{count}\t{pct}%")

    async def ab_testing_analysis(self) -> None:
        """Analyze A/B testing results"""
        print("\n🧪 A/B Testing Analysis")
        print("=" * 50)

        query = """
        SELECT
            variant_name,
            round(avg(quality_score), 2) as avg_quality,
            count(*) as sample_size,
            round(avg(generation_time), 1) as avg_time,
            round(avg(cost), 3) as avg_cost
        FROM (
            SELECT
                e.variant_name,
                m1.value as quality_score,
                m2.value as generation_time,
                m3.value as cost
            FROM tensorzero.experimentation e
            JOIN tensorzero.metrics m1 ON e.inference_id = m1.inference_id
            JOIN tensorzero.metrics m2 ON e.inference_id = m2.inference_id
            JOIN tensorzero.metrics m3 ON e.inference_id = m3.inference_id
            WHERE m1.metric_name = 'video_quality_score'
                AND m2.metric_name = 'video_generation_time'
                AND m3.metric_name = 'video_generation_cost'
                AND e.timestamp >= now() - INTERVAL 14 DAY
        ) subquery
        GROUP BY variant_name
        ORDER BY avg_quality DESC
        """

        results = await self.run_query(query)

        if not results:
            print("No A/B testing data available")
            return

        print("Variant Performance (Quality vs Cost):")
        print("Variant\t\tQuality\tSamples\tTime(s)\tCost($)")
        print("-" * 60)

        for row in results:
            variant = row.get("variant_name", "Unknown")[:15]
            quality = row.get("avg_quality", "0")
            samples = row.get("sample_size", "0")
            time = row.get("avg_time", "0")
            cost = row.get("avg_cost", "0")
            print(f"{variant}\t{quality}\t{samples}\t\t{time}\t{cost}")

    async def cost_quality_analysis(self) -> None:
        """Analyze cost vs quality tradeoffs"""
        print("\n💰 Cost vs Quality Analysis")
        print("=" * 50)

        query = """
        SELECT
            round(avg_quality, 1) as quality_bucket,
            round(avg(cost_value), 3) as avg_cost,
            count(*) as video_count,
            round(avg(generation_time), 1) as avg_time
        FROM (
            SELECT
                m1.value as avg_quality,
                m2.value as cost_value,
                m3.value as generation_time
            FROM tensorzero.metrics m1
            JOIN tensorzero.metrics m2 ON m1.inference_id = m2.inference_id
            JOIN tensorzero.metrics m3 ON m1.inference_id = m3.inference_id
            WHERE m1.metric_name = 'video_quality_score'
                AND m2.metric_name = 'video_generation_cost'
                AND m3.metric_name = 'video_generation_time'
                AND m1.timestamp >= now() - INTERVAL 30 DAY
        ) subquery
        GROUP BY round(avg_quality, 1)
        ORDER BY quality_bucket DESC
        """

        results = await self.run_query(query)

        if not results:
            print("No cost/quality data available")
            return

        print("Quality vs Cost Tradeoff:")
        print("Quality\tCount\tAvg Cost\tAvg Time")
        print("-" * 40)

        for row in results:
            quality = row.get("quality_bucket", "0")
            count = row.get("video_count", "0")
            cost = row.get("avg_cost", "0")
            time = row.get("avg_time", "0")
            print(f"{quality}\t{count}\t${cost}\t\t{time}s")

    async def real_time_dashboard(self) -> None:
        """Show real-time metrics dashboard"""
        print("\n📈 Real-Time Dashboard")
        print("=" * 50)

        query = """
        SELECT
            formatDateTime(now(), '%Y-%m-%d %H:%i:%s') as current_time,
            round((SELECT avg(value) FROM tensorzero.metrics
                   WHERE metric_name = 'video_quality_score'
                   AND timestamp >= now() - INTERVAL 1 HOUR), 2) as quality_last_hour,
            (SELECT count(*) FROM tensorzero.metrics
             WHERE metric_name = 'video_quality_score'
             AND timestamp >= now() - INTERVAL 1 HOUR) as videos_last_hour,
            round((SELECT avg(value) FROM tensorzero.metrics
                   WHERE metric_name = 'video_generation_time'
                   AND timestamp >= now() - INTERVAL 1 HOUR), 1) as avg_time_last_hour,
            round((SELECT sum(value) FROM tensorzero.metrics
                   WHERE metric_name = 'video_generation_cost'
                   AND timestamp >= now() - INTERVAL 1 HOUR), 3) as cost_last_hour
        """

        results = await self.run_query(query)

        if results:
            row = results[0]
            print(f"Current Time: {row.get('current_time', 'Unknown')}")
            print(f"Quality (Last Hour): {row.get('quality_last_hour', 'N/A')}/10")
            print(f"Videos (Last Hour): {row.get('videos_last_hour', '0')}")
            print(f"Avg Time (Last Hour): {row.get('avg_time_last_hour', 'N/A')}s")
            print(f"Total Cost (Last Hour): ${row.get('cost_last_hour', '0')}")
        else:
            print("No real-time data available")

    async def show_dashboard(self) -> None:
        """Run complete analytics dashboard"""
        print("🎯 Video Quality Analytics Dashboard")
        print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()

        await self.quality_trends_dashboard()
        await self.ab_testing_analysis()
        await self.cost_quality_analysis()
        await self.real_time_dashboard()

        print("\n💡 Insights:")
        print("- Monitor quality trends to detect degradation")
        print("- Compare A/B test variants for optimal settings")
        print("- Balance cost vs quality for business requirements")
        print("- Use real-time metrics for operational monitoring")

    async def close(self):
        """Close HTTP client"""
        await self.client.aclose()


async def main():
    """Main analytics dashboard"""
    dashboard = AnalyticsDashboard()

    try:
        await dashboard.show_dashboard()
    except Exception as e:
        print(f"❌ Dashboard error: {e}")
    finally:
        await dashboard.close()


if __name__ == "__main__":
    asyncio.run(main())
