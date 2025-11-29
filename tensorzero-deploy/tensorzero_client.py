"""
TensorZero Client for Fast Agent Integration

This module provides a clean async client for TensorZero inference and feedback.
All prompts are stored in TensorZero templates - NO PROMPTS IN CODE.
"""

import httpx
from typing import Dict, Any, Optional, AsyncIterator, Union
import json
import logging

logger = logging.getLogger(__name__)


class TensorZeroClient:
    """Async client for TensorZero inference and feedback"""

    def __init__(self, base_url: str = "http://localhost:3000"):
        self.base_url = base_url
        self.client = httpx.AsyncClient(timeout=60.0)

    async def inference(
        self,
        function_name: str,
        input: Dict[str, Any],
        stream: bool = False,
        variant_name: Optional[str] = None,
        episode_id: Optional[str] = None,
    ) -> Union[Dict[str, Any], AsyncIterator[Dict[str, Any]]]:
        """Call TensorZero inference endpoint

        Args:
            function_name: Name of TensorZero function (e.g., "script_generator")
            input: Parameters matching the function's schema
            stream: Enable streaming response
            variant_name: Specific variant to use (for testing)
            episode_id: Episode ID for tracking multi-turn conversations

        Returns:
            Inference result or async iterator for streaming
        """
        payload = {"function_name": function_name, "input": input, "stream": stream}

        if variant_name:
            payload["variant_name"] = variant_name
        if episode_id:
            payload["episode_id"] = episode_id

        logger.info(f"📡 Calling TensorZero function: {function_name}")

        if stream:
            return self._stream_inference(payload)
        else:
            response = await self.client.post(f"{self.base_url}/inference", json=payload)
            response.raise_for_status()
            result = response.json()

            # Extract the actual content from TensorZero response
            if "output" in result and "parsed" in result["output"]:
                return result["output"]["parsed"]
            return result

    async def _stream_inference(self, payload: Dict) -> AsyncIterator[Dict]:
        """Handle streaming responses"""
        async with self.client.stream(
            "POST", f"{self.base_url}/inference", json=payload
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data = json.loads(line[6:])
                    yield data

    async def feedback(
        self,
        inference_id: str,
        metric_name: str,
        value: Union[float, bool],
        episode_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Send feedback to TensorZero for learning

        Args:
            inference_id: ID from inference response
            metric_name: Name of metric (defined in tensorzero.toml)
            value: Metric value (float or boolean)
            episode_id: Episode ID if tracking conversations
        """
        payload = {"inference_id": inference_id, "metric_name": metric_name, "value": value}

        if episode_id:
            payload["episode_id"] = episode_id

        logger.info(f"📊 Sending feedback for metric: {metric_name}")

        response = await self.client.post(f"{self.base_url}/feedback", json=payload)
        response.raise_for_status()
        return response.json()

    async def report_metric(
        self,
        metric_name: str,
        value: Any,
        inference_id: Optional[str] = None,
        episode_id: Optional[str] = None,
    ):
        """
        Sends feedback to TensorZero.
        """
        url = f"{self.base_url}/feedback"
        payload = {
            "metric_name": metric_name,
            "value": value,
        }

        if inference_id:
            payload["inference_id"] = inference_id
        elif episode_id:
            payload["episode_id"] = episode_id
        else:
            logger.warning("⚠️ Cannot report metric: No ID provided")
            return

        try:
            await self.client.post(url, json=payload)
            logger.info(f"📈 Metric Reported: {metric_name} = {value}")
        except Exception as e:
            logger.error(f"❌ Failed to report metric: {e}")

    async def close(self):
        """Close the HTTP client"""
        await self.client.aclose()
