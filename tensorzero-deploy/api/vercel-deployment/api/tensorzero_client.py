import os
import httpx
from typing import Optional, Dict, Any


class TensorZeroClient:
    """Async client for TensorZero gateway inference calls"""

    def __init__(self):
        self.base_url = os.getenv("TENSORZERO_URL", "http://localhost:3000")
        self.timeout = 60.0

    async def call_function(
        self,
        function_name: str,
        input_data: Dict[str, Any],
        variant_name: Optional[str] = None,
        stream: bool = False,
    ) -> Dict[str, Any]:
        """
        Call a TensorZero function via HTTP POST

        Args:
            function_name: Name of the function to call
            input_data: Input parameters for the function
            variant_name: Optional specific variant to use
            stream: Whether to stream the response

        Returns:
            JSON response from TensorZero

        Raises:
            httpx.HTTPStatusError: If the request fails
        """
        payload = {"function_name": function_name, "input": input_data, "stream": stream}

        if variant_name:
            payload["variant_name"] = variant_name

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{self.base_url}/inference",
                json=payload,
                headers={"Content-Type": "application/json"},
            )
            response.raise_for_status()
            return response.json()

    async def health_check(self) -> bool:
        """Check if TensorZero gateway is healthy"""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(f"{self.base_url}/health")
                return response.status_code == 200
        except:
            return False
