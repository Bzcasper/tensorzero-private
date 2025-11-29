#!/usr/bin/env python3
"""
Test script for Media Server MCP server
"""

import asyncio
import subprocess
import sys
import time
from pathlib import Path


async def test_mcp_server():
    """Test the MCP server by running it and checking if it starts properly"""

    print("🧪 Testing Media Server MCP Server...")

    # Start the MCP server as a subprocess
    try:
        process = subprocess.Popen(
            [sys.executable, "media_server_mcp.py"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=Path(__file__).parent,
        )

        # Give it a moment to start
        time.sleep(2)

        # Check if process is still running
        if process.poll() is None:
            print("✅ MCP Server started successfully")
            process.terminate()
            process.wait()
            return True
        else:
            stdout, stderr = process.communicate()
            print(f"❌ MCP Server failed to start")
            print(f"STDOUT: {stdout}")
            print(f"STDERR: {stderr}")
            return False

    except Exception as e:
        print(f"❌ Error testing MCP server: {e}")
        return False


if __name__ == "__main__":
    success = asyncio.run(test_mcp_server())
    sys.exit(0 if success else 1)
