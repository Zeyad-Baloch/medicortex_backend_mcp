import sys
import json
import httpx
import asyncio
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent
from typing import Any


FASTAPI_BASE_URL = "http://localhost:8000"


mcp = Server("medicortex-health-agent")


print("Initializing MediCortex MCP Server...", file=sys.stderr, flush=True)

@mcp.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="learn_baseline",
            description="Train personalized health baseline for a user.",
            inputSchema={
                "type": "object",
                "properties": {
                    "user_id": {"type": "string"},
                    "data": {"type": "string"},
                    "days": {"type": "integer", "default": 7}
                },
                "required": ["user_id", "data"]
            }
        ),
        Tool(
            name="detect_anomalies",
            description="Detect health anomalies based on user's baseline.",
            inputSchema={
                "type": "object",
                "properties": {
                    "user_id": {"type": "string"},
                    "data": {"type": "string"}
                },
                "required": ["user_id", "data"]
            }
        ),
        Tool(
            name="get_baseline",
            description="Retrieve user's learned health baseline.",
            inputSchema={
                "type": "object",
                "properties": {
                    "user_id": {"type": "string"}
                },
                "required": ["user_id"]
            }
        )
    ]

@mcp.call_tool()
async def call_tool(name: str, arguments: Any) -> list[TextContent]:

    print(f"Tool call received: {name}", file=sys.stderr, flush=True)

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            if name == "learn_baseline":
                data = json.loads(arguments["data"])
                resp = await client.post(f"{FASTAPI_BASE_URL}/baseline/train",
                                       json={"user_id": arguments["user_id"], "data": data, "days": arguments.get("days", 7)})
                return [TextContent(type="text", text=json.dumps(resp.json()))]

            elif name == "detect_anomalies":
                data = json.loads(arguments["data"])
                resp = await client.post(f"{FASTAPI_BASE_URL}/anomaly/detect",
                                       json={"user_id": arguments["user_id"], "data": data})
                return [TextContent(type="text", text=json.dumps(resp.json()))]

            elif name == "get_baseline":
                resp = await client.get(f"{FASTAPI_BASE_URL}/baseline/{arguments['user_id']}")
                return [TextContent(type="text", text=json.dumps(resp.json()))]
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr, flush=True)
        return [TextContent(type="text", text=f"Error: {str(e)}")]

async def main():

    print("Starting Stdio Server...", file=sys.stderr, flush=True)
    async with stdio_server() as (read_stream, write_stream):
        await mcp.run(read_stream, write_stream, mcp.create_initialization_options())

if __name__ == "__main__":
    asyncio.run(main())