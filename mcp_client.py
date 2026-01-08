import os
import sys
import json
import asyncio
import subprocess
from dotenv import load_dotenv
from google import genai
from google.genai import types
from typing import Dict, Any, List

load_dotenv()

class MCPClient:
    def __init__(self, server_script_path: str):
        self.server_script = server_script_path
        self.process = None
        self.request_id = 0
        self.tools = []

    async def connect(self):
        print("Connecting to MCP Server...")
        self.process = await asyncio.create_subprocess_exec(
            sys.executable, "-u", self.server_script,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )


        await self._send_request("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "medicortex-client", "version": "1.0.0"}
        })

        await self._send_notification("notifications/initialized")
        tools_response = await self._send_request("tools/list", {})
        self.tools = tools_response.get("tools", [])
        print(f"✓ Connected! Tools: {[t['name'] for t in self.tools]}")

    async def _send_request(self, method: str, params: Dict) -> Dict:
        self.request_id += 1
        req = {"jsonrpc": "2.0", "id": self.request_id, "method": method, "params": params}
        self.process.stdin.write((json.dumps(req) + "\n").encode())
        await self.process.stdin.drain()

        line = await self.process.stdout.readline()
        if not line:
            return {}
        return json.loads(line.decode()).get("result", {})

    async def _send_notification(self, method: str):
        notif = {"jsonrpc": "2.0", "method": method}
        self.process.stdin.write((json.dumps(notif) + "\n").encode())
        await self.process.stdin.drain()

    async def call_tool(self, name: str, args: Dict) -> str:
        resp = await self._send_request("tools/call", {"name": name, "arguments": args})
        # Extract text from the content list
        return resp.get("content", [{}])[0].get("text", "No response from tool.")



async def main():

    mcp_client = MCPClient("mcp_server.py")
    await mcp_client.connect()


    gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


    health_tool = types.Tool(
        function_declarations=[
            types.FunctionDeclaration(
                name="detect_anomalies",
                description="Checks health vitals for risks using your trained ML models.",
                parameters={
                    "type": "object",
                    "properties": {
                        "user_id": {"type": "string"},
                        "data": {"type": "string"}  # The ML model expects a JSON string
                    },
                    "required": ["user_id", "data"]
                }
            )
        ]
    )

    user_query = "Check health for user_123. HR is 145 and Stress is 95. Use the full data record."
    print(f"\n User: {user_query}")


    messages = [
        types.Content(
            role="user",
            parts=[types.Part(text=user_query)]
        )
    ]

    # ReAct Loop
    max_iterations = 5
    for i in range(max_iterations):
        # Step 1: Reason
        response = gemini_client.models.generate_content(
            model="gemini-2.0-flash",
            contents=messages,
            config=types.GenerateContentConfig(tools=[health_tool])
        )

        model_content = response.candidates[0].content
        model_part = model_content.parts[0]


        if model_part.function_call:
            fc = model_part.function_call
            print(f" Gemini Reasoned: Calling {fc.name}...")

            tool_args = dict(fc.args)


            if fc.name == "detect_anomalies":
                tool_args["data"] = json.dumps([{
                    "timestamp": "2026-01-08T14:00:00",
                    "heart_rate": 145,
                    "steps": 5,
                    "sleep_quality": 0,
                    "stress_level": 95,
                    "calories": 15
                }])


            messages.append(model_content)


            tool_result = await mcp_client.call_tool(fc.name, tool_args)
            print(f"✓ Observation: Tool Result received.")


            messages.append(types.Content(
                role="tool",
                parts=[
                    types.Part.from_function_response(
                        name=fc.name,
                        response={"result": tool_result}
                    )
                ]
            ))
        else:

            print(f"\n FINAL INSIGHT:\n{model_part.text}")
            break

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n Client stopped.")