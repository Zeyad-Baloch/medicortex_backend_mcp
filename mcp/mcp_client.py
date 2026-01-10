import os
import sys
import json
import asyncio
import subprocess
from dotenv import load_dotenv
from groq import Groq
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
        return resp.get("content", [{}])[0].get("text", "No response from tool.")

    async def cleanup(self):
        """Properly cleanup the MCP server process"""
        if self.process:
            try:
                self.process.terminate()
                await asyncio.wait_for(self.process.wait(), timeout=2.0)
            except asyncio.TimeoutError:
                self.process.kill()
            except Exception:
                pass


async def main():
    # Validate API key
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("❌ ERROR: GROQ_API_KEY not found!")
        print("\nPlease:")
        print("  1. Go to: https://console.groq.com/keys")
        print("  2. Create a free account (no credit card needed!)")
        print("  3. Generate an API key")
        print("  4. Add to .env file: GROQ_API_KEY=your_key_here")
        return

    print(f"✓ Groq API Key loaded")

    # Connect to MCP Server
    mcp_client = MCPClient("mcp/server.py")

    try:
        await mcp_client.connect()

        # Initialize Groq client
        groq_client = Groq(api_key=api_key)
        print("✓ Groq client initialized")

        # Define health tools (Groq uses OpenAI-style function calling)
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "detect_anomalies",
                    "description": "Detects health anomalies based on user's learned baseline. The user must have a trained baseline first.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "user_id": {
                                "type": "string",
                                "description": "The user's unique identifier (e.g., 'user_123')"
                            },
                            "data": {
                                "type": "string",
                                "description": "JSON string containing array of health data points with timestamp, heart_rate, steps, sleep_quality, stress_level, calories"
                            }
                        },
                        "required": ["user_id", "data"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_baseline",
                    "description": "Retrieves the user's trained health baseline to see their normal ranges",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "user_id": {
                                "type": "string",
                                "description": "The user's unique identifier"
                            }
                        },
                        "required": ["user_id"]
                    }
                }
            }
        ]

        user_query = "Check health for user_123. HR is 145 and Stress is 95. Use the full data record."
        print(f"\n👤 User: {user_query}")

        messages = [
            {
                "role": "system",
                "content": "You are a health monitoring AI assistant. You have access to tools to check health baselines and detect anomalies. Always check the user's baseline first to understand their normal ranges, then detect anomalies."
            },
            {
                "role": "user",
                "content": user_query
            }
        ]

        # ReAct Loop
        max_iterations = 5
        for i in range(max_iterations):
            try:
                response = groq_client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=messages,
                    tools=tools,
                    tool_choice="auto",
                    temperature=0.5,
                    max_tokens=2000
                )

                assistant_message = response.choices[0].message

                # Check if model wants to call a function
                if assistant_message.tool_calls:
                    tool_call = assistant_message.tool_calls[0]
                    function_name = tool_call.function.name
                    function_args = json.loads(tool_call.function.arguments)

                    print(f"🤖 Groq Reasoned: Calling {function_name}...")

                    # Prepare the health data for detect_anomalies
                    if function_name == "detect_anomalies":
                        # Only override if not already provided
                        if "data" not in function_args or not function_args["data"]:
                            function_args["data"] = json.dumps([{
                                "timestamp": "2026-01-09T14:00:00",
                                "heart_rate": 145,
                                "steps": 5,
                                "sleep_quality": 0,
                                "stress_level": 95,
                                "calories": 15
                            }])

                    # Add assistant's message to conversation
                    messages.append({
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": tool_call.id,
                                "type": "function",
                                "function": {
                                    "name": function_name,
                                    "arguments": json.dumps(function_args)
                                }
                            }
                        ]
                    })

                    # Call the tool via MCP
                    tool_result = await mcp_client.call_tool(function_name, function_args)
                    print(f"✓ Observation: Tool Result received.")

                    # Add tool result to conversation
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": function_name,
                        "content": tool_result
                    })

                else:
                    # Model has finished reasoning and provided final answer
                    print(f"\n💡 FINAL INSIGHT:\n{assistant_message.content}")
                    break

            except Exception as e:
                error_msg = str(e)
                if "rate_limit" in error_msg.lower():
                    print(f"❌ Rate limit hit (rare with Groq!). Wait 10 seconds...")
                else:
                    print(f"❌ Error in iteration {i+1}: {error_msg[:200]}")
                break

        print("\n✅ Session complete!")

    finally:
        # Cleanup MCP connection
        await mcp_client.cleanup()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Client stopped.")
    except Exception as e:
        print(f"❌ Fatal error: {e}")
