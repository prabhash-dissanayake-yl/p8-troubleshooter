import asyncio

from agentkernel.cli import CLI
from agentkernel.openai import OpenAIModule
from agents import Agent
from agents.mcp import MCPServerStdio

math_agent = Agent(
    name="math",
    handoff_description="Specialist agent for math questions",
    instructions="You provide help with math problems. Give short and direct answers exactly to the question. "
    "Don't provide any explanations nor additional details.",
)

general_agent = Agent(
    name="general",
    handoff_description="Agent for general questions",
    instructions="You provide assistance with general queries. Give short and direct answers exactly to the question. "
    "Don't provide any explanations nor additional details",
)

cloudwatch_mcp_server = MCPServerStdio(
    name="AWS CloudWatch MCP Server",
    params={
        "command": "uvx",
        "args": ["awslabs.cloudwatch-mcp-server@latest"],
        "env": {"AWS_PROFILE": "temp", "FASTMCP_LOG_LEVEL": "ERROR"},
    },
)
cloudwatch_agent = Agent(
    name="cloudwatch",
    instructions="You assist with AWS CloudWatch related queries.",
    mcp_servers=[cloudwatch_mcp_server],
)

triage_agent = Agent(
    name="triage",
    instructions="You determine which agent to use based on the user's question. Give short and direct answers exactly to the question. "
    "Don't provide any explanations nor additional details",
    handoffs=[general_agent, math_agent, cloudwatch_agent],
)


async def main():
    await cloudwatch_mcp_server.connect()

    OpenAIModule([triage_agent, math_agent, general_agent, cloudwatch_agent])

    cli = CLI()
    await cli.run()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except asyncio.CancelledError:
        pass
