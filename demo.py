import asyncio
import os
from typing import Any, Dict

from agentkernel.cli import CLI
from agentkernel.openai import OpenAIModule
from agents import Agent, function_tool
from agents.mcp import MCPServerStdio
from lightrag.llm.openai import gpt_4o_mini_complete, openai_embed  # type: ignore
from pydantic import BaseModel
from unified_retrieval.core.retriever import UnifiedRetriever

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


class RetrievalArgs(BaseModel):
    query: str
    top_k: int = 5


async def lightrag_retrieve(query: str, top_k: int = 5) -> Dict[str, Any]:
    """LightRAG adapter for graph-based retrieval
    Args:
        query (str): The query text for retrieval
        top_k (int): Number of top results to retrieve
    Returns:
        Dict[str, Any]: Retrieval results as a dictionary
    """
    retriever = UnifiedRetriever.from_config(
        {
            "adapter": "lightrag",
            "working_dir": os.getenv("LIGHTRAG_WORKING_DIR", "./lightrag_storage"),
            "llm_model_func": gpt_4o_mini_complete,
            "embedding_func": openai_embed,
        }
    )

    # Ensure LightRAG storages are initialized
    try:
        await retriever.adapter._async_initialize_storage()
    except Exception:
        # If async initialization fails or is unnecessary, continue
        pass

    import asyncio as _asyncio

    resp = await _asyncio.to_thread(retriever.retrieve, query, top_k)
    return resp


lightrag_retrieval_tool = function_tool(
    lightrag_retrieve,
    name_override="lightrag_retrieval",
    description_override="LightRAG adapter for graph-based retrieval",
    strict_mode=True,
)
retrieval_agent = Agent(
    name="retrieval",
    handoff_description="Agent for retrieval/search-related questions",
    instructions="You provide assistance with retrieval and search queries. Don't ask any clarifying questions. Give short and direct answers exactly to the question.",
    tools=[lightrag_retrieval_tool],
)

triage_agent = Agent(
    name="triage",
    instructions="You determine which agent to use based on the user's question. Give short and direct answers exactly to the question. "
    "Don't provide any explanations nor additional details",
    handoffs=[general_agent, math_agent, cloudwatch_agent, retrieval_agent],
)


async def main():
    await cloudwatch_mcp_server.connect()

    OpenAIModule([triage_agent, math_agent, general_agent, cloudwatch_agent, retrieval_agent])

    cli = CLI()
    await cli.run()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except asyncio.CancelledError:
        pass
