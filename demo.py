import asyncio
import os
from typing import Any, Dict

from agentkernel.api import RESTAPI
from agentkernel.openai import OpenAIModule
from agents import Agent, function_tool
from agents.mcp import MCPServerStdio
from lightrag.llm.openai import gpt_4o_mini_complete, openai_embed  # type: ignore
from pydantic import BaseModel
from unified_retrieval.core.retriever import UnifiedRetriever

cloudwatch_mcp_server = MCPServerStdio(
    name="AWS CloudWatch MCP Server",
    params={
        "command": "uvx",
        "args": ["awslabs.cloudwatch-mcp-server@latest"],
        "env": {
            "AWS_REGION": "us-east-1",
            "AWS_ACCESS_KEY_ID": os.getenv("AWS_ACCESS_KEY_ID", ""),
            "AWS_SECRET_ACCESS_KEY": os.getenv("AWS_SECRET_ACCESS_KEY", ""),
            "AWS_SESSION_TOKEN": os.getenv("AWS_SESSION_TOKEN", ""),
            "FASTMCP_LOG_LEVEL": "ERROR",
        },
    },
    client_session_timeout_seconds=180,
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


def create_agents():
    """Create and configure all agents."""
    cloudwatch_agent = Agent(
        name="cloudwatch",
        instructions="You assist with AWS CloudWatch related queries.",
        mcp_servers=[cloudwatch_mcp_server],
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
        handoffs=[cloudwatch_agent, retrieval_agent],
    )

    return [triage_agent, cloudwatch_agent, retrieval_agent]


async def main():
    """Setup and initialize the agent system."""
    agents = create_agents()

    OpenAIModule(agents)

    import uvicorn
    from agentkernel.api.handler import AgentRESTRequestHandler
    from agentkernel.core.config import AKConfig

    handlers = [AgentRESTRequestHandler()]
    host = AKConfig.get().api.host
    port = AKConfig.get().api.port

    routers = []
    for handler in handlers:
        if handler is not None:
            routers.append(handler.get_router())

    if AKConfig.get().a2a.enabled:
        from agentkernel.api.a2a.handler import A2ARESTRequestHandler

        routers.append(A2ARESTRequestHandler.get_catalog_router())
        routers.extend(A2ARESTRequestHandler.get_agent_routers())
    if AKConfig.get().mcp.enabled:
        from agentkernel.api.mcp.akmcp import MCP

        mcp_app = MCP.get_http_app()
        app = RESTAPI._create_app(routers=routers, lifespan=mcp_app.lifespan)
        app.mount("/mcp", mcp_app)
    else:
        app = RESTAPI._create_app(routers=routers)
    # Add any custom routers
    for router in RESTAPI._custom_routers:
        app.include_router(router, prefix=AKConfig.get().api.custom_router_prefix)

    # Ensure MCP stdio servers start/stop inside the FastAPI lifespan
    async def _mcp_startup():
        try:
            await cloudwatch_mcp_server.connect()
        except Exception:
            pass

    async def _mcp_shutdown():
        try:
            await cloudwatch_mcp_server.cleanup()
        except Exception:
            pass

    app.add_event_handler("startup", _mcp_startup)
    app.add_event_handler("shutdown", _mcp_shutdown)

    config = uvicorn.Config(app=app, host=host, port=port, reload=False)
    server = uvicorn.Server(config=config)
    await server.serve()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except asyncio.CancelledError:
        pass
