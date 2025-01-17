"""Enhanced conversational retrieval graph with improved state management and monitoring.

This module implements a sophisticated graph-based conversation system with:
- Versioned state management
- Performance monitoring and metrics collection
- Comprehensive error handling
- Type-safe implementations
"""

from typing import Any, Literal, TypedDict, cast, Optional
from dataclasses import asdict
from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langchain_openai import ChatOpenAI
from langgraph.types import interrupt, Command
from main_graph.graph_states import AgentState, Router, GradeHallucinations, InputState, StateVersion
from utils.prompt import (
    ROUTER_SYSTEM_PROMPT,
    RESEARCH_PLAN_SYSTEM_PROMPT,
    MORE_INFO_SYSTEM_PROMPT,
    GENERAL_SYSTEM_PROMPT,
    CHECK_HALLUCINATIONS,
    RESPONSE_SYSTEM_PROMPT
)
from subgraph.graph_builder import researcher_graph
from langchain_core.documents import Document
from langgraph.graph import END, START, StateGraph
from langgraph.checkpoint.memory import MemorySaver
import logging
import time
from typing import Dict, List
from prometheus_client import Counter, Histogram, start_http_server
from utils.utils import config

# Enhanced logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress verbose logging from dependencies
for logger_name in ["openai", "urllib3", "httpx"]:
    logging.getLogger(logger_name).setLevel(logging.WARNING)
    logging.getLogger(logger_name).propagate = False

# Load configuration
GPT_4o_MINI = config["llm"]["gpt_4o_mini"]
GPT_4o = config["llm"]["gpt_4o"]
TEMPERATURE = config["llm"]["temperature"]

# Metrics
QUERY_COUNTER = Counter('rag_queries_total', 'Total number of queries processed')
QUERY_DURATION = Histogram('rag_query_duration_seconds', 'Time spent processing queries')
ERROR_COUNTER = Counter('rag_errors_total', 'Total number of errors encountered')
HALLUCINATION_COUNTER = Counter('rag_hallucinations_total', 'Total number of detected hallucinations')

# Start metrics server
start_http_server(8000)

class GraphError(Exception):
    """Base exception for graph-related errors."""
    pass

class StateError(GraphError):
    """Exception for state-related errors."""
    pass

class QueryError(GraphError):
    """Exception for query processing errors."""
    pass

async def analyze_and_route_query(
    state: AgentState,
    *,
    config: RunnableConfig
) -> dict[str, Router]:
    """Analyze and route user query with enhanced error handling and monitoring."""
    QUERY_COUNTER.inc()
    start_time = time.time()
    
    try:
        model = ChatOpenAI(model=GPT_4o, temperature=TEMPERATURE, streaming=True)
        messages = [
            {"role": "system", "content": ROUTER_SYSTEM_PROMPT}
        ] + state.messages
        
        logger.info("Analyzing query for routing")
        response = cast(Router, await model.with_structured_output(Router).ainvoke(messages))
        
        # Record metrics
        QUERY_DURATION.observe(time.time() - start_time)
        
        return {"router": response}
        
    except Exception as e:
        ERROR_COUNTER.inc()
        logger.error(f"Query analysis failed: {e}")
        raise QueryError(f"Failed to analyze query: {str(e)}")

def route_query(
    state: AgentState,
) -> Literal["create_research_plan", "ask_for_more_info", "respond_to_general_query"]:
    """Route query based on classification with validation."""
    try:
        _type = state.router["type"]
        if _type == "environmental":
            return "create_research_plan"
        elif _type == "more-info":
            return "ask_for_more_info"
        elif _type == "general":
            return "respond_to_general_query"
        else:
            raise ValueError(f"Unknown router type {_type}")
    except Exception as e:
        ERROR_COUNTER.inc()
        logger.error(f"Query routing failed: {e}")
        raise StateError(f"Failed to route query: {str(e)}")

async def create_research_plan(
    state: AgentState,
    *,
    config: RunnableConfig
) -> dict[str, Union[List[str], str]]:
    """Create research plan with performance monitoring."""
    start_time = time.time()
    
    try:
        class Plan(TypedDict):
            steps: List[str]

        model = ChatOpenAI(model=GPT_4o_MINI, temperature=TEMPERATURE, streaming=True)
        messages = [
            {"role": "system", "content": RESEARCH_PLAN_SYSTEM_PROMPT}
        ] + state.messages
        
        logger.info("Generating research plan")
        response = cast(Plan, await model.with_structured_output(Plan).ainvoke(messages))
        
        # Record metrics
        QUERY_DURATION.observe(time.time() - start_time)
        
        return {"steps": response["steps"], "documents": "delete"}
        
    except Exception as e:
        ERROR_COUNTER.inc()
        logger.error(f"Research plan creation failed: {e}")
        raise QueryError(f"Failed to create research plan: {str(e)}")

async def ask_for_more_info(
    state: AgentState,
    *,
    config: RunnableConfig
) -> dict[str, List[BaseMessage]]:
    """Request additional information with error handling."""
    try:
        model = ChatOpenAI(model=GPT_4o_MINI, temperature=TEMPERATURE, streaming=True)
        system_prompt = MORE_INFO_SYSTEM_PROMPT.format(logic=state.router["logic"])
        messages = [{"role": "system", "content": system_prompt}] + state.messages
        
        response = await model.ainvoke(messages)
        return {"messages": [response]}
        
    except Exception as e:
        ERROR_COUNTER.inc()
        logger.error(f"More info request failed: {e}")
        raise QueryError(f"Failed to request more information: {str(e)}")

async def conduct_research(state: AgentState) -> dict[str, Any]:
    """Execute research with monitoring and validation."""
    start_time = time.time()
    
    try:
        if not state.steps:
            raise StateError("No research steps available")
            
        result = await researcher_graph.ainvoke({"question": state.steps[0]})
        docs = result.get("documents", [])
        
        logger.info(f"Retrieved {len(docs)} documents for step: {state.steps[0]}")
        QUERY_DURATION.observe(time.time() - start_time)
        
        return {
            "documents": docs,
            "steps": state.steps[1:],
            "state_version": StateVersion(
                timestamp=time.time(),
                checksum=hash(str(docs) + str(state.steps[1:]))
            )
        }
        
    except Exception as e:
        ERROR_COUNTER.inc()
        logger.error(f"Research execution failed: {e}")
        raise QueryError(f"Failed to conduct research: {str(e)}")

def check_finished(state: AgentState) -> Literal["respond", "conduct_research"]:
    """Validate research completion status."""
    try:
        return "respond" if not state.steps else "conduct_research"
    except Exception as e:
        ERROR_COUNTER.inc()
        logger.error(f"Completion check failed: {e}")
        raise StateError(f"Failed to check completion status: {str(e)}")

async def respond_to_general_query(
    state: AgentState,
    *,
    config: RunnableConfig
) -> dict[str, List[BaseMessage]]:
    """Handle general queries with monitoring."""
    try:
        model = ChatOpenAI(model=GPT_4o_MINI, temperature=TEMPERATURE, streaming=True)
        system_prompt = GENERAL_SYSTEM_PROMPT.format(logic=state.router["logic"])
        messages = [{"role": "system", "content": system_prompt}] + state.messages
        
        response = await model.ainvoke(messages)
        return {"messages": [response]}
        
    except Exception as e:
        ERROR_COUNTER.inc()
        logger.error(f"General query response failed: {e}")
        raise QueryError(f"Failed to respond to general query: {str(e)}")

def format_docs(docs: Optional[List[Document]]) -> str:
    """Format documents with validation."""
    if not docs:
        return "<documents></documents>"
        
    try:
        formatted = "\n".join(
            f'<document{" " + " ".join(f"{k}={v!r}" for k, v in (doc.metadata or {}).items()) if doc.metadata else ""}>\n{doc.page_content}\n</document>'
            for doc in docs
        )
        return f"<documents>\n{formatted}\n</documents>"
        
    except Exception as e:
        logger.error(f"Document formatting failed: {e}")
        return "<documents></documents>"

async def check_hallucinations(
    state: AgentState,
    *,
    config: RunnableConfig
) -> dict[str, Any]:
    """Enhanced hallucination detection with metrics."""
    try:
        model = ChatOpenAI(model=GPT_4o_MINI, temperature=TEMPERATURE, streaming=True)
        system_prompt = CHECK_HALLUCINATIONS.format(
            documents=state.documents,
            generation=state.messages[-1]
        )
        
        messages = [{"role": "system", "content": system_prompt}] + state.messages
        response = cast(
            GradeHallucinations,
            await model.with_structured_output(GradeHallucinations).ainvoke(messages)
        )
        
        if response.binary_score == "0":
            HALLUCINATION_COUNTER.inc()
            
        return {"hallucination": response}
        
    except Exception as e:
        ERROR_COUNTER.inc()
        logger.error(f"Hallucination check failed: {e}")
        raise QueryError(f"Failed to check for hallucinations: {str(e)}")

def human_approval(state: AgentState) -> Literal["END", "respond"]:
    """Process human approval with validation."""
    try:
        if state.hallucination.binary_score == "1":
            return "END"
            
        retry_generation = interrupt({
            "question": "Is this correct?",
            "llm_output": state.messages[-1]
        })
        
        return "respond" if retry_generation == "y" else "END"
        
    except Exception as e:
        ERROR_COUNTER.inc()
        logger.error(f"Human approval processing failed: {e}")
        raise StateError(f"Failed to process human approval: {str(e)}")

async def respond(
    state: AgentState,
    *,
    config: RunnableConfig
) -> dict[str, List[BaseMessage]]:
    """Generate final response with monitoring."""
    start_time = time.time()
    
    try:
        model = ChatOpenAI(model=GPT_4o, temperature=TEMPERATURE, streaming=True)
        context = format_docs(state.documents)
        prompt = RESPONSE_SYSTEM_PROMPT.format(context=context)
        messages = [{"role": "system", "content": prompt}] + state.messages
        
        response = await model.ainvoke(messages)
        QUERY_DURATION.observe(time.time() - start_time)
        
        return {"messages": [response]}
        
    except Exception as e:
        ERROR_COUNTER.inc()
        logger.error(f"Response generation failed: {e}")
        raise QueryError(f"Failed to generate response: {str(e)}")

# Initialize graph with enhanced error handling
try:
    checkpointer = MemorySaver()
    builder = StateGraph(AgentState, input=InputState)
    
    # Add nodes
    builder.add_node("analyze_and_route_query", analyze_and_route_query)
    builder.add_edge(START, "analyze_and_route_query")
    builder.add_conditional_edges("analyze_and_route_query", route_query)
    
    builder.add_node("create_research_plan", create_research_plan)
    builder.add_node("ask_for_more_info", ask_for_more_info)
    builder.add_node("respond_to_general_query", respond_to_general_query)
    builder.add_node("conduct_research", conduct_research)
    builder.add_node("respond", respond)
    builder.add_node("check_hallucinations", check_hallucinations)
    
    # Add edges
    builder.add_conditional_edges(
        "check_hallucinations",
        human_approval,
        {"END": END, "respond": "respond"}
    )
    builder.add_edge("create_research_plan", "conduct_research")
    builder.add_conditional_edges("conduct_research", check_finished)
    builder.add_edge("respond", "check_hallucinations")
    
    # Compile graph
    graph = builder.compile(checkpointer=checkpointer)
    logger.info("Graph successfully initialized")
    
except Exception as e:
    ERROR_COUNTER.inc()
    logger.critical(f"Failed to initialize graph: {e}")
    raise RuntimeError(f"Graph initialization failed: {str(e)}")
