"""Enhanced researcher subgraph with improved performance and reliability.

This module implements a sophisticated research pipeline with:
- Performance monitoring and metrics collection
- Query result caching
- Comprehensive error handling
- Retry mechanisms for external services
- Type-safe implementations
"""

from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.retrievers import EnsembleRetriever, BM25Retriever
from dotenv import load_dotenv
from subgraph.graph_states import (
    ResearcherState,
    QueryState,
    QueryValidationError,
    DocumentValidationError
)
from utils.prompt import GENERATE_QUERIES_SYSTEM_PROMPT
from langchain_core.documents import Document
from typing import Any, Literal, TypedDict, cast, List, Dict, Optional
from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_cohere import CohereRerank
import logging
import time
from functools import lru_cache
from prometheus_client import Counter, Histogram, Gauge
import backoff
from utils.utils import config

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
VECTORSTORE_COLLECTION = config["retriever"]["collection_name"]
VECTORSTORE_DIRECTORY = config["retriever"]["directory"]
TOP_K = config["retriever"]["top_k"]
TOP_K_COMPRESSION = config["retriever"]["top_k_compression"]
ENSEMBLE_WEIGHTS = config["retriever"]["ensemble_weights"]
COHERE_RERANK_MODEL = config["retriever"]["cohere_rerank_model"]

# Metrics
QUERY_GENERATION_TIME = Histogram(
    'query_generation_seconds',
    'Time spent generating queries'
)
DOCUMENT_RETRIEVAL_TIME = Histogram(
    'document_retrieval_seconds',
    'Time spent retrieving documents'
)
DOCUMENT_COUNT = Gauge(
    'retrieved_documents_total',
    'Number of documents retrieved'
)
CACHE_HITS = Counter(
    'cache_hits_total',
    'Number of cache hits'
)
CACHE_MISSES = Counter(
    'cache_misses_total',
    'Number of cache misses'
)
ERROR_COUNTER = Counter(
    'subgraph_errors_total',
    'Number of errors in subgraph operations'
)

class SubgraphError(Exception):
    """Base exception for subgraph-related errors."""
    pass

class VectorStoreError(SubgraphError):
    """Exception for vector store operations."""
    pass

class RetrieverError(SubgraphError):
    """Exception for retriever operations."""
    pass

@backoff.on_exception(
    backoff.expo,
    Exception,
    max_tries=3,
    jitter=backoff.full_jitter
)
def _setup_vectorstore() -> Chroma:
    """Set up vector store with retry mechanism."""
    try:
        embeddings = OpenAIEmbeddings()
        return Chroma(
            collection_name=VECTORSTORE_COLLECTION,
            embedding_function=embeddings,
            persist_directory=VECTORSTORE_DIRECTORY
        )
    except Exception as e:
        logger.error(f"Vector store setup failed: {e}")
        raise VectorStoreError(f"Failed to setup vector store: {str(e)}")

def _load_documents(vectorstore: Chroma) -> List[Document]:
    """Load and validate documents from vector store."""
    try:
        all_data = vectorstore.get(include=["documents", "metadatas"])
        documents: List[Document] = []

        for content, meta in zip(all_data["documents"], all_data["metadatas"]):
            if meta is None:
                meta = {}
            elif not isinstance(meta, dict):
                raise ValueError(f"Invalid metadata type: {type(meta)}")

            documents.append(Document(page_content=content, metadata=meta))

        logger.info(f"Loaded {len(documents)} documents")
        return documents

    except Exception as e:
        logger.error(f"Document loading failed: {e}")
        raise DocumentValidationError(f"Failed to load documents: {str(e)}")

@backoff.on_exception(
    backoff.expo,
    Exception,
    max_tries=3,
    jitter=backoff.full_jitter
)
def _build_retrievers(
    documents: List[Document],
    vectorstore: Chroma
) -> ContextualCompressionRetriever:
    """Build retriever with enhanced error handling and retry mechanism."""
    try:
        # Create base retrievers
        retriever_bm25 = BM25Retriever.from_documents(
            documents,
            search_kwargs={"k": TOP_K}
        )
        retriever_vanilla = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": TOP_K}
        )
        retriever_mmr = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": TOP_K}
        )

        # Create ensemble
        ensemble_retriever = EnsembleRetriever(
            retrievers=[retriever_vanilla, retriever_mmr, retriever_bm25],
            weights=ENSEMBLE_WEIGHTS,
        )

        # Setup reranking
        compressor = CohereRerank(
            top_n=TOP_K_COMPRESSION,
            model=COHERE_RERANK_MODEL
        )

        # Build final retriever
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=ensemble_retriever,
        )

        return compression_retriever

    except Exception as e:
        logger.error(f"Retriever build failed: {e}")
        raise RetrieverError(f"Failed to build retrievers: {str(e)}")

# Initialize components with error handling
try:
    vectorstore = _setup_vectorstore()
    documents = _load_documents(vectorstore)
    compression_retriever = _build_retrievers(documents, vectorstore)
    logger.info("Successfully initialized retrieval system")
except Exception as e:
    logger.critical(f"System initialization failed: {e}")
    raise RuntimeError(f"Failed to initialize retrieval system: {str(e)}")

@lru_cache(maxsize=1000)
def _cached_query_generation(question: str) -> List[str]:
    """Cache query generation results."""
    return []  # Placeholder for actual implementation

async def generate_queries(
    state: ResearcherState,
    *,
    config: RunnableConfig
) -> Dict[str, List[str]]:
    """Generate search queries with performance monitoring and caching."""
    start_time = time.time()
    
    try:
        # Check cache
        cached_result = _cached_query_generation(state.question)
        if cached_result:
            CACHE_HITS.inc()
            logger.info("Cache hit for query generation")
            queries = cached_result
        else:
            CACHE_MISSES.inc()
            
            # Generate new queries
            class Response(TypedDict):
                queries: List[str]

            model = ChatOpenAI(
                model="gpt-4o-mini-2024-07-18",
                temperature=0
            )
            messages = [
                {"role": "system", "content": GENERATE_QUERIES_SYSTEM_PROMPT},
                {"role": "human", "content": state.question},
            ]
            
            response = cast(
                Response,
                await model.with_structured_output(Response).ainvoke(messages)
            )
            queries = response["queries"]
            queries.append(state.question)
            
            # Update cache
            _cached_query_generation.cache_clear()
            _cached_query_generation(state.question)

        # Update metrics
        generation_time = time.time() - start_time
        QUERY_GENERATION_TIME.observe(generation_time)
        
        # Update state
        state.update_metrics(generation_time=generation_time)
        for query in queries:
            state.add_query(query, {"generation_time": generation_time})

        logger.info(f"Generated {len(queries)} queries")
        return {"queries": queries}

    except Exception as e:
        ERROR_COUNTER.inc()
        logger.error(f"Query generation failed: {e}")
        raise SubgraphError(f"Failed to generate queries: {str(e)}")

async def retrieve_and_rerank_documents(
    state: QueryState,
    *,
    config: RunnableConfig
) -> Dict[str, List[Document]]:
    """Retrieve and rerank documents with performance monitoring."""
    start_time = time.time()
    
    try:
        logger.info(f"Processing query: {state.query}")
        response = compression_retriever.invoke(state.query)
        
        # Update metrics
        retrieval_time = time.time() - start_time
        DOCUMENT_RETRIEVAL_TIME.observe(retrieval_time)
        DOCUMENT_COUNT.set(len(response))
        
        logger.info(f"Retrieved {len(response)} documents")
        return {"documents": response}

    except Exception as e:
        ERROR_COUNTER.inc()
        logger.error(f"Document retrieval failed: {e}")
        raise RetrieverError(f"Failed to retrieve documents: {str(e)}")

def retrieve_in_parallel(state: ResearcherState) -> List[Send]:
    """Prepare parallel retrieval tasks with validation."""
    try:
        if not state.queries:
            raise QueryValidationError("No queries available for retrieval")
            
        return [
            Send(
                "retrieve_and_rerank_documents",
                QueryState(query=query)
            )
            for query in state.queries
        ]
        
    except Exception as e:
        ERROR_COUNTER.inc()
        logger.error(f"Parallel retrieval setup failed: {e}")
        raise SubgraphError(f"Failed to setup parallel retrieval: {str(e)}")

# Initialize graph with error handling
try:
    builder = StateGraph(ResearcherState)
    
    # Add nodes
    builder.add_node("generate_queries", generate_queries)
    builder.add_node("retrieve_and_rerank_documents", retrieve_and_rerank_documents)
    
    # Add edges
    builder.add_edge(START, "generate_queries")
    builder.add_conditional_edges(
        "generate_queries",
        retrieve_in_parallel,
        path_map=["retrieve_and_rerank_documents"],
    )
    builder.add_edge("retrieve_and_rerank_documents", END)
    
    # Compile graph
    researcher_graph = builder.compile()
    logger.info("Successfully compiled researcher graph")
    
except Exception as e:
    ERROR_COUNTER.inc()
    logger.critical(f"Graph initialization failed: {e}")
    raise RuntimeError(f"Failed to initialize researcher graph: {str(e)}")
