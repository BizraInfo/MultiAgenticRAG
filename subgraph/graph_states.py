"""Enhanced states for the researcher subgraph with validation and monitoring.

This module implements robust state management for the researcher subgraph with:
- Comprehensive state validation
- Type safety
- Performance tracking
- State history
"""

from dataclasses import dataclass, field
from typing import Annotated, List, Dict, Optional
from datetime import datetime
from langchain_core.documents import Document
from utils.utils import reduce_docs
from pydantic import BaseModel, Field, validator
import logging
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SubgraphStateError(Exception):
    """Base exception for subgraph state-related errors."""
    pass

class QueryValidationError(SubgraphStateError):
    """Exception for query validation errors."""
    pass

class DocumentValidationError(SubgraphStateError):
    """Exception for document validation errors."""
    pass

class QueryMetrics(BaseModel):
    """Track performance metrics for query processing."""
    
    generation_time: float = Field(description="Time taken to generate queries")
    retrieval_time: float = Field(description="Time taken to retrieve documents")
    document_count: int = Field(description="Number of documents retrieved")
    relevance_scores: List[float] = Field(default_factory=list, description="Document relevance scores")
    
    @validator("generation_time", "retrieval_time")
    def validate_time(cls, v):
        """Ensure time values are positive."""
        if v < 0:
            raise ValueError("Time values must be positive")
        return v

class QueryState(BaseModel):
    """Enhanced query state with validation and metrics."""
    
    query: str = Field(description="Search query string")
    timestamp: float = Field(default_factory=lambda: datetime.now().timestamp())
    metadata: Dict = Field(default_factory=dict)
    
    @validator("query")
    def validate_query(cls, v):
        """Validate query string."""
        if not v.strip():
            raise QueryValidationError("Query string cannot be empty")
        if len(v) > 1000:
            raise QueryValidationError("Query string too long")
        return v.strip()
    
    def to_dict(self) -> Dict:
        """Convert state to dictionary for serialization."""
        return {
            "query": self.query,
            "timestamp": self.timestamp,
            "metadata": self.metadata
        }

@dataclass(kw_only=True)
class ResearcherState:
    """Enhanced researcher state with comprehensive tracking and validation."""
    
    # Core components
    question: str
    queries: List[str] = field(default_factory=list)
    documents: Annotated[List[Document], reduce_docs] = field(default_factory=list)
    
    # Metrics and tracking
    created_at: float = field(default_factory=lambda: datetime.now().timestamp())
    metrics: QueryMetrics = field(
        default_factory=lambda: QueryMetrics(
            generation_time=0.0,
            retrieval_time=0.0,
            document_count=0,
            relevance_scores=[]
        )
    )
    query_history: List[QueryState] = field(default_factory=list)
    
    def __post_init__(self):
        """Validate state after initialization."""
        self.validate()
    
    def validate(self) -> None:
        """Comprehensive state validation."""
        if not self.question.strip():
            raise QueryValidationError("Research question cannot be empty")
            
        if len(self.question) > 1000:
            raise QueryValidationError("Research question too long")
            
        if not isinstance(self.queries, list):
            raise QueryValidationError("Queries must be a list")
            
        if not isinstance(self.documents, list):
            raise DocumentValidationError("Documents must be a list")
            
        # Validate each query
        for query in self.queries:
            if not isinstance(query, str) or not query.strip():
                raise QueryValidationError("Invalid query in list")
    
    def add_query(self, query: str, metadata: Optional[Dict] = None) -> None:
        """Add new query with validation."""
        try:
            query_state = QueryState(
                query=query,
                metadata=metadata or {}
            )
            self.queries.append(query_state.query)
            self.query_history.append(query_state)
            logger.info(f"Added query: {query}")
        except Exception as e:
            logger.error(f"Failed to add query: {e}")
            raise QueryValidationError(f"Failed to add query: {str(e)}")
    
    def update_metrics(
        self,
        generation_time: Optional[float] = None,
        retrieval_time: Optional[float] = None,
        document_count: Optional[int] = None,
        relevance_scores: Optional[List[float]] = None
    ) -> None:
        """Update performance metrics."""
        try:
            if generation_time is not None:
                self.metrics.generation_time = generation_time
            if retrieval_time is not None:
                self.metrics.retrieval_time = retrieval_time
            if document_count is not None:
                self.metrics.document_count = document_count
            if relevance_scores is not None:
                self.metrics.relevance_scores = relevance_scores
                
            logger.info(f"Updated metrics: {self.metrics.dict()}")
        except Exception as e:
            logger.error(f"Failed to update metrics: {e}")
            raise SubgraphStateError(f"Failed to update metrics: {str(e)}")
    
    def to_dict(self) -> Dict:
        """Convert state to dictionary for serialization."""
        return {
            "question": self.question,
            "queries": self.queries,
            "documents": [
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata
                }
                for doc in self.documents
            ],
            "created_at": self.created_at,
            "metrics": self.metrics.dict(),
            "query_history": [
                query.to_dict() for query in self.query_history
            ]
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "ResearcherState":
        """Create state from dictionary representation."""
        try:
            return cls(
                question=data["question"],
                queries=data["queries"],
                documents=[
                    Document(
                        page_content=doc["content"],
                        metadata=doc.get("metadata", {})
                    )
                    for doc in data["documents"]
                ],
                created_at=data["created_at"],
                metrics=QueryMetrics(**data["metrics"]),
                query_history=[
                    QueryState(**query) for query in data["query_history"]
                ]
            )
        except Exception as e:
            logger.error(f"Failed to deserialize state: {e}")
            raise SubgraphStateError(f"Failed to deserialize state: {str(e)}")
