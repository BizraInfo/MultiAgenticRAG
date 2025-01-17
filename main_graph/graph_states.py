"""Enhanced state management system with versioning and validation.

This module implements a robust state management system with:
- State versioning and rollback capability
- Comprehensive state validation
- Type-safe state transitions
- State history tracking
"""

from dataclasses import dataclass, field
from typing import Annotated, Literal, TypedDict, List, Optional, Dict
from datetime import datetime
from langchain_core.documents import Document
from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages
from utils.utils import reduce_docs
from pydantic import BaseModel, Field, validator
import json
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class StateValidationError(Exception):
    """Exception raised for state validation errors."""
    pass

class StateVersion(BaseModel):
    """Track state versions for rollback capability."""
    
    timestamp: float = Field(description="Unix timestamp of state version")
    checksum: int = Field(description="Hash of state data for integrity checking")
    
    @validator("timestamp")
    def validate_timestamp(cls, v):
        """Ensure timestamp is not in the future."""
        if v > datetime.now().timestamp():
            raise ValueError("Timestamp cannot be in the future")
        return v

class StateTransition(BaseModel):
    """Track state transitions for debugging and analysis."""
    
    from_state: str
    to_state: str
    timestamp: float = Field(default_factory=lambda: datetime.now().timestamp())
    metadata: Dict = Field(default_factory=dict)

class Router(TypedDict):
    """Enhanced query classification with validation."""
    
    logic: str
    type: Literal["more-info", "environmental", "general"]

class GradeHallucinations(BaseModel):
    """Enhanced hallucination detection with confidence scoring."""
    
    binary_score: str = Field(
        description="Answer is grounded in the facts, '1' or '0'"
    )
    confidence: float = Field(
        default=1.0,
        description="Confidence score for the binary classification",
        ge=0.0,
        le=1.0
    )
    
    @validator("binary_score")
    def validate_score(cls, v):
        """Ensure binary score is valid."""
        if v not in ["0", "1"]:
            raise ValueError("Binary score must be '0' or '1'")
        return v

@dataclass(kw_only=True)
class InputState:
    """Enhanced input state with validation and history tracking."""
    
    messages: Annotated[List[AnyMessage], add_messages] = field(default_factory=list)
    created_at: float = field(default_factory=lambda: datetime.now().timestamp())
    
    def validate(self) -> None:
        """Validate input state integrity."""
        if not self.messages:
            raise StateValidationError("Messages list cannot be empty")
        if self.created_at > datetime.now().timestamp():
            raise StateValidationError("Creation timestamp cannot be in the future")
    
    def to_dict(self) -> Dict:
        """Convert state to dictionary for serialization."""
        return {
            "messages": [
                {
                    "content": msg.content,
                    "type": msg.type,
                    "additional_kwargs": msg.additional_kwargs
                }
                for msg in self.messages
            ],
            "created_at": self.created_at
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "InputState":
        """Create state from dictionary representation."""
        try:
            return cls(
                messages=[
                    AnyMessage(
                        content=msg["content"],
                        type=msg["type"],
                        additional_kwargs=msg.get("additional_kwargs", {})
                    )
                    for msg in data["messages"]
                ],
                created_at=data["created_at"]
            )
        except Exception as e:
            raise StateValidationError(f"Failed to deserialize state: {str(e)}")

@dataclass(kw_only=True)
class AgentState(InputState):
    """Enhanced agent state with versioning and comprehensive tracking."""
    
    # Core state components
    router: Router = field(
        default_factory=lambda: Router(type="general", logic="")
    )
    steps: List[str] = field(default_factory=list)
    documents: Annotated[List[Document], reduce_docs] = field(default_factory=list)
    hallucination: GradeHallucinations = field(
        default_factory=lambda: GradeHallucinations(binary_score="0")
    )
    
    # State management
    version: StateVersion = field(
        default_factory=lambda: StateVersion(
            timestamp=datetime.now().timestamp(),
            checksum=0
        )
    )
    previous_versions: List[StateVersion] = field(default_factory=list)
    transitions: List[StateTransition] = field(default_factory=list)
    
    def __post_init__(self):
        """Validate state after initialization."""
        self.validate()
        self._update_checksum()
    
    def _update_checksum(self) -> None:
        """Update state version checksum."""
        state_data = {
            "messages": self.to_dict()["messages"],
            "router": self.router,
            "steps": self.steps,
            "documents": [doc.page_content for doc in self.documents],
            "hallucination": self.hallucination.dict()
        }
        self.version.checksum = hash(json.dumps(state_data, sort_keys=True))
    
    def validate(self) -> None:
        """Comprehensive state validation."""
        super().validate()
        
        if not isinstance(self.router, dict):
            raise StateValidationError("Invalid router type")
            
        if "type" not in self.router or self.router["type"] not in [
            "more-info", "environmental", "general"
        ]:
            raise StateValidationError("Invalid router type")
            
        if not isinstance(self.steps, list):
            raise StateValidationError("Steps must be a list")
            
        if not isinstance(self.documents, list):
            raise StateValidationError("Documents must be a list")
            
        if not isinstance(self.hallucination, GradeHallucinations):
            raise StateValidationError("Invalid hallucination type")
    
    def transition_to(self, new_state: str, metadata: Optional[Dict] = None) -> None:
        """Record state transition with metadata."""
        current_state = self.router["type"]
        
        transition = StateTransition(
            from_state=current_state,
            to_state=new_state,
            metadata=metadata or {}
        )
        
        self.transitions.append(transition)
        logger.info(f"State transition: {current_state} -> {new_state}")
    
    def create_checkpoint(self) -> None:
        """Create state checkpoint for potential rollback."""
        self._update_checksum()
        self.previous_versions.append(self.version)
        
        self.version = StateVersion(
            timestamp=datetime.now().timestamp(),
            checksum=self.version.checksum
        )
        
        logger.info(f"Created state checkpoint at {self.version.timestamp}")
    
    def rollback_to(self, timestamp: float) -> bool:
        """Rollback state to previous version.
        
        Args:
            timestamp: Target timestamp to rollback to
            
        Returns:
            bool: True if rollback successful, False otherwise
        """
        for version in reversed(self.previous_versions):
            if version.timestamp == timestamp:
                self.version = version
                logger.info(f"Rolled back state to {timestamp}")
                return True
                
        logger.warning(f"No checkpoint found for timestamp {timestamp}")
        return False
    
    def get_transition_history(self) -> List[Dict]:
        """Get formatted transition history for analysis."""
        return [
            {
                "timestamp": t.timestamp,
                "from_state": t.from_state,
                "to_state": t.to_state,
                "metadata": t.metadata
            }
            for t in self.transitions
        ]
