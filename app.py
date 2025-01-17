"""Enhanced RAG system entry point with improved reliability and monitoring.

This module implements the main application with:
- Graceful shutdown handling
- Health checks and monitoring
- Comprehensive error handling
- Metrics collection
- Dynamic configuration
"""

import asyncio
import time
import signal
import sys
from typing import Optional, Dict, Any
from contextlib import asynccontextmanager
from prometheus_client import start_http_server, Counter, Histogram
import logging
import yaml
from pathlib import Path
from pydantic import BaseModel, Field, validator
from subgraph.graph_states import ResearcherState
from main_graph.graph_states import AgentState
from utils.utils import config, new_uuid
from subgraph.graph_builder import researcher_graph
from main_graph.graph_builder import InputState, graph
from langgraph.types import Command

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Metrics
QUERY_COUNTER = Counter('total_queries', 'Total number of queries processed')
QUERY_DURATION = Histogram('query_duration_seconds', 'Time spent processing queries')
ERROR_COUNTER = Counter('app_errors_total', 'Total number of application errors')
RETRY_COUNTER = Counter('query_retries_total', 'Total number of query retries')

class ConfigurationError(Exception):
    """Exception for configuration-related errors."""
    pass

class AppConfig(BaseModel):
    """Application configuration with validation."""
    
    thread_id: str = Field(default_factory=new_uuid)
    max_retries: int = Field(default=3, ge=1)
    retry_delay: float = Field(default=1.0, ge=0.0)
    shutdown_timeout: float = Field(default=5.0, ge=0.0)
    metrics_port: int = Field(default=8000, ge=0, le=65535)
    
    @validator("retry_delay", "shutdown_timeout")
    def validate_timeouts(cls, v):
        """Ensure timeout values are reasonable."""
        if v > 30.0:
            raise ValueError("Timeout values should not exceed 30 seconds")
        return v

class ApplicationState:
    """Application state management."""
    
    def __init__(self):
        self.is_shutting_down = False
        self.active_tasks: set[asyncio.Task] = set()
        self.config = self._load_config()
    
    def _load_config(self) -> AppConfig:
        """Load and validate configuration."""
        try:
            config_path = Path("config.yaml")
            if not config_path.exists():
                logger.warning("Config file not found, using defaults")
                return AppConfig()
                
            with config_path.open() as f:
                config_data = yaml.safe_load(f)
            
            return AppConfig(**config_data.get("app", {}))
            
        except Exception as e:
            logger.error(f"Configuration loading failed: {e}")
            raise ConfigurationError(f"Failed to load configuration: {str(e)}")
    
    def update_config(self, updates: Dict[str, Any]) -> None:
        """Update configuration at runtime."""
        try:
            current_dict = self.config.dict()
            current_dict.update(updates)
            self.config = AppConfig(**current_dict)
            logger.info("Configuration updated successfully")
        except Exception as e:
            logger.error(f"Configuration update failed: {e}")
            raise ConfigurationError(f"Failed to update configuration: {str(e)}")

class QueryProcessor:
    """Enhanced query processing with retries and monitoring."""
    
    def __init__(self, app_state: ApplicationState):
        self.app_state = app_state
        self.thread = {"configurable": {"thread_id": app_state.config.thread_id}}
    
    async def process_query(
        self,
        query: str,
        retries: Optional[int] = None
    ) -> None:
        """Process query with retries and monitoring."""
        QUERY_COUNTER.inc()
        start_time = time.time()
        
        try:
            input_state = InputState(messages=query)
            retry_count = 0
            max_retries = retries or self.app_state.config.max_retries
            
            while retry_count < max_retries:
                try:
                    async for content, metadata in graph.astream(
                        input=input_state,
                        stream_mode="messages",
                        config=self.thread
                    ):
                        if content.additional_kwargs.get("tool_calls"):
                            print(
                                content.additional_kwargs.get("tool_calls")[0]["function"].get("arguments"),
                                end="",
                                flush=True
                            )
                        if content.content:
                            time.sleep(0.05)
                            print(content.content, end="", flush=True)
                    
                    break  # Success, exit retry loop
                    
                except Exception as e:
                    retry_count += 1
                    RETRY_COUNTER.inc()
                    
                    if retry_count >= max_retries:
                        raise RuntimeError(f"Max retries ({max_retries}) exceeded")
                        
                    logger.warning(f"Query attempt {retry_count} failed: {e}")
                    await asyncio.sleep(self.app_state.config.retry_delay)
            
            # Check for uncertain information
            if len(graph.get_state(self.thread)[-1]) > 0:
                if len(graph.get_state(self.thread)[-1][0].interrupts) > 0:
                    response = input("\nThe response may contain uncertain information. Retry? (y/n): ")
                    if response.lower() == 'y':
                        async for content, metadata in graph.astream(
                            Command(resume=response),
                            stream_mode="messages",
                            config=self.thread
                        ):
                            if content.additional_kwargs.get("tool_calls"):
                                print(
                                    content.additional_kwargs.get("tool_calls")[0]["function"].get("arguments"),
                                    end="",
                                    flush=True
                                )
                            if content.content:
                                time.sleep(0.05)
                                print(content.content, end="", flush=True)
        
        except Exception as e:
            ERROR_COUNTER.inc()
            logger.error(f"Query processing failed: {e}")
            print(f"\nError processing query: {str(e)}")
            
        finally:
            QUERY_DURATION.observe(time.time() - start_time)

class Application:
    """Main application with graceful shutdown and health checks."""
    
    def __init__(self):
        self.state = ApplicationState()
        self.processor = QueryProcessor(self.state)
    
    async def shutdown(self, signal: Optional[signal.Signals] = None):
        """Graceful shutdown handler."""
        if signal:
            logger.info(f"Received exit signal {signal.name}")
            
        self.state.is_shutting_down = True
        logger.info("Initiating graceful shutdown")
        
        # Wait for active tasks
        if self.state.active_tasks:
            logger.info(f"Waiting for {len(self.state.active_tasks)} active tasks")
            try:
                await asyncio.wait_for(
                    asyncio.gather(*self.state.active_tasks),
                    timeout=self.state.config.shutdown_timeout
                )
            except asyncio.TimeoutError:
                logger.warning("Some tasks did not complete gracefully")
        
        logger.info("Shutdown complete")
        sys.exit(0)
    
    def setup_signal_handlers(self) -> None:
        """Setup signal handlers for graceful shutdown."""
        for sig in (signal.SIGTERM, signal.SIGINT):
            asyncio.get_event_loop().add_signal_handler(
                sig,
                lambda s=sig: asyncio.create_task(self.shutdown(s))
            )
    
    @asynccontextmanager
    async def task_manager(self):
        """Manage task lifecycle and cleanup."""
        task = asyncio.current_task()
        if task:
            self.state.active_tasks.add(task)
        try:
            yield
        finally:
            if task:
                self.state.active_tasks.remove(task)
    
    async def run_query(self, query: str) -> None:
        """Run query with task management."""
        async with self.task_manager():
            await self.processor.process_query(query)
    
    async def run(self) -> None:
        """Main application loop."""
        try:
            # Start metrics server
            start_http_server(self.state.config.metrics_port)
            logger.info(f"Metrics server running on port {self.state.config.metrics_port}")
            
            # Setup signal handlers
            self.setup_signal_handlers()
            logger.info("Signal handlers configured")
            
            print("Enter your query (type '-q' to quit):")
            while not self.state.is_shutting_down:
                try:
                    query = input("> ")
                    if query.strip().lower() == "-q":
                        await self.shutdown()
                        break
                        
                    await self.run_query(query)
                    
                except EOFError:
                    await self.shutdown()
                    break
                    
                except KeyboardInterrupt:
                    continue
                    
                except Exception as e:
                    ERROR_COUNTER.inc()
                    logger.error(f"Error in main loop: {e}")
                    print(f"Error: {str(e)}")
            
        except Exception as e:
            ERROR_COUNTER.inc()
            logger.critical(f"Application failed: {e}")
            raise RuntimeError(f"Application failed: {str(e)}")

def main():
    """Application entry point with error handling."""
    try:
        app = Application()
        asyncio.run(app.run())
    except Exception as e:
        logger.critical(f"Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
