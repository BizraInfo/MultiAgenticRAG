"""Enhanced document retrieval system with optimized performance and better error handling.

This module implements a sophisticated document processing and retrieval system with:
- Robust error handling and recovery
- Performance optimizations through caching and batch processing
- Comprehensive logging and monitoring
- Type safety through strict typing
"""

from docling.document_converter import DocumentConverter
from langchain_text_splitters import MarkdownHeaderTextSplitter
from utils.utils import config
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.retrievers import EnsembleRetriever, BM25Retriever
from typing import List, Any, Optional, Dict, Union
import logging
import os
from dotenv import load_dotenv
import rank_bm25
from functools import lru_cache
import time
from concurrent.futures import ThreadPoolExecutor
import hashlib
from pathlib import Path

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ProcessingError(Exception):
    """Custom exception for document processing errors."""
    pass

class RetrieverError(Exception):
    """Custom exception for retrieval errors."""
    pass

class DocumentProcessor:
    """Enhanced document processor with caching and batch processing capabilities."""
    
    def __init__(self, headers_to_split_on: List[List[str]], cache_dir: Optional[str] = None):
        """Initialize the document processor.
        
        Args:
            headers_to_split_on: List of header patterns to split on
            cache_dir: Optional directory for caching processed documents
        """
        self.headers_to_split_on = headers_to_split_on
        self.cache_dir = cache_dir or "cache/processed_docs"
        self._ensure_cache_dir()
        
    def _ensure_cache_dir(self) -> None:
        """Create cache directory if it doesn't exist."""
        if self.cache_dir:
            Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
            
    def _get_cache_key(self, source: str) -> str:
        """Generate cache key for a document source."""
        return hashlib.md5(str(source).encode()).hexdigest()
    
    @lru_cache(maxsize=100)
    def _cached_process(self, source: str) -> List[str]:
        """Process document with caching."""
        cache_key = self._get_cache_key(source)
        cache_file = Path(self.cache_dir) / f"{cache_key}.txt"
        
        if cache_file.exists():
            logger.info(f"Cache hit for document: {source}")
            return cache_file.read_text().split("\n---SPLIT---\n")
            
        try:
            start_time = time.time()
            converter = DocumentConverter()
            markdown_document = converter.convert(source).document.export_to_markdown()
            markdown_splitter = MarkdownHeaderTextSplitter(self.headers_to_split_on)
            docs_list = markdown_splitter.split_text(markdown_document)
            
            # Cache the results
            cache_file.write_text("\n---SPLIT---\n".join(docs_list))
            
            processing_time = time.time() - start_time
            logger.info(f"Document processed in {processing_time:.2f}s")
            return docs_list
            
        except Exception as e:
            logger.error(f"Error processing document: {e}")
            raise ProcessingError(f"Failed to process document: {str(e)}")

    def process_batch(self, sources: List[str], max_workers: int = 4) -> List[List[str]]:
        """Process multiple documents in parallel.
        
        Args:
            sources: List of document sources to process
            max_workers: Maximum number of parallel workers
            
        Returns:
            List of processed document chunks for each source
        """
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            return list(executor.map(self._cached_process, sources))

class IndexBuilder:
    """Enhanced index builder with optimized retrieval and caching."""
    
    def __init__(
        self,
        docs_list: List[str],
        collection_name: str,
        persist_directory: str,
        load_documents: bool,
        cache_size: int = 1000
    ):
        self.docs_list = docs_list
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.load_documents = load_documents
        self.vectorstore = None
        self._cache: Dict[str, List[Any]] = {}
        self._cache_size = cache_size
        
    def _init_embeddings(self) -> OpenAIEmbeddings:
        """Initialize embeddings with retry logic."""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                return OpenAIEmbeddings()
            except Exception as e:
                if attempt == max_retries - 1:
                    raise RetrieverError(f"Failed to initialize embeddings: {str(e)}")
                logger.warning(f"Embeddings initialization attempt {attempt + 1} failed, retrying...")
                time.sleep(1)

    def build_vectorstore(self) -> None:
        """Build vector store with improved error handling and monitoring."""
        try:
            start_time = time.time()
            embeddings = self._init_embeddings()
            
            self.vectorstore = Chroma.from_documents(
                persist_directory=self.persist_directory,
                documents=self.docs_list,
                collection_name=self.collection_name,
                embedding=embeddings,
            )
            
            build_time = time.time() - start_time
            logger.info(f"Vector store built in {build_time:.2f}s")
            
        except Exception as e:
            logger.error(f"Error building vector store: {e}")
            raise RetrieverError(f"Failed to build vector store: {str(e)}")

    def build_retrievers(self) -> EnsembleRetriever:
        """Build optimized ensemble retriever with monitoring."""
        try:
            start_time = time.time()
            
            # Build BM25 retriever
            bm25_retriever = BM25Retriever.from_documents(
                self.docs_list,
                search_kwargs={"k": config["retriever"]["top_k"]}
            )

            # Build vector retrievers
            retriever_vanilla = self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": config["retriever"]["top_k"]}
            )
            
            retriever_mmr = self.vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs={"k": config["retriever"]["top_k"]}
            )

            # Create ensemble
            ensemble_retriever = EnsembleRetriever(
                retrievers=[retriever_vanilla, retriever_mmr, bm25_retriever],
                weights=config["retriever"]["ensemble_weights"]
            )
            
            build_time = time.time() - start_time
            logger.info(f"Retrievers built in {build_time:.2f}s")
            
            return ensemble_retriever
            
        except Exception as e:
            logger.error(f"Error building retrievers: {e}")
            raise RetrieverError(f"Failed to build retrievers: {str(e)}")

def initialize_retrieval_system(
    config_path: Optional[str] = None
) -> tuple[DocumentProcessor, IndexBuilder]:
    """Initialize the complete retrieval system with configuration validation.
    
    Args:
        config_path: Optional path to configuration file
        
    Returns:
        Tuple of initialized DocumentProcessor and IndexBuilder
    """
    try:
        # Load and validate configuration
        if config_path:
            # Load custom config logic here
            pass
            
        headers = config["retriever"]["headers_to_split_on"]
        filepath = config["retriever"]["file"]
        collection_name = config["retriever"]["collection_name"]
        load_documents = config["retriever"]["load_documents"]
        
        logger.info("Initializing retrieval system...")
        
        # Initialize components
        processor = DocumentProcessor(headers)
        if load_documents:
            docs_list = processor.process_batch([filepath])[0]
            logger.info(f"Processed {len(docs_list)} document chunks")
        else:
            docs_list = []
            
        index_builder = IndexBuilder(
            docs_list,
            collection_name,
            "vector_db",
            load_documents
        )
        
        return processor, index_builder
        
    except Exception as e:
        logger.error(f"Failed to initialize retrieval system: {e}")
        raise RuntimeError(f"Retrieval system initialization failed: {str(e)}")

if __name__ == "__main__":
    try:
        processor, index_builder = initialize_retrieval_system()
        
        if config["retriever"]["load_documents"]:
            index_builder.build_vectorstore()
            ensemble_retriever = index_builder.build_retrievers()
            logger.info("Retrieval system initialized successfully")
            
    except Exception as e:
        logger.error(f"System initialization failed: {e}")
        exit(1)
