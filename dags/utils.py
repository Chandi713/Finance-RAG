import os
import ast
import json
import time
import logging
from edgar import *
import pandas as pd
import urllib.request
from typing import Any
from typing import Optional
from dataclasses import dataclass
from qdrant_client import QdrantClient
from datasets import concatenate_datasets
from sentence_transformers.util import cos_sim
from sentence_transformers import SentenceTransformer
from qdrant_client.models import Distance, VectorParams, PointStruct

set_identity("ENTER YOUR EMAIL ID")

# Global instances using lazy initialization pattern to optimize resource usage
_model_instance: Optional[SentenceTransformer] = None
_qdrant_client_instance: Optional[QdrantClient] = None

def get_model() -> SentenceTransformer:
    """
    Retrieves or initializes a domain-specific financial sentence transformer model.
    Uses lazy initialization with global caching to prevent redundant model loading.
    
    Returns:
        SentenceTransformer: Pre-trained financial domain model with extended sequence length
    """
    global _model_instance
    if _model_instance is None:
        # Load specialized financial domain model for better semantic understanding of financial texts
        _model_instance = SentenceTransformer(
            "Yaksh170802/gte-finance-model", trust_remote_code=True
        )
        # Extended sequence length to accommodate longer financial document chunks
        _model_instance.max_seq_length = 2000
    logging.info("Model instance retrieved")
    return _model_instance

def get_qdrant_pipeline() -> "QdrantPipeline":
    """
    Implements singleton pattern for Qdrant vector database connection management.
    Ensures single database connection throughout application lifecycle for efficiency.
    
    Returns:
        QdrantPipeline: Singleton instance of the vector database pipeline
    """
    global _qdrant_client_instance
    if _qdrant_client_instance is None:
        _qdrant_client_instance = QdrantPipeline(
            qdrant_url=os.getenv("QDRANT_ENDPOINT"),
            api_key=os.getenv("QDRANT_API_KEY")
        )
    return _qdrant_client_instance


class QdrantPipeline:
    """
    Handles vector database operations for financial document retrieval system.
    Manages collection creation, indexing strategy, and batch data ingestion.
    """
    
    def __init__(self, qdrant_url, api_key, collection_name="Finance_RAG_DB"):
        # Extended timeout for handling large financial document batches
        self.client = QdrantClient(url=qdrant_url, api_key=api_key, timeout=600)
        self.collection_name = collection_name
        
    def create_collection(self, vector_size=768):
        """
        Initializes vector collection with optimized configuration for financial document retrieval.
        Creates payload indexes to enable efficient filtering by company and temporal dimensions.
        
        Args:
            vector_size (int): Dimensionality of embedding vectors (default: 768 for transformer models)
        """
        # COSINE distance optimal for normalized embeddings from transformer models
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
        )
        
        # Strategic indexing for common financial document filtering patterns
        try:
            # Keyword index for exact company name matching in financial queries
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="company",
                field_schema="keyword"
            )
            
            # Integer index for efficient temporal filtering (year-based analysis)
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="year", 
                field_schema="integer"
            )
            
            print("Created payload indexes for company and year filtering")
            
        except Exception as e:
            # Graceful degradation: system continues with fallback filtering if indexing fails
            print(f"Warning: Could not create indexes: {e}")
            print("You may need to create indexes manually or filtering will use fallback method")
    
    def store_to_database(self, df, batch_size: int = 50):
        """
        Performs batch ingestion of financial document embeddings with robust error handling.
        Implements chunked processing to handle memory constraints and connection timeouts.
        
        Args:
            df (DataFrame): Structured data containing chunks, embeddings, and metadata
            batch_size (int): Number of vectors per batch to optimize upload performance
        """
        all_points = []
        # Timestamp-based ID generation ensures uniqueness across batch uploads
        point_id = int(time.time())
        
        logging.info(f"Processing {len(df)} dataframe rows...")
        
        for idx, row in df.iterrows():
            try:
                # Defensive parsing of metadata with multiple fallback strategies
                metadata = row['metadata']
                if isinstance(metadata, str):
                    try:
                        # Primary: JSON parsing for structured metadata
                        metadata = json.loads(metadata)
                    except:
                        try:
                            # Fallback: AST literal evaluation for Python literal structures
                            metadata = ast.literal_eval(metadata)
                        except:
                            logging.warning(f"Warning: Could not parse metadata for row {idx}, using empty dict")
                            metadata = {}
                            
                elif not isinstance(metadata, dict):
                    logging.warning(f"Warning: Metadata for row {idx} is not a dict or string, using empty dict")
                    metadata = {}
                
                # Safe deserialization of embedding vectors
                vectors = row['Encodings']
                if isinstance(vectors, str):
                    try:
                        vectors = ast.literal_eval(vectors)
                    except:
                        logging.warning(f"Warning: Could not parse vectors for row {idx}, skipping")
                        continue
                
                # Normalize chunks to list format for consistent processing
                chunks = row['chunks']
                if isinstance(chunks, str):
                    try:
                        chunks = ast.literal_eval(chunks)
                    except:
                        chunks = [chunks]
                elif not isinstance(chunks, list):
                    chunks = [chunks]
                
                # Vector-chunk alignment: each embedding corresponds to its text chunk
                for vector, chunk in zip(vectors, chunks):
                    point = PointStruct(
                        id=point_id,
                        vector=vector,
                        payload={
                            'chunks': str(chunk),  # String normalization for consistent querying
                            'company': metadata.get('company', 'Unknown'),
                            'year': metadata.get('year', 'Unknown'),
                            'report': metadata.get('report', 'Unknown'),
                            'item': metadata.get('item', 'Unknown'),
                            'source_file': f"dataframe_row_{idx}"
                        }
                    )
                    all_points.append(point)
                    point_id += 1
                    
            except Exception as e:
                logging.error(f"Error processing row {idx}: {e}")
                continue
        
        if not all_points:
            logging.warning("No valid points to upload")
            return
        
        # Batch processing to mitigate database connection timeout constraints
        logging.info(f"Uploading {len(all_points)} points in batches of {batch_size}")
        total_batches = (len(all_points) + batch_size - 1) // batch_size
        
        for i in range(0, len(all_points), batch_size):
            batch = all_points[i:i + batch_size]
            batch_num = (i // batch_size) + 1
            
            try:
                logging.info(f"Uploading batch {batch_num}/{total_batches} ({len(batch)} points)")
                
                # Synchronous upsert with wait=True ensures data consistency
                self.client.upsert(
                    collection_name=self.collection_name, 
                    points=batch,
                    wait=True
                )
                
                logging.info(f"Successfully uploaded batch {batch_num}")
                # Rate limiting to prevent overwhelming the database connection
                time.sleep(1)
                
            except Exception as e:
                logging.error(f"Batch {batch_num} failed: {e}")
                raise
        
        logging.info(f"Successfully uploaded {len(all_points)} points from dataframe")
        
        # Post-ingestion index verification for optimal query performance
        self._ensure_indexes()

    def _ensure_indexes(self):
        """
        Post-ingestion index verification to guarantee optimal query performance.
        Implements graceful degradation if index creation fails after data upload.
        """
        try:
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="company",
                field_schema="keyword"
            )
            
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="year",
                field_schema="integer"  
            )
            
            logging.info("Ensured payload indexes exist for company and year filtering")
            
        except Exception as e:
            # Index creation may fail if indexes already exist - this is expected behavior
            logging.warning(f"Note: Indexes may already exist: {e}")
            logging.warning("Filtering will use fallback method if needed")
            

def recursive_chunking(text, max_chunk_size, overlap):
    """
    Applies recursive character-based text splitting with semantic awareness.
    Preserves document structure while maintaining optimal chunk sizes for embedding models.
    
    Args:
        text (str): Input text to be chunked
        max_chunk_size (int): Maximum characters per chunk
        overlap (int): Character overlap between consecutive chunks
        
    Returns:
        list: Text chunks optimized for semantic embedding
    """
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        # Recursive splitter respects document structure (sentences, paragraphs) while enforcing size limits
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=max_chunk_size,
            chunk_overlap=overlap,
            length_function=len
        )
        texts = text_splitter.split_text(text)
        return texts
    except Exception as e:
        logging.error(f"Error in recursive_chunking: {e}")
        # Failsafe: return original text as single chunk to prevent data loss
        return [text]


def make_document_chunks(item_dict, max_chunk_size, overlap):
    """
    Orchestrates document processing pipeline with overlap-based continuity preservation.
    Creates both standard and overlapped chunks to maintain semantic continuity across boundaries.
    
    Args:
        item_dict (dict): Document items with metadata
        max_chunk_size (int): Target chunk size for optimal embedding performance
        overlap (int): Overlap size to preserve context across chunk boundaries
        
    Returns:
        dict: Processed chunks with preserved metadata and cross-boundary context
    """
    chunk_dict = {}
    
    try:
        # Sequential processing prevents memory overflow with large financial documents
        for item_name, text_content in item_dict.items():
            if item_name == 'metadata':
                continue  # Metadata is structural, not content for embedding
                
            if not isinstance(text_content, str):
                logging.warning(f"Skipping non-text item: {item_name}")
                continue
                
            # Primary chunking strategy: recursive character splitting
            recursive_chunks = recursive_chunking(text_content, max_chunk_size, overlap)
            
            # Secondary chunking: overlap-based continuity preservation for multi-chunk documents
            overlapped_chunks = []
            if len(recursive_chunks) > 1:
                # Create bridge chunks that span chunk boundaries to preserve context
                overlapped_chunks = [
                    recursive_chunks[i][-overlap:] + '\n' + recursive_chunks[i+1][:overlap] 
                    for i in range(0, len(recursive_chunks)-1)
                ]
            
            # Metadata inheritance pattern: preserve document-level metadata at chunk level
            item_metadata = {}
            if 'metadata' in item_dict and isinstance(item_dict['metadata'], dict):
                item_metadata = item_dict['metadata'].copy()
            item_metadata['item'] = item_name  # Item-level identification for retrieval
            
            # Dual-strategy chunking: standard + overlap chunks for comprehensive coverage
            chunk_dict[item_name] = {
                "chunks": recursive_chunks + overlapped_chunks,
                "metadata": item_metadata
            }
    except Exception as e:
        logging.error(f"Error in make_document_chunks: {e}")
    
    return chunk_dict


def generate_embeddings(df):
    """
    Orchestrates embedding generation with hybrid cloud-local fallback strategy.
    Prioritizes Azure cloud embeddings with automatic fallback to local model for resilience.
    
    Args:
        df (DataFrame): Document chunks requiring vectorization
        
    Returns:
        None: Modifies DataFrame in-place by adding 'Encodings' column
    """
    document_encodings = []
    logging.info("Generating Model Encodings")
    
    for index, value in df.iterrows():
        # Structured payload for Azure ML endpoint compatibility
        data = {"texts": value["chunks"]}
        body = str.encode(json.dumps(data))
        url = os.getenv('AZURE_ENDPOINT')
        api_key = os.getenv('AZURE_API_KEY')
        
        # Environment validation for cloud service dependencies
        if not url:
            raise Exception("An endpoint URL should be provided to invoke the endpoint")
        if not api_key:
            raise Exception("A key should be provided to invoke the endpoint")

        try:
            # Primary strategy: Azure ML endpoint for scalable cloud-based embeddings
            headers = {'Content-Type':'application/json', 'Accept': 'application/json', 'Authorization':('Bearer '+ api_key)}
            req = urllib.request.Request(url, body, headers)
            response = urllib.request.urlopen(req)
            result = response.read()
            # Double JSON parsing required for Azure ML response format
            intermediate_result = json.loads(json.loads(result.decode('utf-8')))
            encodings = intermediate_result["embeddings"]
            document_encodings.append(encodings)
            
        except urllib.error.HTTPError as error:
            # Fallback strategy: local model ensures system resilience during cloud outages
            print("The request failed with status code: " + str(error.code))
            print("Continuing with locally available model")
            model = get_model()
            # Batch processing for memory efficiency with local model
            encodings = model.encode(
                value["chunks"],
                batch_size=16)
            document_encodings.append(encodings.tolist())
        
        logging.info(f"Generated encodings for row {index+1}")
        
    logging.info("Completed generating encodings")
    # In-place DataFrame modification for memory efficiency
    df["Encodings"] = document_encodings
