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

set_identity("capcool79@gmail.com")

_model_instance: Optional[SentenceTransformer] = None
_qdrant_client_instance: Optional[QdrantClient] = None

# Defining helper functions
def get_model() -> SentenceTransformer:
    global _model_instance
    if _model_instance is None:
        _model_instance = SentenceTransformer(
            "Yaksh170802/gte-finance-model", trust_remote_code=True
        )
        _model_instance.max_seq_length = 2000
    logging.info("Model instance retrieved")
    return _model_instance

# Using Singleton Design Pattern to set Qdrant client instance just once and use repeatedly
def get_qdrant_pipeline() -> "QdrantPipeline":
    global _qdrant_client_instance
    if _qdrant_client_instance is None:
        _qdrant_client_instance = QdrantPipeline(
            qdrant_url=os.getenv("QDRANT_ENDPOINT"),
            api_key=os.getenv("QDRANT_API_KEY")
        )
    return _qdrant_client_instance


class QdrantPipeline:
    def __init__(self, qdrant_url, api_key, collection_name="Finance_RAG_DB"):
        self.client = QdrantClient(url=qdrant_url, api_key=api_key, timeout=600)
        self.collection_name = collection_name
        
    def create_collection(self, vector_size=768):
        """Create collection with payload indexes for company and year filtering"""
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
        )
        
        # Creating payload indexes for filtering
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
            
            print("Created payload indexes for company and year filtering")
            
        except Exception as e:
            print(f"Warning: Could not create indexes: {e}")
            print("You may need to create indexes manually or filtering will use fallback method")
    
    def store_to_database(self, df, batch_size: int = 50):
        """Upload dataframe to Qdrant with simple batch processing."""
        all_points = []
        point_id = int(time.time())  # Simple ID generation using timestamp
        
        logging.info(f"Processing {len(df)} dataframe rows...")
        
        for idx, row in df.iterrows():
            try:
                # Parse metadata
                metadata = row['metadata']
                if isinstance(metadata, str):
                    try:
                        metadata = json.loads(metadata)
                    except:
                        try:
                            metadata = ast.literal_eval(metadata)
                        except:
                            logging.warning(f"Warning: Could not parse metadata for row {idx}, using empty dict")
                            metadata = {}
                            
                elif not isinstance(metadata, dict):
                    logging.warning(f"Warning: Metadata for row {idx} is not a dict or string, using empty dict")
                    metadata = {}
                
                # Parse vectors
                vectors = row['Encodings']
                if isinstance(vectors, str):
                    try:
                        vectors = ast.literal_eval(vectors)
                    except:
                        logging.warning(f"Warning: Could not parse vectors for row {idx}, skipping")
                        continue
                
                # Parse chunks
                chunks = row['chunks']
                if isinstance(chunks, str):
                    try:
                        chunks = ast.literal_eval(chunks)
                    except:
                        chunks = [chunks]
                elif not isinstance(chunks, list):
                    chunks = [chunks]
                
                # Creating points for each vector-chunk pair
                for vector, chunk in zip(vectors, chunks):
                    point = PointStruct(
                        id=point_id,
                        vector=vector,
                        payload={
                            'chunks': str(chunk),  # Ensuring chunk is string
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
        
        # Simple batch upload to resolve the database connection time-limit issue
        logging.info(f"Uploading {len(all_points)} points in batches of {batch_size}")
        total_batches = (len(all_points) + batch_size - 1) // batch_size
        
        for i in range(0, len(all_points), batch_size):
            batch = all_points[i:i + batch_size]
            batch_num = (i // batch_size) + 1
            
            try:
                logging.info(f"Uploading batch {batch_num}/{total_batches} ({len(batch)} points)")
                
                self.client.upsert(
                    collection_name=self.collection_name, 
                    points=batch,
                    wait=True
                )
                
                logging.info(f"Successfully uploaded batch {batch_num}")
                time.sleep(1)  # Simple delay between batches
                
            except Exception as e:
                logging.error(f"Batch {batch_num} failed: {e}")
                raise
        
        logging.info(f"Successfully uploaded {len(all_points)} points from dataframe")
        
        # Ensure indexes exist after upload
        self._ensure_indexes()

    def _ensure_indexes(self):
        """Ensure payload indexes exist after data upload"""
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
            logging.warning(f"Note: Indexes may already exist: {e}")
            logging.warning("Filtering will use fallback method if needed")
            

def recursive_chunking(text, max_chunk_size, overlap):
    """Split text into chunks using recursive character splitting."""
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=max_chunk_size,
            chunk_overlap=overlap,
            length_function=len
        )
        texts = text_splitter.split_text(text)
        return texts
    except Exception as e:
        logging.error(f"Error in recursive_chunking: {e}")
        return [text]  # Return original text as single chunk if splitting fails


def make_document_chunks(item_dict, max_chunk_size, overlap):
    """Create document chunks from items with overlap."""
    chunk_dict = {}
    
    try:
        # Process items sequentially to avoid overwhelming the API
        for item_name, text_content in item_dict.items():
            if item_name == 'metadata':
                continue  # Skip metadata entry as it's not text content
                
            if not isinstance(text_content, str):
                logging.warning(f"Skipping non-text item: {item_name}")
                continue
                
            # Create recursive chunks
            recursive_chunks = recursive_chunking(text_content, max_chunk_size, overlap)
            
            # Create overlapped chunks if there's more than one chunk
            overlapped_chunks = []
            if len(recursive_chunks) > 1:
                overlapped_chunks = [
                    recursive_chunks[i][-overlap:] + '\n' + recursive_chunks[i+1][:overlap] 
                    for i in range(0, len(recursive_chunks)-1)
                ]
            
            # Copy metadata and add item name
            item_metadata = {}
            if 'metadata' in item_dict and isinstance(item_dict['metadata'], dict):
                item_metadata = item_dict['metadata'].copy()
            item_metadata['item'] = item_name
            
            # Store chunks and metadata
            chunk_dict[item_name] = {
                "chunks": recursive_chunks + overlapped_chunks,
                "metadata": item_metadata
            }
    except Exception as e:
        logging.error(f"Error in make_document_chunks: {e}")
    
    return chunk_dict


def generate_embeddings(df):
    document_encodings = []
    logging.info("Generating Model Encodings")
    for index, value in df.iterrows():
        data = {"texts": value["chunks"]}
        body = str.encode(json.dumps(data))
        url = os.getenv('AZURE_ENDPOINT')
        api_key = os.getenv('AZURE_API_KEY')
        if not url:
            raise Exception("An endpoint URL should be provided to invoke the endpoint")
        if not api_key:
            raise Exception("A key should be provided to invoke the endpoint")

        try:
            headers = {'Content-Type':'application/json', 'Accept': 'application/json', 'Authorization':('Bearer '+ api_key)}
            req = urllib.request.Request(url, body, headers)
            response = urllib.request.urlopen(req)
            result = response.read()
            intermediate_result = json.loads(json.loads(result.decode('utf-8')))
            encodings = intermediate_result["embeddings"]
            document_encodings.append(encodings)
        except urllib.error.HTTPError as error:
            print("The request failed with status code: " + str(error.code))
            print("Continuing with locally available model")
            model = get_model()
            encodings = model.encode(
                value["chunks"],
                batch_size=16)
            document_encodings.append(encodings.tolist())
        
        logging.info(f"Generated encodings for row {index+1}")
    logging.info("Completed generating encodings")
    df["Encodings"] = document_encodings