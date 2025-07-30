from .qdrant import QdrantPipeline, FinancialRAGSystem
import os
from dotenv import load_dotenv

load_dotenv()

qdrant_pipeline_instance = QdrantPipeline(
        qdrant_url = os.getenv("QDRANT_URL"), 
        api_key = os.getenv("QDRANT_API_KEY")
    )
qdrant_pipeline_instance.create_indexes_manually()

# Initialize RAG system
rag_pipeline_instance = FinancialRAGSystem(
    qdrant_pipeline=qdrant_pipeline_instance,
    gemini_api_key = os.getenv("GEMINI_API_KEY")
)