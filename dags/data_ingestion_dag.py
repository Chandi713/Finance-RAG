import logging
import pandas as pd
from edgar import *
from airflow import DAG
from typing import Dict, Any
from datetime import datetime
from dotenv import load_dotenv
from airflow.models import Variable
from airflow.decorators import task
from pendulum import datetime as pendulum_datetime
from sentence_transformers import SentenceTransformer
from airflow.operators.python import get_current_context
from utils import (
    make_document_chunks,
    generate_embeddings,
    get_qdrant_pipeline,
)
load_dotenv()
set_identity("ENTER YOUR EMAIL ID")

logging.basicConfig(level=logging.INFO)

# DAG configuration optimized for financial document processing workflows
with DAG(
    dag_id='finance_rag_ingestion_dag',
    schedule_interval='@daily',  # Daily execution aligns with financial reporting cycles
    start_date=pendulum_datetime(2025, 1, 1, tz="UTC"),
    catchup=False,  # Prevents historical backfill to avoid overwhelming EDGAR API
    description='A DAG to ingest financial reports for RAG',
    tags=['finance', 'rag', 'edgar'],
) as dag:

    @task
    def get_company_reports_and_extract_items() -> Dict[str, Any]:
        """
        Orchestrates EDGAR API interaction for multi-year 10-K report acquisition and item extraction.
        Implements dynamic company selection with hierarchical parameter resolution (DAG run -> Variable).
        Performs immediate in-memory processing to avoid Airflow XCom serialization limitations.
        
        Returns:
            Dict: Serializable structure containing extracted financial items with metadata hierarchy
        """
        # Dynamic parameter resolution: prioritize runtime configuration over static variables
        context = get_current_context()
        dag_run = context.get('dag_run')
        
        # Hierarchical parameter resolution strategy for operational flexibility
        if dag_run and dag_run.conf and 'company_ticker' in dag_run.conf:
            company_ticker = dag_run.conf['company_ticker']
            logging.info(f"Using company_ticker from DAG trigger: {company_ticker}")
        else:
            # Fallback to Airflow Variables for scheduled executions
            company_ticker = Variable.get("company_ticker", default_var="MSFT")
            logging.info(f"Using company_ticker from Airflow Variable: {company_ticker}")
        
        try:
            company = Company(company_ticker)
            # Temporal filtering strategy: current date as upper bound for comprehensive historical coverage
            current_date_str = datetime.now().strftime("%Y-%m-%d")
            company_10k_reports = company.get_filings(
                filing_date=f":{current_date_str}"
            ).filter(form="10-K")
            
            # Initialize with most recent report year for reverse chronological processing
            current_year = int(company_10k_reports.end_date[:4])

            result = {}
            no_of_reports = 0
            
            # Reverse chronological processing: prioritize recent data for RAG relevance
            for report in company_10k_reports:
                try:
                    # Circuit breaker: limit to 5 years for manageable processing scope
                    if no_of_reports == 5:
                        break
                    
                    report_year = current_year
                    
                    # Immediate object materialization to extract items before XCom serialization
                    report_obj = report.obj()
                    item_data = {}
                    
                    # Item-level extraction with granular error handling
                    for item in report_obj.items:
                        try:
                            # Dynamic item access using string formatting for flexibility
                            item_data[f'{item}'] = report_obj[f'{item}']
                            logging.info(f"Extracted {item} for {company_ticker} - {report_year}")
                        except Exception as e:
                            # Item-level failure isolation: continue processing other items
                            logging.warning(f"Failed to extract item {item} for year {report_year}: {e}")

                    # Structured data organization for downstream processing consistency
                    result[str(report_year)] = {
                        "metadata": {
                            "company": company_ticker,
                            "year": str(report_year),
                            "report": "10-K"
                        },
                        "items": item_data
                    }
                    
                    print(f"Processed 10-K for {company_ticker} for year {report_year}")
                    current_year -= 1
                    no_of_reports += 1
                    
                except (ValueError, TypeError) as e:
                    # Year-level failure recovery: continue with next chronological year
                    logging.warning(f"Failed to process report for year {current_year}: {e}")
                    current_year -= 1

        except Exception as e:
            # Company-level failure handling: return empty structure for graceful degradation
            logging.error(f"Failed to fetch reports for {company_ticker}: {e}")
            return {
                "item_dict_per_year": {},
                "company_ticker": company_ticker
            }

        return {
            "item_dict_per_year": result,
            "company_ticker": company_ticker
        }

    @task
    def chunk_documents(data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Applies document chunking strategy across multi-year financial datasets.
        Maintains temporal organization while optimizing chunk sizes for embedding models.
        Implements year-level isolation for processing resilience.
        
        Args:
            data (Dict): Multi-year financial document structure with metadata
            
        Returns:
            Dict: Temporally organized chunks ready for embedding generation
        """
        item_dict = data["item_dict_per_year"]
        ticker = data["company_ticker"]
        chunks_by_year = {}

        # Year-level processing isolation for fault tolerance
        for year, year_data in item_dict.items():
            try:
                # Optimized chunking parameters for financial document characteristics
                chunks = make_document_chunks(
                    year_data["items"],
                    max_chunk_size=7500,  # Balance between context preservation and model limits
                    overlap=200  # Semantic continuity preservation across boundaries
                )
                chunks_by_year[year] = chunks
                logging.info(f"Created chunks for {ticker} - {year}")
            except Exception as e:
                # Year-level failure isolation: continue processing other years
                logging.warning(f"Chunking failed for {ticker} - {year}: {e}")

        return chunks_by_year

    @task
    def store_document_chunks(document_chunks_by_year: Dict[str, Any]) -> Dict[str, Any]:
        """
        Orchestrates embedding generation and vector database ingestion pipeline.
        Implements year-wise processing with comprehensive metrics tracking.
        Provides operational resilience through isolated error handling per temporal segment.
        
        Args:
            document_chunks_by_year (Dict): Temporally organized document chunks
            
        Returns:
            Dict: Processing metrics including success status and ingestion statistics
        """
        qdrant_pipeline = get_qdrant_pipeline()
        logging.info(f"Qdrant Pipeline Instance Fetched")

        # Operational metrics for pipeline monitoring
        total_chunks = 0
        total_items_processed = 0

        # Year-wise processing maintains temporal organization in vector store
        for year, chunks in document_chunks_by_year.items():
            try:
                logging.info(f"Processing year {year}...")
                
                # DataFrame transformation for structured processing pipeline
                df = pd.DataFrame(chunks).T.reset_index().rename(columns={"index": "Items"})
                # Quality gate: filter out empty chunk collections to prevent embedding errors
                df = df[df['chunks'].apply(len) > 0].reset_index(drop=True)
                
                if df.empty:
                    logging.warning(f"No valid chunks found for year {year}, skipping")
                    continue

                logging.info(f"Generating embeddings for {len(df)} items in {year}...")
                # In-place embedding generation for memory efficiency
                generate_embeddings(df)
                
                logging.info(f"Storing {len(df)} items for {year} in Qdrant...")
                # Conservative batch sizing for database connection stability
                qdrant_pipeline.store_to_database(df, batch_size=50)
                
                # Granular metrics collection for operational visibility
                year_chunks = 0
                for item_chunks in df['chunks']:
                    year_chunks += len(item_chunks)
                
                total_chunks += year_chunks
                total_items_processed += len(df)
                
                logging.info(f"Completed year {year}: {len(df)} items, {year_chunks} chunks")
                
            except Exception as e:
                # Year-level failure isolation: partial success is preferable to complete failure
                logging.error(f"Failed to process year {year}: {e}")
                continue

        logging.info(f"Storage completed - {total_items_processed} items, {total_chunks} total chunks stored in Qdrant")
        return {
            "status": "success", 
            "chunk_count": total_chunks,
            "items_processed": total_items_processed
        }
    
    # Task orchestration: linear dependency chain for financial document processing pipeline
    # Airflow's TaskFlow API automatically handles XCom serialization between decorated tasks
    extracted_items = get_company_reports_and_extract_items()
    doc_chunks = chunk_documents(extracted_items)
    store_result = store_document_chunks(doc_chunks)
