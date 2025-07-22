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
set_identity("capcool79@gmail.com")

# Set up logging
logging.basicConfig(level=logging.INFO)

# Define DAG
with DAG(
    dag_id='finance_rag_ingestion_dag',
    schedule_interval='@daily',
    start_date=pendulum_datetime(2025, 1, 1, tz="UTC"),
    catchup=False,
    description='A DAG to ingest financial reports for RAG',
    tags=['finance', 'rag', 'edgar'],
) as dag:

    @task
    def get_company_reports_and_extract_items() -> Dict[str, Any]:
        """Fetch 10-K reports for a company from EDGAR and extract item content."""
        # Check for company_ticker in DAG run configuration
        context = get_current_context()
        dag_run = context.get('dag_run')
        
        if dag_run and dag_run.conf and 'company_ticker' in dag_run.conf:
            company_ticker = dag_run.conf['company_ticker']
            logging.info(f"Using company_ticker from DAG trigger: {company_ticker}")
        else:
            company_ticker = Variable.get("company_ticker", default_var="MSFT")
            logging.info(f"Using company_ticker from Airflow Variable: {company_ticker}")
        
        try:
            company = Company(company_ticker)
            # Fetch 10-K filings up to the current date for broader coverage
            current_date_str = datetime.now().strftime("%Y-%m-%d")
            company_10k_reports = company.get_filings(
                filing_date=f":{current_date_str}"
            ).filter(form="10-K")
            current_year = int(company_10k_reports.end_date[:4])

            result = {}
            no_of_reports = 0
            
            # Process reports directly without storing TenK objects
            for report in company_10k_reports:
                try:
                    if no_of_reports == 5:
                        break
                    
                    # Use periodOfReport to get the fiscal year end
                    report_year = current_year
                    
                    # Get the report object and extract items immediately
                    report_obj = report.obj()
                    item_data = {}
                    
                    # Extract text content from each item
                    for item in report_obj.items:
                        try:
                            item_data[f'{item}'] = report_obj[f'{item}']
                            logging.info(f"Extracted {item} for {company_ticker} - {report_year}")
                        except Exception as e:
                            logging.warning(f"Failed to extract item {item} for year {report_year}: {e}")

                    # Store the extracted data (serializable)
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
                    logging.warning(f"Failed to process report for year {current_year}: {e}")
                    current_year -= 1

        except Exception as e:
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
        """Split item text into manageable chunks."""
        item_dict = data["item_dict_per_year"]
        ticker = data["company_ticker"]
        chunks_by_year = {}

        for year, year_data in item_dict.items():
            try:
                chunks = make_document_chunks(
                    year_data["items"],
                    max_chunk_size=7500,
                    overlap=200
                )
                chunks_by_year[year] = chunks
                logging.info(f"Created chunks for {ticker} - {year}")
            except Exception as e:
                logging.warning(f"Chunking failed for {ticker} - {year}: {e}")

        return chunks_by_year

    @task
    def store_document_chunks(document_chunks_by_year: Dict[str, Any]) -> Dict[str, Any]:
        """Embed and store document chunks into Qdrant."""
        qdrant_pipeline = get_qdrant_pipeline()
        logging.info(f"Qdrant Pipeline Instance Fetched")

        total_chunks = 0
        total_items_processed = 0

        for year, chunks in document_chunks_by_year.items():
            try:
                logging.info(f"Processing year {year}...")
                
                df = pd.DataFrame(chunks).T.reset_index().rename(columns={"index": "Items"})
                df = df[df['chunks'].apply(len) > 0].reset_index(drop=True)
                
                if df.empty:
                    logging.warning(f"No valid chunks found for year {year}, skipping")
                    continue

                logging.info(f"Generating embeddings for {len(df)} items in {year}...")
                generate_embeddings(df)
                
                logging.info(f"Storing {len(df)} items for {year} in Qdrant...")
                # Pass batch_size parameter - smaller batches for better reliability
                qdrant_pipeline.store_to_database(df, batch_size=50)
                
                # Count chunks for this year
                year_chunks = 0
                for item_chunks in df['chunks']:
                    year_chunks += len(item_chunks)
                
                total_chunks += year_chunks
                total_items_processed += len(df)
                
                logging.info(f"Completed year {year}: {len(df)} items, {year_chunks} chunks")
                
            except Exception as e:
                logging.error(f"Failed to process year {year}: {e}")
                # Continue with next year instead of failing completely
                continue

        logging.info(f"Storage completed - {total_items_processed} items, {total_chunks} total chunks stored in Qdrant")
        return {
            "status": "success", 
            "chunk_count": total_chunks,
            "items_processed": total_items_processed
        }
    
    # Define DAG flow - now tasks don't need any parameters as they get context directly
    extracted_items = get_company_reports_and_extract_items()
    doc_chunks = chunk_documents(extracted_items)
    store_result = store_document_chunks(doc_chunks)