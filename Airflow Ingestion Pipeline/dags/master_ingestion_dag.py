import os
import logging
from airflow.models.dag import DAG
from qdrant_client import QdrantClient
from pendulum import datetime as pendulum_datetime
from airflow.operators.python import PythonOperator
from qdrant_client.http.models import Distance, VectorParams
from qdrant_client.http.exceptions import UnexpectedResponse
from airflow.operators.trigger_dagrun import TriggerDagRunOperator


# --- 1. Configuration ---
# Qdrant Credentials and Collection Details
QDRANT_ENDPOINT = os.getenv("QDRANT_ENDPOINT")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")
VECTOR_PARAMS = VectorParams(size=768, distance=Distance.COSINE)

# Companies data organized by sector
COMPANIES_DATA = {
    "Health": {
        "UnitedHealth Group Incorporated": "UNH",
        "Pfizer Inc.": "PFE",
        "Johnson & Johnson": "JNJ",
        "AbbVie Inc.": "ABBV",
        "Eli Lilly and Company": "LLY"
    },
    "Technology": {
        "Apple Inc.": "AAPL",
        "Microsoft Corporation": "MSFT",
        "Alphabet Inc.": "GOOGL",
        "Meta Platforms Inc.": "META",
        "NVIDIA Corporation": "NVDA"
    },
    "Manufacturing": {
        "Caterpillar Inc.": "CAT",
        "General Electric Company": "GE",
        "3M Company": "MMM",
        "Honeywell International Inc.": "HON",
        "Illinois Tool Works Inc.": "ITW"
    },
    "Defence": {
        "Lockheed Martin Corporation": "LMT",
        "Northrop Grumman Corporation": "NOC",
        "Raytheon Technologies Corporation": "RTX",
        "General Dynamics Corporation": "GD",
        "L3Harris Technologies Inc.": "LHX"
    },
    "Finance": {
        "Capital One Financial Co.": "COF",
        "Bank of America Corporation": "BAC",
        "Wells Fargo & Company": "WFC",
        "Morgan Stanley": "MS"
    }
}

# Extract all tickers for processing
COMPANIES_TO_PROCESS = []
for sector, companies in COMPANIES_DATA.items():
    for company_name, ticker in companies.items():
        COMPANIES_TO_PROCESS.append(ticker)
logging.info(f"Companies to be processed: {COMPANIES_TO_PROCESS}")
logging.info(f"Total companies: {len(COMPANIES_TO_PROCESS)}")

# --- 2. Qdrant Setup Function ---
def setup_qdrant_collection():
    """
    Connects to Qdrant, deletes the existing collection if it exists,
    and creates a fresh new one. This ensures a clean state for the ingestion run.
    """
    logging.info("--- Starting Qdrant Database Setup ---")
    client = QdrantClient(url=QDRANT_ENDPOINT, api_key=QDRANT_API_KEY)

    try:
        # Check if the collection already exists
        collections_response = client.get_collections()
        collection_names = [c.name for c in collections_response.collections]

        if COLLECTION_NAME in collection_names:
            logging.info(f"Collection '{COLLECTION_NAME}' found. Deleting...")
            delete_result = client.delete_collection(collection_name=COLLECTION_NAME)
            if delete_result:
                logging.info(f"Successfully deleted collection '{COLLECTION_NAME}'.")
            else:
                # If deletion fails, we raise an error as the state is uncertain.
                raise ConnectionError(f"API call to delete collection '{COLLECTION_NAME}' failed.")
        else:
            logging.info(f"Collection '{COLLECTION_NAME}' not found. Skipping deletion.")

    except UnexpectedResponse as e:
        logging.error(f"An unexpected error occurred with the Qdrant service: {e}")
        raise
    except Exception as e:
        logging.error(f"A general error occurred during the deletion check: {e}")
        raise

    # Create a new collection. Using recreate_collection is idempotent and safe.
    logging.info(f"Creating a new collection named '{COLLECTION_NAME}'...")
    try:
        # recreate_collection handles both deleting if exists and creating fresh.
        # It's a robust way to ensure the desired state.
        client.recreate_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VECTOR_PARAMS,
        )
        logging.info(f"Successfully created collection '{COLLECTION_NAME}'.")
    except Exception as e:
        logging.error(f"Fatal: Failed to create collection '{COLLECTION_NAME}': {e}")
        raise

    logging.info("--- Qdrant Database Setup Complete ---")


# --- 3. Parent DAG Definition ---
with DAG(
    dag_id='master_ingestion_dag',
    start_date=pendulum_datetime(2025, 1, 1, tz="UTC"), 
    schedule=None,
    catchup=False,
    tags=['finance', 'rag', 'orchestration']
) as dag:

    # Task 1: Set up the Qdrant collection
    setup_qdrant_task = PythonOperator(
        task_id='setup_qdrant_collection',
        python_callable=setup_qdrant_collection,
    )

    # Initialize the dependency chain with the setup task
    last_task = setup_qdrant_task

    # Task 2: Dynamically create and chain TriggerDagRunOperator tasks for each company
    for ticker in COMPANIES_TO_PROCESS:
        trigger_child_dag_task = TriggerDagRunOperator(
            task_id=f'trigger_ingestion_for_{ticker.lower()}',
            trigger_dag_id='finance_rag_ingestion_dag',  # The ID of the DAG you provided
            conf={'company_ticker': ticker},                     # Pass the ticker in the config
            wait_for_completion=True,                            # Ensures sequential execution
            poke_interval=60,                                    # Check status every 60 seconds
            retries=1,                                           # Retry a failed trigger once
        )
                
        # Chain the tasks: last_task -> current_trigger_task
        last_task >> trigger_child_dag_task
        
        # Update last_task to the newly created task for the next loop iteration
        last_task = trigger_child_dag_task