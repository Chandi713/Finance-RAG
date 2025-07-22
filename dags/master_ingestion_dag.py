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
# Vector database configuration for financial document embeddings
QDRANT_ENDPOINT = os.getenv("QDRANT_ENDPOINT")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")
# COSINE distance optimal for normalized transformer embeddings (768-dim financial model)
VECTOR_PARAMS = VectorParams(size=768, distance=Distance.COSINE)

# Sector-based portfolio organization for systematic financial analysis coverage
# Strategic selection covers major economic sectors with high-liquidity companies
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

# Flatten hierarchical structure for linear processing pipeline
COMPANIES_TO_PROCESS = []
for sector, companies in COMPANIES_DATA.items():
    for company_name, ticker in companies.items():
        COMPANIES_TO_PROCESS.append(ticker)
logging.info(f"Companies to be processed: {COMPANIES_TO_PROCESS}")
logging.info(f"Total companies: {len(COMPANIES_TO_PROCESS)}")

# --- 2. Qdrant Setup Function ---
def setup_qdrant_collection():
    """
    Orchestrates vector database initialization with idempotent collection management.
    Implements defensive database state management through existence verification and cleanup.
    Ensures pristine collection state for consistent multi-company ingestion workflows.
    
    Raises:
        ConnectionError: When collection deletion fails, indicating uncertain database state
        UnexpectedResponse: When Qdrant service encounters unexpected conditions
    """
    logging.info("--- Starting Qdrant Database Setup ---")
    client = QdrantClient(url=QDRANT_ENDPOINT, api_key=QDRANT_API_KEY)

    try:
        # Defensive state verification: check existing collections before operations
        collections_response = client.get_collections()
        collection_names = [c.name for c in collections_response.collections]

        if COLLECTION_NAME in collection_names:
            logging.info(f"Collection '{COLLECTION_NAME}' found. Deleting...")
            delete_result = client.delete_collection(collection_name=COLLECTION_NAME)
            if delete_result:
                logging.info(f"Successfully deleted collection '{COLLECTION_NAME}'.")
            else:
                # Fail-fast pattern: uncertain state is worse than clean failure
                raise ConnectionError(f"API call to delete collection '{COLLECTION_NAME}' failed.")
        else:
            logging.info(f"Collection '{COLLECTION_NAME}' not found. Skipping deletion.")

    except UnexpectedResponse as e:
        # Qdrant-specific exception handling for service-level issues
        logging.error(f"An unexpected error occurred with the Qdrant service: {e}")
        raise
    except Exception as e:
        logging.error(f"A general error occurred during the deletion check: {e}")
        raise

    # Idempotent collection creation using atomic recreate operation
    logging.info(f"Creating a new collection named '{COLLECTION_NAME}'...")
    try:
        # recreate_collection provides atomicity: handles both deletion and creation
        # More robust than separate delete/create operations for distributed systems
        client.recreate_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VECTOR_PARAMS,
        )
        logging.info(f"Successfully created collection '{COLLECTION_NAME}'.")
    except Exception as e:
        # Fatal error classification: collection creation failure blocks entire pipeline
        logging.error(f"Fatal: Failed to create collection '{COLLECTION_NAME}': {e}")
        raise

    logging.info("--- Qdrant Database Setup Complete ---")


# --- 3. Parent DAG Definition ---
# Manual trigger strategy for controlled multi-company ingestion workflows
with DAG(
    dag_id='master_ingestion_dag',
    start_date=pendulum_datetime(2025, 1, 1, tz="UTC"), 
    schedule=None,  # Manual trigger only: prevents unintended automated execution
    catchup=False,  # No historical backfill to avoid overwhelming EDGAR API and vector DB
    tags=['finance', 'rag', 'orchestration']
) as dag:

    # Task 1: Database state preparation before multi-company processing
    setup_qdrant_task = PythonOperator(
        task_id='setup_qdrant_collection',
        python_callable=setup_qdrant_collection,
    )

    # Sequential dependency chain initialization
    last_task = setup_qdrant_task

    # Task 2: Dynamic task generation with sequential execution strategy
    # Creates linear dependency chain for controlled resource utilization
    for ticker in COMPANIES_TO_PROCESS:
        trigger_child_dag_task = TriggerDagRunOperator(
            task_id=f'trigger_ingestion_for_{ticker.lower()}',
            trigger_dag_id='finance_rag_ingestion_dag',  # Target child DAG for individual company processing
            conf={'company_ticker': ticker},             # Dynamic parameter injection for each company
            wait_for_completion=True,                    # Synchronous execution: prevents resource overwhelming
            poke_interval=60,                            # Polling frequency balance: responsiveness vs. system load
            retries=1,                                   # Limited retry policy: prevents infinite failure loops
        )
                
        # Linear dependency chaining: ensures sequential company processing
        # Critical for EDGAR API rate limiting and vector database stability
        last_task >> trigger_child_dag_task
        
        # Dependency chain propagation: each task becomes predecessor for next iteration
        last_task = trigger_child_dag_task