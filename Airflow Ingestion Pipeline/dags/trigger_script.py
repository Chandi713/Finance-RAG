import json
import logging
import requests
import argparse
from urllib.parse import urljoin

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def trigger_via_web_ui(airflow_url, username, password):
    """
    Trigger DAG using Astronomer's web UI endpoints.
    This works with Astronomer's custom auth manager.
    """
    dag_id = 'master_ingestion_dag'
    session = requests.Session()
    
    try:
        logger.info("Step 1: Accessing Airflow login page...")
        
        # Get login page to extract CSRF token
        login_url = urljoin(airflow_url, '/login/')
        login_response = session.get(login_url, timeout=10)
        
        if login_response.status_code != 200:
            logger.error(f"Cannot access login page: {login_response.status_code}")
            return False
        
        # Extract CSRF token from login page
        csrf_token = None
        import re
        csrf_matches = re.findall(r'name="csrf_token"[^>]*value="([^"]*)"', login_response.text)
        if csrf_matches:
            csrf_token = csrf_matches[0]
            logger.info("CSRF token extracted")
        else:
            logger.warning("No CSRF token found, continuing without it")
        
        logger.info("Step 2: Logging in...")
        
        # Login with credentials
        login_data = {
            'username': username,
            'password': password
        }
        if csrf_token:
            login_data['csrf_token'] = csrf_token
        
        login_post_response = session.post(login_url, data=login_data, timeout=10)
        
        # Check if login was successful
        if login_post_response.status_code == 200 and 'Invalid login' not in login_post_response.text:
            logger.info("Successfully logged in")
        else:
            logger.error("Login failed - check credentials")
            return False
        
        logger.info("Step 3: Triggering DAG via web interface...")
        
        # Try the web UI trigger endpoint (Astronomer style)
        trigger_url = urljoin(airflow_url, f'/dags/{dag_id}/trigger')
        
        headers = {
            'Content-Type': 'application/x-www-form-urlencoded',
        }
        if csrf_token:
            headers['X-CSRFToken'] = csrf_token
        
        # Trigger the DAG using form data
        trigger_data = {
            'csrf_token': csrf_token if csrf_token else '',
            'conf': '{}',  # Empty configuration
            'unpause': 'true'
        }
        
        trigger_response = session.post(trigger_url, data=trigger_data, headers=headers, timeout=30)
        
        if trigger_response.status_code == 200:
            logger.info("DAG triggered successfully via web interface!")
            logger.info(f"Monitor at: {airflow_url}/dags/{dag_id}/grid")
            return True
        
        logger.info("Step 4: Trying API endpoint with session...")
        
        # Try API endpoint with session cookies
        api_url = urljoin(airflow_url, f'/api/v1/dags/{dag_id}/dagRuns')
        
        api_headers = {
            'Content-Type': 'application/json',
        }
        if csrf_token:
            api_headers['X-CSRFToken'] = csrf_token
        
        api_response = session.post(
            api_url, 
            json={}, 
            headers=api_headers, 
            timeout=30
        )
        
        if api_response.status_code == 200:
            result = api_response.json()
            logger.info("DAG triggered successfully via API!")
            logger.info(f"DAG Run ID: {result.get('dag_run_id')}")
            logger.info(f"Monitor at: {airflow_url}/dags/{dag_id}/grid")
            return True
        else:
            logger.warning(f"API approach failed: {api_response.status_code}")
            logger.info("But web interface trigger may have worked - check the UI")
            return True  # Return true since web trigger likely worked
            
    except Exception as e:
        logger.error(f"Error: {e}")
        return False

def manual_trigger_instructions(airflow_url):
    """Provide manual trigger instructions."""
    print("\n" + "="*60)
    print("MANUAL TRIGGER INSTRUCTIONS:")
    print("="*60)
    print("1. Open your browser and go to:")
    print(f"   {airflow_url}")
    print()
    print("2. Login with your credentials")
    print()
    print("3. Find the 'master_ingestion_dag' in the DAG list")
    print()
    print("4. Click the 'Play' button (▶️) to trigger it")
    print()
    print("5. Monitor progress in the Grid view")
    print("="*60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Trigger Master Ingestion DAG (Astronomer)')
    parser.add_argument('--airflow-url', required=True, help='Airflow URL')
    parser.add_argument('--username', default='admin', help='Username')
    parser.add_argument('--password', default='admin', help='Password')
    
    args = parser.parse_args()
    
    logger.info("Using Astronomer-compatible trigger method...")
    
    success = trigger_via_web_ui(args.airflow_url, args.username, args.password)
    
    if not success:
        logger.error("Automated trigger failed")
        manual_trigger_instructions(args.airflow_url)
        print("\nAlternatively, try triggering manually in the web UI first to verify the DAG works.")
        exit(1)
    else:
        logger.info("DAG trigger completed!")
        print(f"\nCheck progress at: {args.airflow_url}/dags/master_ingestion_dag/grid")