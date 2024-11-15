from airflow import DAG
from airflow.decorators import task
from airflow.providers.http.hooks.http import HttpHook
from airflow.utils.dates import days_ago
from datetime import timedelta
import logging


default_args = {
    'owner': 'airflow',
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}


with DAG(
    'collect_data_dag',
    default_args=default_args,
    description='DAG to collect data from local API using TaskFlow API',
    schedule_interval='@daily',
    start_date=days_ago(1),
    catchup=False,
    tags=['example', 'http'],
) as dag:

    # Data Collection Tasks.
    @task()
    def collect_data():
        
        http_hook = HttpHook(method='GET', http_conn_id='collect_data_dag')
        
        try:
            response = http_hook.run(endpoint='/api/collect-data')
            response.raise_for_status()  # Raises an error for bad responses
            logging.info(f"Data collected successfully: {response.text}")
            return response.text
        except Exception as e:
            logging.error(f"Failed to collect data: {e}")
            raise ValueError(f"API request failed: {e}")

    collect_data()
