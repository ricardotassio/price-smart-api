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
) as dag:

    @task()
    def collect_data():
        http_hook = HttpHook(method='GET', http_conn_id='collect_data_dag')
        response = http_hook.run(endpoint='api/collect-data')

        # Checa e registra a resposta
        if response.status_code == 200:
            logging.info("Data collected successfully: %s", response.text)
            return response.text
        else:
            logging.error("Failed to collect data: %s", response.text)
            raise ValueError("API request failed with status code {}".format(response.status_code))

    collect_data()
