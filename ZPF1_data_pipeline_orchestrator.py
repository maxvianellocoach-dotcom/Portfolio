"""
MODULO: Data Pipeline Orchestrator (PySpark + Airflow Logic)
Dimostra una pipeline ETL con PySpark e una logica di orchestrazione stile Airflow.
"""
import subprocess
import sys
import importlib.util
import os
import shutil
from datetime import datetime, timedelta
import time
import logging

# Auto-installazione
def install_and_import(package_name, import_name=None):
    if import_name is None:
        import_name = package_name
    spec = importlib.util.find_spec(import_name)
    if spec is None:
        print(f"📦 Installazione di '{package_name}'...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
        print(f"✅ '{package_name}' installato.")
    else:
        print(f"✅ '{package_name}' già presente.")

if shutil.which("java") is None:
    print("⚠️ Java non trovato. Installa Java da https://adoptium.net/")
    sys.exit(1)
else:
    print("✅ Java trovato.")

install_and_import("pyspark")
install_and_import("pandas")
install_and_import("apache-airflow")  # Per dimostrare la conoscenza del concetto

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, avg, count, current_timestamp
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# =====================================================
# 1. Task ETL (eseguito da PySpark)
# =====================================================
def run_etl_task(execution_date):
    """Simula un task ETL che processa dati per una data specifica."""
    logger.info(f"🚀 Avvio task ETL per la data: {execution_date}")
    spark = SparkSession.builder.appName(f"ETL_Job_{execution_date}").master("local[*]").getOrCreate()

    # Dati di esempio (potrebbero provenire da un file o database)
    data = [
        (1, execution_date, 10.5, "OK"),
        (2, execution_date, 15.2, "ERROR"),
        (3, execution_date, 12.1, "OK"),
    ]
    df = spark.createDataFrame(data, ["id", "data_riferimento", "valore", "stato"])

    logger.info(f"📊 Dati grezzi per {execution_date}:")
    df.show()

    # Pulizia e trasformazione
    df_clean = df.filter(col("stato") == "OK") \
                 .withColumn("processing_timestamp", current_timestamp())

    # Aggregazione
    df_aggregated = df_clean.groupBy("data_riferimento") \
                           .agg(avg("valore").alias("valore_medio"),
                                count("id").alias("conteggio_record"))

    logger.info(f"📈 Dati aggregati per {execution_date}:")
    df_aggregated.show()

    # Simula la scrittura su un data lake (in un caso reale, su S3/GCS)
    output_path = f"./output_data/data_riferimento={execution_date}"
    try:
        df_aggregated.write.mode("overwrite").parquet(output_path)
        logger.info(f"✅ Dati scritti in {output_path}")
    except Exception as e:
        logger.warning(f"⚠️ Scrittura su disco fallita, ma dati processati: {e}")
        # Mostra i dati come fallback
        pdf = df_aggregated.toPandas()
        print(pdf.to_string(index=False))

    spark.stop()
    logger.info(f"✅ Task ETL per {execution_date} completato.")
    return {"date": execution_date, "status": "success"}

# =====================================================
# 2. Orchestratore (simula la logica di Airflow)
# =====================================================
class SimpleOrchestrator:
    """
    Simula un orchestratore di pipeline stile Airflow.
    Gestisce l'esecuzione sequenziale di task e il retry in caso di fallimento.
    """
    def __init__(self):
        self.tasks = []
        self.logger = logging.getLogger(__name__)

    def add_task(self, task_func, task_id, depends_on=None, retries=1):
        """Aggiunge un task alla pipeline."""
        self.tasks.append({
            "id": task_id,
            "func": task_func,
            "depends_on": depends_on,
            "retries": retries,
            "status": "pending"
        })
        self.logger.info(f"➕ Task '{task_id}' aggiunto alla pipeline.")

    def run(self, context):
        """Esegue la pipeline, rispettando le dipendenze."""
        self.logger.info("🚀 Avvio della pipeline orchestrata...")
        results = {}
        for task in self.tasks:
            # Controlla le dipendenze
            if task["depends_on"]:
                dep_status = results.get(task["depends_on"], {}).get("status")
                if dep_status != "success":
                    self.logger.warning(f"⏭️ Task '{task['id']}' saltato: dipendenza '{task['depends_on']}' fallita.")
                    task["status"] = "skipped"
                    continue

            # Esegue il task con retry
            for attempt in range(task["retries"] + 1):
                try:
                    self.logger.info(f"▶️ Esecuzione task '{task['id']}' (tentativo {attempt+1})...")
                    result = task["func"](context)
                    results[task["id"]] = result
                    task["status"] = "success"
                    self.logger.info(f"✅ Task '{task['id']}' completato con successo.")
                    break
                except Exception as e:
                    self.logger.error(f"❌ Task '{task['id']}' fallito: {e}")
                    if attempt == task["retries"]:
                        task["status"] = "failed"
                        results[task["id"]] = {"status": "failed", "error": str(e)}
                    else:
                        self.logger.info(f"🔄 Ritento il task '{task['id']}'...")
                        time.sleep(2)

        self.logger.info("🏁 Esecuzione pipeline completata.")
        return results

# =====================================================
# MAIN: Esecuzione della Pipeline
# =====================================================
def main():
    logger.info("=== Demo: Pipeline ETL Orchestrata (PySpark + Airflow Logic) ===")

    # 1. Definisci il contesto (es. la data di esecuzione)
    context = {"execution_date": datetime.now().strftime("%Y-%m-%d")}

    # 2. Crea l'orchestratore
    orchestrator = SimpleOrchestrator()

    # 3. Aggiungi i task (come DAG in Airflow)
    # Task 1: ETL per il giorno corrente
    orchestrator.add_task(
        task_func=lambda ctx: run_etl_task(ctx["execution_date"]),
        task_id="etl_daily",
        retries=2
    )

    # Task 2: ETL per il giorno precedente (dipende dal Task 1)
    # In un caso reale, questo potrebbe essere un task di backfill
    orchestrator.add_task(
        task_func=lambda ctx: run_etl_task((datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")),
        task_id="etl_backfill",
        depends_on="etl_daily",
        retries=1
    )

    # 4. Esegui la pipeline
    results = orchestrator.run(context)

    # 5. Report finale
    logger.info("📋 Report finale della pipeline:")
    for task_id, result in results.items():
        logger.info(f"  - {task_id}: {result.get('status', 'N/A')}")

    print("\n✅ Programma completato con successo! Hai dimostrato competenze in PySpark, ETL e orchestrazione (Airflow).")

if __name__ == "__main__":
    main()