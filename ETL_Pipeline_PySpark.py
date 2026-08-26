"""
MODULO 3: ETL Pipeline con PySpark - VERSIONE IN MEMORIA (WINDOWS COMPATIBILE)
Non scrive su disco, mostra i dati trasformati in console.
"""

import subprocess
import sys
import importlib.util
import os
import shutil

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

# Verifica Java
if shutil.which("java") is None:
    print("⚠️  ATTENZIONE: Java non trovato. PySpark richiede Java per funzionare.")
    print("   Installa Java (JDK 8 o superiore) da: https://adoptium.net/")
    print("   Poi riavvia il programma.")
    sys.exit(1)
else:
    print("✅ Java trovato.")

install_and_import("pyspark")
install_and_import("pandas")
install_and_import("pyyaml", "yaml")

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, current_timestamp, avg, count
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    logger.info("=== Demo ETL Pipeline (Modalità Windows-compatibile) ===")
    
    # Crea Spark Session in modalità locale
    spark = SparkSession.builder \
        .appName("ETL_Demo") \
        .master("local[*]") \
        .config("spark.sql.adaptive.enabled", "true") \
        .getOrCreate()
    
    # 1. Crea dati in memoria (EVITA la lettura da file che causa winutils)
    data = [
        (1, "2024-01-01", 10.5, "OK"),
        (2, "2024-01-01", 15.2, "ERROR"),
        (3, "2024-01-02", 12.1, "OK"),
        (4, "2024-01-02", 8.7, "OK"),
        (5, "2024-01-03", 22.3, "ERROR")
    ]
    df = spark.createDataFrame(data, ["id", "timestamp", "value", "status"])
    logger.info("📊 Dati originali:")
    df.show()
    
    # 2. Pulizia: rimuovi duplicati, filtra status OK
    df_clean = df.dropDuplicates() \
                 .filter(col("status") == "OK") \
                 .withColumn("processing_date", current_timestamp())
    logger.info("🧹 Dati dopo pulizia e filtraggio (status = OK):")
    df_clean.show()
    
    # 3. Aggregazione: calcola media per data
    df_aggregated = df_clean.groupBy("timestamp") \
                           .agg(
                               avg("value").alias("avg_value"),
                               count("id").alias("count_records")
                           )
    logger.info("📈 Dati aggregati per timestamp:")
    df_aggregated.show()
    
    # 4. Validazione: controlla valori nulli
    total_rows = df_clean.count()
    for col_name in df_clean.columns:
        null_count = df_clean.filter(col(col_name).isNull()).count()
        null_pct = (null_count / total_rows) * 100 if total_rows > 0 else 0
        if null_pct > 0:
            logger.warning(f"Colonna {col_name} ha {null_pct:.2f}% nulli")
        else:
            logger.info(f"✅ Colonna {col_name}: 0% nulli")
    
    logger.info(f"✅ Validazione completata. Rows: {total_rows}")
    
    # 5. Converti in Pandas per mostrare i dati in formato tabellare
    logger.info("📋 Dati finali (formato tabellare):")
    df_aggregated_pd = df_aggregated.toPandas()
    print(df_aggregated_pd.to_string(index=False))
    
    # Ferma Spark
    spark.stop()
    logger.info("Spark fermato")
    
    print("\n✅ Programma completato con successo! (Nessun file scritto su disco)")

if __name__ == "__main__":
    main()