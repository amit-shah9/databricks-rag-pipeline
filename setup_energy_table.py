from databricks.connect import DatabricksSession
import config

spark = DatabricksSession.builder.clusterId(config.CLUSTER_ID).getOrCreate()

# Small structured table of current figures NOT in the Wikipedia corpus —
# so the SQL tool genuinely extends what the agent can answer.
rows = [
    ("2026-06-01", "day_ahead_price_eur_mwh", 84.5),
    ("2026-06-01", "wind_generation_gw", 22.3),
    ("2026-06-01", "solar_generation_gw", 18.7),
    ("2026-06-01", "co2_intensity_g_kwh", 312.0),
    ("2026-06-02", "day_ahead_price_eur_mwh", 79.1),
    ("2026-06-02", "wind_generation_gw", 25.8),
    ("2026-06-02", "solar_generation_gw", 16.2),
    ("2026-06-02", "co2_intensity_g_kwh", 298.0),
]
df = spark.createDataFrame(rows, ["date", "metric", "value"])
table = f"{config.FQ_SCHEMA}.live_energy_metrics"
df.write.mode("overwrite").saveAsTable(table)
print(f"Wrote {df.count()} rows to {table}")
df.show()