import os
import re
import json
import pandas as pd

# Crear estructura de carpetas del Lakehouse si no existen
os.makedirs("bronze", exist_ok=True)
os.makedirs("silver", exist_ok=True)
os.makedirs("gold", exist_ok=True)

# ==============================================================================
# CAPA SILVER: LIMPIEZA Y NORMALIZACIÓN
# ==============================================================================

# 1. Procesamiento de clientes (clientes.json -> clean_customers.csv)
with open("bronze/clientes.json", "r", encoding="utf-8") as f:
    raw_customers = json.load(f)

df_customers = pd.DataFrame(raw_customers)

# Normalizar tipo de dato de id_cliente a entero (corrige "120" -> 120)
df_customers["id_cliente"] = pd.to_numeric(df_customers["id_cliente"], errors="coerce")
df_customers = df_customers.dropna(subset=["id_cliente"])
df_customers["id_cliente"] = df_customers["id_cliente"].astype(int)

# Filtrar cuentas ficticias de sistema (TestUser y Debug)
df_customers = df_customers[
    ~df_customers["nombre"].astype(str).str.lower().isin(["testuser", "debug"])
].copy()

# Imputar valores faltantes o inválidos en nombre y país
df_customers["nombre"] = (
    df_customers["nombre"].astype(str).str.strip().replace("", pd.NA).fillna("None")
)
df_customers["pais"] = (
    df_customers["pais"].replace(["??", ""], pd.NA).fillna("N.A")
)

# Estandarizar nombres de columnas al esquema Silver
df_customers = df_customers.rename(
    columns={
        "id_cliente": "customer_id",
        "nombre": "customer_name",
        "pais": "country",
    }
)

# Guardar dataset limpio de clientes
df_customers.to_csv("silver/clean_customers.csv", index=False)

# 2. Limpieza y filtrado de logs (logs.txt -> clean_logs.csv)
valid_customer_ids = set(df_customers["customer_id"])
clean_logs = []

with open("bronze/logs.txt", "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        # Conservar solo eventos operativos válidos (INFO/WARN) con user numérico y acción no vacía
        match = re.match(r"^\[(INFO|WARN)\]\s+user=(\d+)\s+action=(\w+)$", line)
        if match:
            level, user_id, action = match.groups()
            user_id = int(user_id)
            if user_id in valid_customer_ids:
                clean_logs.append({"level": level, "customer_id": user_id, "action": action})

df_logs = pd.DataFrame(clean_logs)
df_logs.to_csv("silver/clean_logs.csv", index=False)

# 3. Procesamiento de ventas (ventas.csv -> clean_sales.csv)
df_sales = pd.read_csv("bronze/ventas.csv")

# Convertir id_cliente e importe a numérico para detectar nulos o texto ("abc")
df_sales["id_cliente"] = pd.to_numeric(df_sales["id_cliente"], errors="coerce")
df_sales["importe"] = pd.to_numeric(df_sales["importe"], errors="coerce")

# Aplicar reglas de calidad de datos:
# - Cliente válido existente en clean_customers
# - Importe positivo (> 0) y sin outliers extremos (< 5000, descarta el 9999)
# - Excluir productos de prueba ("test")
df_sales = df_sales[
    df_sales["id_cliente"].isin(valid_customer_ids)
    & (df_sales["importe"] > 0)
    & (df_sales["importe"] < 5000)
    & (df_sales["producto"].astype(str).str.lower() != "test")
].copy()

df_sales["id_cliente"] = df_sales["id_cliente"].astype(int)
df_sales["producto"] = df_sales["producto"].astype(str).str.strip().str.lower()

# Estandarizar nombres de columnas al esquema Silver
df_sales = df_sales.rename(
    columns={
        "id_pedido": "order_id",
        "id_cliente": "customer_id",
        "producto": "product",
        "importe": "amount",
    }
)

# Guardar dataset limpio de ventas
df_sales.to_csv("silver/clean_sales.csv", index=False)

# ==============================================================================
# CAPA GOLD: AGREGACIONES Y VALOR ANALÍTICO (OLAP)
# ==============================================================================

# Tabla 1: Ventas e ingresos totales por cliente
gold_sales_by_customer = (
    df_sales.groupby("customer_id", as_index=False)
    .agg(total_orders=("order_id", "count"), total_revenue=("amount", "sum"))
    .merge(df_customers[["customer_id", "customer_name", "country"]], on="customer_id", how="left")
    .sort_values(by="total_revenue", ascending=False)
)

# Reordenar columnas para presentación final
gold_sales_by_customer = gold_sales_by_customer[
    ["customer_id", "customer_name", "country", "total_orders", "total_revenue"]
]
gold_sales_by_customer.to_csv("gold/sales_by_customer.csv", index=False)

# Tabla 2: Ingresos totales por producto
gold_sales_by_product = (
    df_sales.groupby("product", as_index=False)
    .agg(units_sold=("order_id", "count"), total_revenue=("amount", "sum"))
    .sort_values(by="total_revenue", ascending=False)
)
gold_sales_by_product.to_csv("gold/sales_by_product.csv", index=False)

print("Proceso ETL completado con éxito.")
print(f"Clientes limpios: {len(df_customers)} | Ventas limpias: {len(df_sales)}")