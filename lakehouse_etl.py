import json
import re
from pathlib import Path
import pandas as pd

# Definición de rutas multiplataforma (respeta "Bronze" con mayúscula)
BASE_DIR = Path(__file__).resolve().parent
BRONZE_DIR = BASE_DIR / "Bronze"
SILVER_DIR = BASE_DIR / "silver"
GOLD_DIR = BASE_DIR / "gold"


def setup_directories() -> None:
    """Crea las carpetas de destino si no existen."""
    SILVER_DIR.mkdir(exist_ok=True)
    GOLD_DIR.mkdir(exist_ok=True)


def build_silver_layer() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Limpia, normaliza y valida los datos crudos de la capa Bronze."""
    # --- 1. CLIENTES ---
    with open(BRONZE_DIR / "clientes.json", "r", encoding="utf-8") as f:
        df_customers = pd.DataFrame(json.load(f))

    raw_cust_count = len(df_customers)

    # Normalizar ID a entero y eliminar nulos o duplicados
    df_customers["id_cliente"] = pd.to_numeric(df_customers["id_cliente"], errors="coerce")
    df_customers = df_customers.dropna(subset=["id_cliente"]).drop_duplicates(subset=["id_cliente"])
    df_customers["id_cliente"] = df_customers["id_cliente"].astype(int)

    # Excluir cuentas de prueba/sistema (TestUser, Debug)
    df_customers = df_customers[
        ~df_customers["nombre"].fillna("").astype(str).str.lower().isin(["testuser", "debug"])
    ].copy()

    # Normalizar nombres vacíos como user_<id>
    df_customers["nombre"] = df_customers["nombre"].fillna("").astype(str).str.strip()
    df_customers["nombre"] = df_customers.apply(
        lambda r: f"user_{r['id_cliente']}" if r["nombre"] in ["", "None", "nan"] else r["nombre"],
        axis=1,
    )

    # Normalizar países faltantes o inválidos ("??", vacío, None) como "null"
    df_customers["pais"] = (
        df_customers["pais"].replace(["??", ""], pd.NA).fillna("null").astype(str).str.upper()
    )

    df_customers = df_customers.rename(
        columns={"id_cliente": "customer_id", "nombre": "customer_name", "pais": "country"}
    )
    df_customers.to_csv(SILVER_DIR / "clean_customers.csv", index=False)

    valid_ids = set(df_customers["customer_id"])

    # --- 2. LOGS ---
    clean_logs = []
    raw_log_count = 0
    log_pattern = re.compile(r"^\[(INFO|WARN)\]\s+user=(\d+)\s+action=(\w+)$")

    with open(BRONZE_DIR / "logs.txt", "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            raw_log_count += 1
            match = log_pattern.match(line)
            if match:
                level, uid_str, action = match.groups()
                uid = int(uid_str)
                if uid in valid_ids:
                    clean_logs.append({"level": level, "customer_id": uid, "action": action.lower()})

    df_logs = pd.DataFrame(clean_logs)
    df_logs.to_csv(SILVER_DIR / "clean_logs.csv", index=False)

    # --- 3. VENTAS ---
    df_sales = pd.read_csv(BRONZE_DIR / "ventas.csv")
    raw_sales_count = len(df_sales)

    df_sales["id_cliente"] = pd.to_numeric(df_sales["id_cliente"], errors="coerce")
    df_sales["importe"] = pd.to_numeric(df_sales["importe"], errors="coerce")
    df_sales["producto"] = df_sales["producto"].fillna("").astype(str).str.strip().str.lower()

    # Filtrar integridad referencial, rangos válidos y excluir productos de test
    df_sales = df_sales[
        df_sales["id_cliente"].isin(valid_ids)
        & (df_sales["importe"] > 0)
        & (df_sales["importe"] < 5000)
        & (df_sales["producto"] != "test")
    ].drop_duplicates(subset=["id_pedido"]).copy()

    df_sales["id_cliente"] = df_sales["id_cliente"].astype(int)
    df_sales = df_sales.rename(
        columns={
            "id_pedido": "order_id",
            "id_cliente": "customer_id",
            "producto": "product",
            "importe": "amount",
        }
    )
    df_sales.to_csv(SILVER_DIR / "clean_sales.csv", index=False)

    # Reporte de auditoría en consola
    print("=== AUDITORÍA DE CALIDAD (BRONZE -> SILVER) ===")
    print(f"Clientes : {len(df_customers)}/{raw_cust_count} válidos ({raw_cust_count - len(df_customers)} descartados)")
    print(f"Ventas   : {len(df_sales)}/{raw_sales_count} válidas ({raw_sales_count - len(df_sales)} descartadas)")
    print(f"Logs     : {len(df_logs)}/{raw_log_count} válidos ({raw_log_count - len(df_logs)} descartados)\n")

    return df_customers, df_logs, df_sales


def build_gold_layer(
    df_customers: pd.DataFrame, df_logs: pd.DataFrame, df_sales: pd.DataFrame
) -> None:
    """Genera las tablas analíticas agregadas (OLAP) en la capa Gold."""
    # Contar eventos válidos por cliente desde clean_logs
    log_activity = (
        df_logs.groupby("customer_id", as_index=False)
        .agg(valid_log_events=("action", "count"))
    )

    # 1. Ventas e ingresos totales por usuario (formato user_<id>, país "null" si falta)
    gold_customers = (
        df_sales.groupby("customer_id", as_index=False)
        .agg(total_orders=("order_id", "count"), total_revenue=("amount", "sum"))
        .merge(df_customers[["customer_id", "country"]], on="customer_id", how="left")
        .merge(log_activity, on="customer_id", how="left")
        .sort_values(by="total_revenue", ascending=False)
    )

    gold_customers["valid_log_events"] = gold_customers["valid_log_events"].fillna(0).astype(int)
    gold_customers["user"] = "user_" + gold_customers["customer_id"].astype(str)
    gold_customers = gold_customers[
        ["user", "country", "total_orders", "total_revenue", "valid_log_events"]
    ]
    gold_customers.to_csv(GOLD_DIR / "sales_by_customer.csv", index=False)

    # 2. Ingresos totales por producto
    gold_products = (
        df_sales.groupby("product", as_index=False)
        .agg(units_sold=("order_id", "count"), total_revenue=("amount", "sum"))
        .sort_values(by="total_revenue", ascending=False)
    )
    gold_products.to_csv(GOLD_DIR / "sales_by_product.csv", index=False)

    print("=== CAPA GOLD GENERADA CON ÉXITO ===")
    print(f"Top usuario  : {gold_customers.iloc[0]['user']} ({gold_customers.iloc[0]['total_revenue']} €)")
    print(f"Top producto : {gold_products.iloc[0]['product']} ({gold_products.iloc[0]['total_revenue']} €)")


if __name__ == "__main__":
    setup_directories()
    customers_df, logs_df, sales_df = build_silver_layer()
    build_gold_layer(customers_df, logs_df, sales_df)