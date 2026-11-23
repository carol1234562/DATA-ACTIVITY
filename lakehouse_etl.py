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
    """Crea las carpetas de destino (silver y gold) si no existen."""
    SILVER_DIR.mkdir(exist_ok=True)
    GOLD_DIR.mkdir(exist_ok=True)


def build_silver_layer() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Limpia, normaliza y valida los datos crudos de la capa Bronze."""
    # --------------------------------------------------------------------------
    # 1. PROCESAMIENTO DE CLIENTES (clientes.json -> silver/clean_customers.csv)
    # --------------------------------------------------------------------------
    with open(BRONZE_DIR / "clientes.json", "r", encoding="utf-8") as f:
        df_customers = pd.DataFrame(json.load(f))

    raw_cust_count = len(df_customers)

    # Normalizar ID de cliente a entero (corrige "120" -> 120) y eliminar duplicados
    df_customers["id_cliente"] = pd.to_numeric(df_customers["id_cliente"], errors="coerce")
    df_customers = df_customers.dropna(subset=["id_cliente"]).drop_duplicates(subset=["id_cliente"])
    df_customers["id_cliente"] = df_customers["id_cliente"].astype(int)

    # Excluir cuentas de prueba/sistema (TestUser y Debug)
    df_customers = df_customers[
        ~df_customers["nombre"].fillna("").astype(str).str.lower().isin(["testuser", "debug"])
    ].copy()

    # Normalizar nombres vacíos o nulos como "user_<id_cliente>"
    df_customers["nombre"] = df_customers["nombre"].fillna("").astype(str).str.strip()
    df_customers["nombre"] = df_customers.apply(
        lambda r: f"user_{r['id_cliente']}" if r["nombre"] in ["", "None", "nan"] else r["nombre"],
        axis=1,
    )

    # Normalizar países faltantes o inválidos ("??", vacío o None) literalmente como "null"
    df_customers["pais"] = df_customers["pais"].replace(["??", ""], pd.NA).fillna("null")
    df_customers["pais"] = df_customers["pais"].apply(
        lambda x: x if x == "null" else str(x).strip().upper()
    )

    # Estandarizar nombres de columnas al esquema Silver
    df_customers = df_customers.rename(
        columns={
            "id_cliente": "customer_id",
            "nombre": "customer_name",
            "pais": "country",
        }
    )
    df_customers.to_csv(SILVER_DIR / "clean_customers.csv", index=False)

    valid_ids = set(df_customers["customer_id"])

    # --------------------------------------------------------------------------
    # 2. PROCESAMIENTO DE LOGS (logs.txt -> silver/clean_logs.csv)
    # --------------------------------------------------------------------------
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
                # Conservar solo eventos de clientes válidos
                if uid in valid_ids:
                    clean_logs.append(
                        {"level": level, "customer_id": uid, "action": action.lower()}
                    )

    df_logs = pd.DataFrame(clean_logs)
    df_logs.to_csv(SILVER_DIR / "clean_logs.csv", index=False)

    # --------------------------------------------------------------------------
    # 3. PROCESAMIENTO DE VENTAS (ventas.csv -> silver/clean_sales.csv)
    # --------------------------------------------------------------------------
    df_sales = pd.read_csv(BRONZE_DIR / "ventas.csv")
    raw_sales_count = len(df_sales)

    # Convertir IDs e importes a numérico para detectar valores corruptos ("abc") o nulos
    df_sales["id_cliente"] = pd.to_numeric(df_sales["id_cliente"], errors="coerce")
    df_sales["importe"] = pd.to_numeric(df_sales["importe"], errors="coerce")
    df_sales["producto"] = df_sales["producto"].fillna("").astype(str).str.strip().str.lower()

    # Filtrar integridad referencial, rangos válidos (> 0 y < 5000) y excluir producto "test"
    df_sales = df_sales[
        df_sales["id_cliente"].isin(valid_ids)
        & (df_sales["importe"] > 0)
        & (df_sales["importe"] < 5000)
        & (df_sales["producto"] != "test")
    ].drop_duplicates(subset=["id_pedido"]).copy()

    df_sales["id_cliente"] = df_sales["id_cliente"].astype(int)

    # Estandarizar nombres de columnas al esquema Silver
    df_sales = df_sales.rename(
        columns={
            "id_pedido": "order_id",
            "id_cliente": "customer_id",
            "producto": "product",
            "importe": "amount",
        }
    )
    df_sales.to_csv(SILVER_DIR / "clean_sales.csv", index=False)

    # Reporte de auditoría de limpieza en consola
    print("=== AUDITORÍA DE CALIDAD (BRONZE -> SILVER) ===")
    print(f"Clientes : {len(df_customers)}/{raw_cust_count} válidos ({raw_cust_count - len(df_customers)} descartados)")
    print(f"Ventas   : {len(df_sales)}/{raw_sales_count} válidas ({raw_sales_count - len(df_sales)} descartadas)")
    print(f"Logs     : {len(df_logs)}/{raw_log_count} válidos ({raw_log_count - len(df_logs)} descartados)\n")

    return df_customers, df_logs, df_sales


def build_gold_layer(
    df_customers: pd.DataFrame, df_logs: pd.DataFrame, df_sales: pd.DataFrame
) -> None:
    """Genera las tablas analíticas agregadas (OLAP) en la capa Gold."""
    # Contar eventos válidos registrados en logs por cada cliente
    log_activity = (
        df_logs.groupby("customer_id", as_index=False)
        .agg(valid_log_events=("action", "count"))
    )

    # 1. Tabla de ventas e ingresos por usuario (formato legible user_<id>, sin nombres personales)
    gold_customers = (
        df_sales.groupby("customer_id", as_index=False)
        .agg(total_orders=("order_id", "count"), total_revenue=("amount", "sum"))
        .merge(df_customers[["customer_id", "country"]], on="customer_id", how="left")
        .merge(log_activity, on="customer_id", how="left")
        .sort_values(by=["total_revenue", "total_orders"], ascending=False)
    )

    gold_customers["valid_log_events"] = gold_customers["valid_log_events"].fillna(0).astype(int)
    gold_customers["user"] = "user_" + gold_customers["customer_id"].astype(str)

    # Seleccionar columnas finales para Gold
    gold_customers = gold_customers[
        ["user", "country", "total_orders", "total_revenue", "valid_log_events"]
    ]
    gold_customers.to_csv(GOLD_DIR / "sales_by_customer.csv", index=False)

    # 2. Filtrar el usuario con mayor número de compras (total_orders)
    max_orders = gold_customers["total_orders"].max()
    top_buyer_by_orders = gold_customers[gold_customers["total_orders"] == max_orders]
    top_buyer_by_orders.to_csv(GOLD_DIR / "top_buyer_by_orders.csv", index=False)

    # 3. Filtrar el usuario que genera mayor ingreso total (total_revenue)
    max_revenue = gold_customers["total_revenue"].max()
    top_buyer_by_revenue = gold_customers[gold_customers["total_revenue"] == max_revenue]

    # 4. Tabla de ingresos totales por producto
    gold_products = (
        df_sales.groupby("product", as_index=False)
        .agg(units_sold=("order_id", "count"), total_revenue=("amount", "sum"))
        .sort_values(by="total_revenue", ascending=False)
    )
    gold_products.to_csv(GOLD_DIR / "sales_by_product.csv", index=False)

    # 5. Filtrar producto más rentable y menos rentable
    most_profitable_product = gold_products.iloc[0]
    least_profitable_product = gold_products.iloc[-1]

    # Resumen ejecutivo en consola
    print("=== CAPA GOLD GENERADA CON ÉXITO ===")
    print(
        f"Usuario con más compras  : {top_buyer_by_orders.iloc[0]['user']} "
        f"({top_buyer_by_orders.iloc[0]['total_orders']} pedidos | {top_buyer_by_orders.iloc[0]['total_revenue']} €)"
    )
    print(
        f"Usuario con más ingreso  : {top_buyer_by_revenue.iloc[0]['user']} "
        f"({top_buyer_by_revenue.iloc[0]['total_revenue']} € | {top_buyer_by_revenue.iloc[0]['total_orders']} pedido)"
    )
    print(
        f"Producto más rentable    : {most_profitable_product['product']} "
        f"({most_profitable_product['total_revenue']} € | {most_profitable_product['units_sold']} uds)"
    )
    print(
        f"Producto menos rentable  : {least_profitable_product['product']} "
        f"({least_profitable_product['total_revenue']} € | {least_profitable_product['units_sold']} uds)"
    )


if __name__ == "__main__":
    setup_directories()
    customers_df, logs_df, sales_df = build_silver_layer()
    build_gold_layer(customers_df, logs_df, sales_df)