{{ config(materialized='view') }}

/*
  Layer: src (Raw Data Layer)
  Model: src_supply_chain_orders
  Description: Abstraction over raw ingestion table with initial column normalization.
*/

WITH raw_orders AS (
    SELECT
        order_id,
        order_date,
        sku_id,
        product_name,
        category AS raw_category,
        supplier_name AS raw_supplier_name,
        warehouse_id,
        units_ordered,
        unit_price,
        discount_percent,
        shipping_mode,
        promised_lead_time_days,
        actual_lead_time_days,
        delivery_date,
        order_status,
        current_stock_level,
        reorder_batch_size,
        customer_region
    FROM {{ source('supply_chain_raw', 'supply_chain_orders') }}
)

SELECT * FROM raw_orders
