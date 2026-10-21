{{ config(materialized='view') }}

/*
  Layer: stg (Cleaned Staging Layer)
  Model: stg_inventory_orders
  Description: Cleans raw order data, handles null values, deduplicates webhook retries,
               standardizes casing/whitespace, and calculates order-level metrics.
*/

WITH deduplicated AS (
    SELECT
        order_id,
        order_date,
        sku_id,
        product_name,
        raw_category,
        raw_supplier_name,
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
        customer_region,
        ROW_NUMBER() OVER (
            PARTITION BY order_id 
            ORDER BY order_date DESC
        ) AS dedup_rank
    FROM {{ ref('src_supply_chain_orders') }}
),

cleaned AS (
    SELECT
        order_id,
        CAST(order_date AS TIMESTAMP) AS order_timestamp,
        CAST(order_date AS DATE) AS order_date,
        sku_id,
        TRIM(product_name) AS product_name,
        
        -- Text cleaning & casing normalization
        TRIM(INITCAP(raw_category)) AS category,
        TRIM(raw_supplier_name) AS supplier_name,
        TRIM(warehouse_id) AS warehouse_id,
        TRIM(shipping_mode) AS shipping_mode,
        
        -- Units & Return handling
        units_ordered,
        CASE WHEN units_ordered < 0 THEN TRUE ELSE FALSE END AS is_return,
        CASE WHEN units_ordered > 0 THEN units_ordered ELSE 0 END AS net_units_sold,
        
        -- Pricing & Null handling for discount
        CAST(unit_price AS NUMERIC(10, 2)) AS unit_price,
        COALESCE(CAST(discount_percent AS NUMERIC(5, 2)), 0.00) AS discount_percent,
        
        -- Financial calculations
        ROUND(CAST(units_ordered * unit_price AS NUMERIC), 2) AS gross_sales_amount,
        ROUND(CAST(units_ordered * unit_price * (1.0 - COALESCE(discount_percent, 0.00)) AS NUMERIC), 2) AS net_revenue,
        
        -- Lead time logic & null handling
        promised_lead_time_days,
        actual_lead_time_days,
        CASE 
            WHEN actual_lead_time_days IS NOT NULL THEN (actual_lead_time_days - promised_lead_time_days)
            ELSE NULL 
        END AS lead_time_variance_days,
        CASE 
            WHEN actual_lead_time_days > promised_lead_time_days THEN TRUE 
            ELSE FALSE 
        END AS is_delivery_delayed,
        
        CAST(delivery_date AS TIMESTAMP) AS delivery_timestamp,
        CAST(delivery_date AS DATE) AS delivery_date,
        order_status,
        
        -- Inventory snapshots
        current_stock_level,
        reorder_batch_size,
        
        -- Geographical null handling
        COALESCE(TRIM(customer_region), 'Unassigned') AS customer_region

    FROM deduplicated
    WHERE dedup_rank = 1
)

SELECT * FROM cleaned
