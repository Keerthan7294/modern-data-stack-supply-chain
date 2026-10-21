{{ config(materialized='table') }}

/*
  Layer: marts (Business Marts Layer)
  Model: fct_inventory_metrics
  Description: Aggregated SKU-level inventory performance, demand volatility,
               lead time variance, statistical safety stock, and reorder thresholds.
*/

WITH order_base AS (
    SELECT * FROM {{ ref('stg_inventory_orders') }}
),

-- Aggregate daily sales to compute demand volatility (mean & standard deviation)
daily_sku_demand AS (
    SELECT
        sku_id,
        order_date,
        SUM(net_units_sold) AS daily_units_sold
    FROM order_base
    GROUP BY sku_id, order_date
),

demand_stats AS (
    SELECT
        sku_id,
        ROUND(AVG(daily_units_sold)::numeric, 2) AS avg_daily_demand,
        ROUND(COALESCE(STDDEV_SAMP(daily_units_sold), 1.0)::numeric, 2) AS stddev_daily_demand,
        MAX(daily_units_sold) AS max_daily_demand,
        MIN(daily_units_sold) AS min_daily_demand
    FROM daily_sku_demand
    GROUP BY sku_id
),

-- Lead time statistics (mean, variance/stddev, delayed shipment rate)
lead_time_stats AS (
    SELECT
        sku_id,
        ROUND(AVG(actual_lead_time_days)::numeric, 2) AS avg_lead_time_days,
        ROUND(COALESCE(STDDEV_SAMP(actual_lead_time_days), 1.0)::numeric, 2) AS stddev_lead_time_days,
        MAX(actual_lead_time_days) AS max_lead_time_days,
        ROUND(
            100.0 * COUNT(CASE WHEN is_delivery_delayed THEN 1 END)::numeric / 
            NULLIF(COUNT(CASE WHEN actual_lead_time_days IS NOT NULL THEN 1 END), 0), 
            2
        ) AS delayed_shipment_rate_pct
    FROM order_base
    WHERE actual_lead_time_days IS NOT NULL
    GROUP BY sku_id
),

-- SKU metadata & snapshot state
sku_metadata AS (
    SELECT
        sku_id,
        MAX(product_name) AS product_name,
        MAX(category) AS category,
        MAX(supplier_name) AS supplier_name,
        MAX(warehouse_id) AS warehouse_id,
        MAX(shipping_mode) AS shipping_mode,
        ROUND(AVG(unit_price)::numeric, 2) AS unit_price,
        MAX(current_stock_level) AS current_stock_level,
        MAX(reorder_batch_size) AS reorder_batch_size,
        COUNT(DISTINCT order_id) AS total_orders_placed,
        SUM(net_units_sold) AS total_units_sold,
        ROUND(SUM(net_revenue)::numeric, 2) AS total_net_revenue,
        ROUND(
            100.0 * COUNT(CASE WHEN is_return THEN 1 END)::numeric / 
            NULLIF(COUNT(*), 0), 
            2
        ) AS return_rate_pct
    FROM order_base
    GROUP BY sku_id
),

combined_metrics AS (
    SELECT
        m.sku_id,
        m.product_name,
        m.category,
        m.supplier_name,
        m.warehouse_id,
        m.shipping_mode,
        m.unit_price,
        m.current_stock_level,
        m.reorder_batch_size,
        m.total_orders_placed,
        m.total_units_sold,
        m.total_net_revenue,
        m.return_rate_pct,
        
        -- Demand parameters
        COALESCE(d.avg_daily_demand, 1.00) AS avg_daily_demand,
        COALESCE(d.stddev_daily_demand, 0.50) AS stddev_daily_demand,
        d.max_daily_demand,
        
        -- Lead time parameters
        COALESCE(l.avg_lead_time_days, 7.00) AS avg_lead_time_days,
        COALESCE(l.stddev_lead_time_days, 1.50) AS stddev_lead_time_days,
        l.max_lead_time_days,
        COALESCE(l.delayed_shipment_rate_pct, 0.00) AS delayed_shipment_rate_pct,
        
        -- Statistical Safety Stock (Z = 1.65 for 95% service level)
        -- Formula: SS = Z * sqrt(L * sigma_D^2 + D^2 * sigma_L^2)
        ROUND(
            1.65 * SQRT(
                (COALESCE(l.avg_lead_time_days, 7.00) * POWER(COALESCE(d.stddev_daily_demand, 0.50), 2)) +
                (POWER(COALESCE(d.avg_daily_demand, 1.00), 2) * POWER(COALESCE(l.stddev_lead_time_days, 1.50), 2))
            )
        ) AS safety_stock_units,
        
        -- Reorder Point (ROP = (D * L) + SS)
        ROUND(
            (COALESCE(d.avg_daily_demand, 1.00) * COALESCE(l.avg_lead_time_days, 7.00)) +
            (
                1.65 * SQRT(
                    (COALESCE(l.avg_lead_time_days, 7.00) * POWER(COALESCE(d.stddev_daily_demand, 0.50), 2)) +
                    (POWER(COALESCE(d.avg_daily_demand, 1.00), 2) * POWER(COALESCE(l.stddev_lead_time_days, 1.50), 2))
                )
            )
        ) AS reorder_point_units,
        
        -- Days of Supply
        ROUND(
            m.current_stock_level::numeric / NULLIF(d.avg_daily_demand, 0),
            1
        ) AS days_of_supply

    FROM sku_metadata m
    LEFT JOIN demand_stats d ON m.sku_id = d.sku_id
    LEFT JOIN lead_time_stats l ON m.sku_id = l.sku_id
),

final_risk_classification AS (
    SELECT
        *,
        -- Stockout Risk Classification
        CASE
            WHEN current_stock_level <= safety_stock_units THEN 'CRITICAL'
            WHEN current_stock_level <= reorder_point_units THEN 'HIGH RISK'
            WHEN current_stock_level <= (reorder_point_units * 2.2) THEN 'HEALTHY'
            ELSE 'OVERSTOCKED'
        END AS stockout_risk_status,
        
        CASE
            WHEN current_stock_level <= reorder_point_units THEN TRUE
            ELSE FALSE
        END AS is_reorder_recommended,
        
        CURRENT_TIMESTAMP AS dbt_updated_at
    FROM combined_metrics
)

SELECT * FROM final_risk_classification
