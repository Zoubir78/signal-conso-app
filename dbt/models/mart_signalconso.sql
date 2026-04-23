-- models/marts/mart_signalconso.sql
-- Couche mart : table finale partitionnée et clusterisée pour requêtes analytiques
-- et consommation par les modèles ML.
-- Matérialisée en TABLE dans BigQuery (voir dbt_project.yml).

{{
  config(
    materialized='table',
    partition_by={
      'field': 'created_date',
      'data_type': 'date',
      'granularity': 'month'
    },
    cluster_by=['category', 'dep_code'],
    labels={'pipeline': 'signalconso', 'layer': 'mart'}
  )
}}

WITH cleaned AS (
    SELECT * FROM {{ ref('int_signalconso_cleaned') }}
),

final AS (
    SELECT
        -- Clés
        source_id,

        -- Temporalité
        created_at,
        DATE(created_at)                        AS created_date,
        EXTRACT(YEAR FROM created_at)           AS year,
        EXTRACT(MONTH FROM created_at)          AS month,

        -- Classification (colonnes cibles pour ML)
        category,
        subcategories_raw                       AS subcategories,
        tags_raw                                AS tags,
        status,

        -- Géographie
        dep_name,
        dep_code,
        reg_name,
        reg_code,

        -- Texte nettoyé (feature principale)
        clean_text,
        token_count,

        -- Qualité
        is_valid,

        -- Provenance
        _ingested_at,
        CURRENT_TIMESTAMP()                     AS _transformed_at

    FROM cleaned
)

SELECT * FROM final
