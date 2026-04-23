-- models/intermediate/int_signalconso_cleaned.sql
-- clean_text, token_count et is_valid étant déjà produits en amont,
-- cette couche enrichit uniquement avec des métriques analytiques utiles.

WITH staged AS (
    SELECT * FROM {{ ref('stg_signalconso') }}
),

enriched AS (
    SELECT
        source_id,
        created_at,
        category,
        subcategories,
        tags,
        status,

        -- Indicateurs de traitement
        contactagreement,
        forwardtoreponseconso,
        signalement_transmis,
        signalement_lu,
        signalement_reponse,

        -- Géographie
        dep_name,
        dep_code,
        reg_code,
        reg_name,

        -- Texte nettoyé
        clean_text,
        token_count,
        is_valid,

        -- Métriques dérivées
        CASE
            WHEN signalement_reponse = 1 THEN 'repondu'
            WHEN signalement_lu      = 1 THEN 'lu'
            WHEN signalement_transmis = 1 THEN 'transmis'
            ELSE 'en_attente'
        END                                             AS traitement_status,

        CASE
            WHEN token_count >= 50 THEN 'long'
            WHEN token_count >= 20 THEN 'moyen'
            ELSE 'court'
        END                                             AS texte_longueur,

        EXTRACT(YEAR  FROM created_at)                  AS year,
        EXTRACT(MONTH FROM created_at)                  AS month,
        FORMAT_DATE('%Y-%m', created_at)                AS year_month

    FROM staged
)

SELECT * FROM enriched