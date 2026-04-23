-- models/intermediate/int_signalconso_cleaned.sql
-- Couche intermediate : construit le champ clean_text et calcule token_count.
-- Équivalent SQL de build_clean_text() et transform_dataframe() en Python.
-- NOTE : include_category_in_text = False par défaut (évite la fuite de cible).

WITH staged AS (
    SELECT * FROM {{ ref('stg_signalconso') }}
),

text_parts AS (
    SELECT
        source_id,
        created_at,
        category,
        subcategories_raw,
        tags_raw,
        status,
        dep_name,
        dep_code,
        reg_name,
        reg_code,
        complaint_text,
        _ingested_at,

        -- Nettoyage individuel de chaque champ texte
        {{ normalize_text('subcategories_raw') }}   AS subcategories_clean,
        {{ clean_multivalue('subcategories_raw') }}  AS subcategories_flat,

        {{ normalize_text('tags_raw') }}             AS tags_clean,
        {{ clean_multivalue('tags_raw') }}           AS tags_flat,

        {{ normalize_text('dep_name') }}             AS dep_name_clean,
        {{ normalize_text('reg_name') }}             AS reg_name_clean,
        {{ normalize_text('status') }}               AS status_clean,
        {{ normalize_text('complaint_text') }}       AS complaint_text_clean

    FROM staged
),

assembled AS (
    SELECT
        *,

        -- Concaténation dans le même ordre que build_clean_text() Python :
        -- subcategories → tags → dep_name → reg_name → status → complaint_text
        -- (category exclu pour éviter la fuite de cible)
        TRIM(
            REGEXP_REPLACE(
                CONCAT_WS(' ',
                    NULLIF(subcategories_flat, ''),
                    NULLIF(tags_flat, ''),
                    NULLIF(dep_name_clean, ''),
                    NULLIF(reg_name_clean, ''),
                    NULLIF(status_clean, ''),
                    NULLIF(complaint_text_clean, '')
                ),
                r'\s+', ' '   -- normalise les espaces multiples
            )
        ) AS clean_text

    FROM text_parts
),

with_metrics AS (
    SELECT
        *,

        -- Équivalent de token_count (split sur espaces)
        ARRAY_LENGTH(SPLIT(TRIM(clean_text), ' ')) AS token_count,

        -- Équivalent de is_valid (min_text_length = 10)
        LENGTH(clean_text) >= 10 AS is_valid

    FROM assembled
)

SELECT *
FROM with_metrics
WHERE is_valid   -- filtre identique à transform_dataframe()
