-- models/staging/stg_signalconso.sql
-- Couche staging : typage strict, renommage canonique, déduplication par source_id.
-- La logique de texte complexe est déléguée à la couche intermediate.

WITH source AS (
    SELECT * FROM {{ source('raw', 'signalconso') }}
),

deduped AS (
    -- Garde la ligne la plus récente pour chaque identifiant source
    SELECT *
    FROM source
    QUALIFY ROW_NUMBER() OVER (
        PARTITION BY COALESCE(source_id, CAST(FARM_FINGERPRINT(TO_JSON_STRING(source)) AS STRING))
        ORDER BY _ingested_at DESC
    ) = 1
),

renamed AS (
    SELECT
        -- Identifiant
        COALESCE(source_id, id, recordid, uuid)                             AS source_id,

        -- Dates
        SAFE_CAST(
            COALESCE(creationdate, created_at, date, publication_date)
            AS TIMESTAMP
        )                                                                    AS created_at,

        -- Classification
        COALESCE(category, categorie, theme)                                 AS category,
        COALESCE(subcategories, sub_category, subcategory, sous_categorie)   AS subcategories_raw,
        COALESCE(tags, tag)                                                  AS tags_raw,
        COALESCE(status, statut, state)                                      AS status,

        -- Géographie
        COALESCE(dep_name, department_name, departement_name, department)    AS dep_name,
        COALESCE(dep_code, department_code)                                  AS dep_code,
        COALESCE(reg_name, region_name)                                      AS reg_name,
        COALESCE(reg_code, region_code)                                      AS reg_code,

        -- Texte libre
        COALESCE(description, narrative, message, content, body, details)    AS complaint_text,

        -- Métadonnées d'ingestion
        _ingested_at

    FROM deduped
)

SELECT * FROM renamed
