-- models/staging/stg_signalconso.sql
-- Couche staging : renommage canonique + typage + déduplication par id.
-- Source : tri-demandes-clients.Complaints.Signal_Conso

WITH source AS (
    SELECT * FROM {{ source('Complaints', 'Signal_Conso') }}
),

deduped AS (
    SELECT *
    FROM source
    QUALIFY ROW_NUMBER() OVER (
        PARTITION BY id
        ORDER BY creationdate DESC
    ) = 1
),

renamed AS (
    SELECT
        -- Identifiant
        id                          AS source_id,

        -- Temporalité
        creationdate                AS created_at,

        -- Classification
        category,
        subcategories,
        tags,
        status,

        -- Signalement
        contactagreement,
        forwardtoreponseconso,
        signalement_transmis,
        signalement_lu,
        signalement_reponse,

        -- Géographie
        dep_name,
        dep_code,
        CAST(reg_code AS STRING)    AS reg_code,
        reg_name,

        -- Texte déjà nettoyé en amont
        clean_text,
        token_count,
        is_valid

    FROM deduped
)

SELECT * FROM renamed
WHERE is_valid = TRUE