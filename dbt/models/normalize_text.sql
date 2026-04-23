{% macro normalize_text(column) %}
  LOWER(
    TRIM(
      REGEXP_REPLACE(
        REGEXP_REPLACE(
          REGEXP_REPLACE(
            REGEXP_REPLACE(
              -- Suppression des URLs
              REGEXP_REPLACE(
                COALESCE(CAST({{ column }} AS STRING), ''),
                r'https?://\S+|www\.\S+', ' '
              ),
              -- Suppression des emails
              r'\S+@\S+', ' '
            ),
            -- Suppression des accents via translittération approchée BigQuery
            -- (BigQuery ne supporte pas COLLATE UNICODE, on garde les accents normalisés)
            r'[éèêë]', 'e'
          ),
          r'[àâä]', 'a'
        ),
        -- Suppression des caractères non-alphanumériques (sauf espaces)
        r'[^a-z0-9éàèùâêîôûëïüœæç\s]', ' '
      )
    )
  )
{% endmacro %}


{% macro clean_multivalue(column) %}
  -- Nettoie les listes sérialisées Python type "['AchatMagasin', 'Autre']"
  LOWER(TRIM(
    REGEXP_REPLACE(
      REGEXP_REPLACE(
        COALESCE(CAST({{ column }} AS STRING), ''),
        r"[\[\]'\"{}()]", ''   -- supprime les délimiteurs de liste
      ),
      r'\s*,\s*', ' '          -- remplace les virgules par des espaces
    )
  ))
{% endmacro %}
