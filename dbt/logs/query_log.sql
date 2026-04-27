-- created_at: 2026-04-27T10:01:43.799328900+00:00
-- finished_at: 2026-04-27T10:01:48.144520600+00:00
-- elapsed: 4.3s
-- outcome: success
-- dialect: bigquery
-- node_id: not available
-- query_id: kwa8E3UM34v5tKuYrroIASd85Ta
-- desc: execute adapter call
/* {"app": "dbt", "connection_name": "", "dbt_version": "2.0.0", "profile_name": "signalconso", "target_name": "dev"} */

    select distinct schema_name from `tri-demandes-clients`.INFORMATION_SCHEMA.SCHEMATA;
  ;
-- created_at: 2026-04-27T10:01:48.145246700+00:00
-- finished_at: 2026-04-27T10:01:49.875875600+00:00
-- elapsed: 1.7s
-- outcome: success
-- dialect: bigquery
-- node_id: not available
-- query_id: COkZ575WfHzvuwwksdNRelYiPcP
-- desc: execute adapter call
/* {"app": "dbt", "connection_name": "", "dbt_version": "2.0.0", "profile_name": "signalconso", "target_name": "dev"} */
create schema if not exists `signalconso_dev_intermediate`;
-- created_at: 2026-04-27T10:01:49.876368100+00:00
-- finished_at: 2026-04-27T10:01:51.871549400+00:00
-- elapsed: 2.0s
-- outcome: success
-- dialect: bigquery
-- node_id: not available
-- query_id: T8QBYwNwKA12CkIyG3p2cKJ06m3
-- desc: execute adapter call
/* {"app": "dbt", "connection_name": "", "dbt_version": "2.0.0", "profile_name": "signalconso", "target_name": "dev"} */
create schema if not exists `signalconso_dev_marts`;
-- created_at: 2026-04-27T10:01:51.872054600+00:00
-- finished_at: 2026-04-27T10:01:53.904986100+00:00
-- elapsed: 2.0s
-- outcome: success
-- dialect: bigquery
-- node_id: not available
-- query_id: 2CIyB3txlxs4K18syh7dPdmCPqQ
-- desc: execute adapter call
/* {"app": "dbt", "connection_name": "", "dbt_version": "2.0.0", "profile_name": "signalconso", "target_name": "dev"} */
create schema if not exists `signalconso_dev_staging`;
-- created_at: 2026-04-27T10:01:53.932184600+00:00
-- finished_at: 2026-04-27T10:01:54.864994300+00:00
-- elapsed: 932ms
-- outcome: error
-- error vendor code: -2147483648
-- error message: Unknown: [BigQuery] googleapi: Error 404: Not found: Dataset tri-demandes-clients:signalconso_dev_staging was not found in location EU, notFound
-- dialect: bigquery
-- node_id: model.signalconso.stg_signalconso
-- query_id: not available
-- desc: get_relation > list_relations call
SELECT
    table_catalog,
    table_schema,
    table_name,
    table_type
FROM
    `tri-demandes-clients`.`signalconso_dev_staging`.INFORMATION_SCHEMA.TABLES;
-- created_at: 2026-04-27T10:01:54.865672800+00:00
-- finished_at: 2026-04-27T10:01:55.496695500+00:00
-- elapsed: 631ms
-- outcome: error
-- error vendor code: -2147483648
-- error message: Unknown: [BigQuery] googleapi: Error 404: Not found: Dataset tri-demandes-clients:signalconso_dev_staging was not found in location EU, notFound
-- dialect: bigquery
-- node_id: model.signalconso.stg_signalconso
-- query_id: not available
-- desc: get_relation adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.signalconso.stg_signalconso", "profile_name": "signalconso", "target_name": "dev"} */
SELECT table_catalog,
                    table_schema,
                    table_name,
                    table_type
                FROM `tri-demandes-clients`.`signalconso_dev_staging`.INFORMATION_SCHEMA.TABLES
                WHERE table_name = 'stg_signalconso';
-- created_at: 2026-04-27T10:01:55.504677500+00:00
-- finished_at: 2026-04-27T10:01:55.952124900+00:00
-- elapsed: 447ms
-- outcome: error
-- error vendor code: -2147483648
-- error message: Unknown: [BigQuery] googleapi: Error 404: Not found: Dataset tri-demandes-clients:signalconso_dev_staging was not found in location EU, notFound (Query: https://console.cloud.google.com/bigquery?project=tri-demandes-clients&j=bq:EU:h6l2piMpMagIlQL533FS1zOcnyJ&page=queryresults)
-- dialect: bigquery
-- node_id: model.signalconso.stg_signalconso
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.signalconso.stg_signalconso", "profile_name": "signalconso", "target_name": "dev"} */


  create or replace view `tri-demandes-clients`.`signalconso_dev_staging`.`stg_signalconso`
  OPTIONS()
  as -- models/staging/stg_signalconso.sql
-- Couche staging : renommage canonique + typage + déduplication par id.
-- Source : tri-demandes-clients.Complaints.Signal_Conso

WITH source AS (
    SELECT * FROM `tri-demandes-clients`.`Complaints`.`Signal_Conso`
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
WHERE is_valid = TRUE;

;
