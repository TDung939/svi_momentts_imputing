import os
import awswrangler as wr
import boto3
import pandas as pd

# This is needed if the default profile is not setup
boto3.setup_default_session(region_name="us-west-2")

s3_results = os.getenv("INSITU_RESULTS_BUCKET")
athena_wg = os.getenv("INSITU_ATHENA_WG")
insitu_db = os.getenv("INSITU_DB")
table_prefix = insitu_db.replace("-db","")

scalars_table = "senaps-partitioned-allscalars-staging"
vectors_table = "senaps-partitioned-allvectors-staging"
images_table = "senaps-partitioned-allimages-staging"

senaps_platforms_table = f"{table_prefix}-senaps-platform-info"
senaps_streams_table = f"{table_prefix}-senaps-stream-info"

# Stripe color mapping for dark themed plots - background stripe is white
hs_stripe_cmap_dark = {"lnadir": "#ff9900",
                       "lu+30": "#009900", "lu30": "#009900",
                       "lu0": "#00ff00",
                       "lu-30": "#ccffcc",
                       "lsky+30": "#b3d1ff", "lsky30": "#b3d1ff",
                       "lsky0": "#0066ff",
                       "lsky-30": "#00ffff",
                       "sky_diffuse": "#ff00ff",  
                       "background": "#ffffff",  # white
                       "blank1": "#5a5a5a", "blank2": "#737373",
                       "blank3": "#8d8d8d", "blank4": "#a6a6a6"
                      }

# Stripe color mapping for light themed plots - background stripe is black
hs_stripe_cmap_light = {"lnadir": "#ff9900",
                        "lu+30": "#009900", "lu30": "#009900",
                        "lu0": "#00ff00",
                        "lu-30": "#ccffcc",
                        "lsky+30": "#b3d1ff", "lsky30": "#b3d1ff",
                        "lsky0": "#0066ff",
                        "lsky-30": "#00ffff",
                        "sky_diffuse": "#ff00ff",  
                        "background": "#000000",  # black
                        "blank1": "#5a5a5a", "blank2": "#737373",
                        "blank3": "#8d8d8d", "blank4": "#a6a6a6"
                       }

def get_all_scalars(hs_id, start_time, end_time) -> pd.DataFrame:
    query = (f'SELECT * FROM "{insitu_db}"."{table_prefix}-{scalars_table}" WHERE \
        platform_id=\'{hs_id}\'\
        AND t BETWEEN \'{start_time}\' AND \'{end_time}\'\
        ORDER BY t DESC;')
    df = wr.athena.read_sql_query(sql=query,
                              database=insitu_db,
                              athena_cache_settings={
                                     "max_cache_seconds": 3600,
                                },
                              workgroup=athena_wg,
                              unload_approach=False,
                              ctas_approach=False
                             )
    return df

def get_all_images(hs_id, start_time, end_time) -> pd.DataFrame:
    query = (f'SELECT * FROM "{insitu_db}"."{table_prefix}-{images_table}" \
                        WHERE platform_id=\'{hs_id}\'\
                        AND t BETWEEN \'{start_time}\' AND \'{end_time}\'\
                        ORDER BY t DESC;')
    df = wr.athena.read_sql_query(sql=query,
                                  database=insitu_db,
                                  athena_cache_settings={
                                         "max_cache_seconds": 3600,
                                    },
                                  workgroup=athena_wg,
                                  unload_approach=False,
                                  ctas_approach=False
                                 )
    return df

def get_all_vectors(hs_id, spectra_names, start_time, end_time) -> pd.DataFrame:
    spectra_selection = ",".join(spectra_names)
    query = f'SELECT location,t,{spectra_selection}  FROM "{insitu_db}"."{table_prefix}-{vectors_table}" \
                        WHERE platform_id=\'{hs_id}\' \
                        AND t BETWEEN \'{start_time}\' AND \'{end_time}\'\
                        ORDER BY t DESC;'
    df = wr.athena.read_sql_query(sql=query,
                                  database=insitu_db,
                                  athena_cache_settings={
                                         "max_cache_seconds": 3600,
                                    },
                                  workgroup=athena_wg,
                                  unload_approach=False,
                                  ctas_approach=False
                                 )
    df = df.groupby(df['t']).aggregate('first')
    return df

def get_senaps_key() -> str:
    secret_id = os.getenv("SENAPS_SECRET_ID")
    secret_client = boto3.client("secretsmanager",region_name="us-west-2")
    senaps_key = secret_client.get_secret_value(SecretId=secret_id)['SecretString']
    return senaps_key

def get_platform_info(hs_id : str) -> dict:
    table = boto3.resource("dynamodb").Table(senaps_platforms_table)
    return table.get_item(Key={"platform_id": hs_id})['Item']

def get_hs_stream_info(stream_id : str) -> dict:
    table = boto3.resource("dynamodb").Table(senaps_streams_table)
    return table.get_item(Key={"stream_id": stream_id})['Item']

def get_all_platforms() -> pd.DataFrame:
    df = wr.dynamodb.read_partiql_query(
        query=f"SELECT platform_id FROM \"{senaps_platforms_table}\"",
    )
    return df

def get_all_streams() -> pd.DataFrame:
    df = wr.dynamodb.read_partiql_query(
        query=f"SELECT stream_id FROM \"{senaps_streams_table}\"",
    )
    return df
