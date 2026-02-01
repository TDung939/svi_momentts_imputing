# Utlities for loading inwater data from various pilot sites
# And consolidated tables
import pandas as pd
import os
import awswrangler as wr
import boto3
import logging

boto3.setup_default_session(region_name="us-west-2")

s3_results = os.getenv("INSITU_RESULTS_BUCKET")
athena_wg = os.getenv("INSITU_ATHENA_WG")
insitu_db = os.getenv("INSITU_DB")
table_prefix = insitu_db.replace("-db","")

eagle_streams_table = f"{table_prefix}-eagleio-stream-info"
eagle_locations_table = f"{table_prefix}-eagleio-location-info"

GBR_ESTUARY_TABLE = f'"{insitu_db}"."{table_prefix}-eagleio-data_all_gbr-staging"'
GBR_BAY_TABLE = f'"{insitu_db}"."{table_prefix}-eagleio-wqual_gbr-staging"'

CBS_EXO1_TABLE = f'"{insitu_db}"."{table_prefix}-eagleio-exo1_data-staging"'
CBS_EXO2_TABLE = f'"{insitu_db}"."{table_prefix}-eagleio-exo2_data-staging"'

SPG_EXO_TABLE = f'"{insitu_db}"."{table_prefix}-eagleio-wq-staging"'

ACCEPTED_PARAMS = ["chlorophyll_ugl","salinity_psu","turbidity_ntu","temperature_degc","phycocyanin_rfu", "phycocyanin_ugl", "phycoerythrin_rfu", "phycoerythrin_ugl", "nitrate_mgl", "fdom_rfu", "fdom_qsu", "do_mgl", "do_sat"]

def get_all_gbr_estuary(start_time,end_time):
    logging.info(f"Loading all data from {GBR_ESTUARY_TABLE} between {start_time} and {end_time}")
    query = f'SELECT * FROM {GBR_ESTUARY_TABLE}\
            WHERE ts BETWEEN \'{start_time}\' and \'{end_time}\'\
            ORDER BY ts'
    df = wr.athena.read_sql_query(sql=query,
                              database=insitu_db,
                              athena_cache_settings={
                                     "max_cache_seconds": 3600,
                                },
                              workgroup=athena_wg,
                              unload_approach=False,
                              ctas_approach=False
                             )
    df = df.dropna(axis=1, how='all')
    return df

def get_all_gbr_bay(start_time, end_time):
    logging.info(f"Loading all data from {GBR_BAY_TABLE} between {start_time} and {end_time}")
    query = f'SELECT * FROM {GBR_BAY_TABLE}\
            WHERE ts BETWEEN \'{start_time}\' and \'{end_time}\'\
            ORDER BY ts'
    df = wr.athena.read_sql_query(sql=query,
                              database=insitu_db,
                              athena_cache_settings={
                                     "max_cache_seconds": 3600,
                                },
                              workgroup=athena_wg,
                              unload_approach=False,
                              ctas_approach=False
                             )
    df = df.dropna(axis=1, how='all')
    return df

def get_cbs_locations(exo_identity : int = 1):
    if exo_identity == 1: 
        exo_table = CBS_EXO1_TABLE
    else:
        exo_table = CBS_EXO2_TABLE
        
    query = f'SELECT count(1) as count,location FROM {exo_table} group by location;'
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

def get_all_cbs(start_time : str, end_time : str, exo_identity : int = 1):
    if exo_identity == 1: 
        exo_table = CBS_EXO1_TABLE
    else:
        exo_table = CBS_EXO2_TABLE
    query = f'SELECT * FROM {exo_table} \
            WHERE ts BETWEEN \'{start_time}\' AND \'{end_time}\'\
            ORDER BY ts DESC;'
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

def get_all_spg(start_time : str, end_time : str):
    query = f'SELECT * FROM {SPG_EXO_TABLE}\
            WHERE ts BETWEEN \'{start_time}\' AND \'{end_time}\'\
            ORDER BY ts DESC;'
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

def get_all_hydrovu(location,param,start_time,end_time):
    query = f'SELECT ts,\"{param}\" FROM "{insitu_db}"."{table_prefix}-hydrovu-staging"\
            WHERE \'{param}\' is NOT NULL\
            AND location=array{location}\
            AND ts BETWEEN \'{start_time}\' AND \'{end_time}\'\
            ORDER by ts DESC;'
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

def get_consolidated_param(start_date, end_date, param, qa_state = False):
    """
    Return x8 high priority parameters from any relevant tables/API's
    from Local datalake , BOM, NWIS or any source in EU, ASIA etc.
    """
    # Route Australia queries to relevant consolidated table
    # Route Malaysia and anthing else using HydroVu those tables
    # Route US (detected using AOI) to NWIS
    # Route EU to ??
    if (param not in ACCEPTED_PARAMS):
        raise ValueError(f"Invalid param : {param} it has to be one of {ACCEPTED_PARAMS}")
    query = f'SELECT ST_Point(location[2],location[1]) as geometry,{param},ts\
        from "{insitu_db}"."{table_prefix}-eagleio-{param}-staging"\
        WHERE location <> ARRAY[0.0,0.0]\
        AND ts > \'{start_date}\' AND ts < \'{end_date}\'\
        ORDER BY ts DESC;'
    df =  wr.athena.read_sql_query(sql=query,
                              database=insitu_db,
                              athena_cache_settings={
                                     "max_cache_seconds": 3600,
                                },
                              workgroup=athena_wg,
                              unload_approach=False,
                              ctas_approach=False
                             )
    return df
    

def load_gbr_lab():
    """
    Load data from a CSV file and format the DataFrame.
    This is stored in projects bucket and can be updated as more lab results become available.
      
    Args:
        Lab_file_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Formatted DataFrame.
    """
    lab_uri = "s3://adias-prod-dc-data-projects/ai4m-cf/gbr_lab_data/combined_lab_data.csv"
    try:
        lab_data = pd.read_csv(lab_uri)  # Read data from CSV file
        # Perform any additional formatting if needed
        return lab_data
    except Exception as e:
        print("An error occurred while loading the data:", e)
        return None  # or handle the error in an appropriate way

def get_inwater_stream_info(stream_id : str):
    table = boto3.resource("dynamodb").Table(eagle_streams_table)
    return table.get_item(Key={"stream_id": stream_id})['Item']


def get_controlled_vocab(stream_name : str):
    """
    Look up some of the arbitrary abbreviations stored in Eagle.io and return a
    name according to the controlled vocabulary for AquaWatch
    """
    query = f'SELECT control_meas, control_unit, meas FROM "{eagle_streams_table}" WHERE contains("meas", ?)'
    df = wr.dynamodb.read_partiql_query(
    query=query,
        parameters=[stream_name],
    )
    return df.dropna()
    