
"Authors: G.Kerrisk,... "
"QA_QC functions version 1: this is a demo that needs imrovment and contributions from software savy folks"

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go

#######################################################################

def tidy_data(data):
    """
    Perform data tidying operations including type conversion.

    Parameters:
        data (pd.DataFrame): Input DataFrame.
       
    Returns:
        pd.DataFrame: Tidied DataFrame.
    """
    # Convert 'Timestamp' column to datetime format
    data['Timestamp'] = pd.to_datetime(data['Timestamp'])
    
    # Convert all other columns to numeric format
    float_columns = [col for col in data.columns if col != 'Timestamp']
    data[float_columns] = data[float_columns].apply(pd.to_numeric, errors='coerce')
    
    return data

   
################################################################
def filter_out_maintenance(data, maintenance_logs):
    """
    Filter out rows falling within specified maintenance periods.

    Parameters:
        data (pd.DataFrame): Input DataFrame.
        maintenance_logs (list): List of tuples representing maintenance periods.

    Returns:
        pd.DataFrame: DataFrame with rows filtered based on maintenance periods.
    """
    maintenance_periods = [(pd.to_datetime(start), pd.to_datetime(end)) for start, end in maintenance_logs]
    
    def in_maintenance(timestamp, maintenance_periods):
        timestamp = pd.to_datetime(timestamp)
        for start, end in maintenance_periods:
            if start <= timestamp <= end:
                return True
        return False 
    
    maintenance_mask = data['Timestamp'].apply(lambda x: in_maintenance(x, maintenance_periods))
    data_cleaned = data[~maintenance_mask]
    
    return data_cleaned

#############################################################################
def Sensor_limitations_QA(data):
    """
    Perform sensor limitation data cleaning. Replace values in specific columns with NaN based on defined conditions.
    Note, this converts parameters from the same sensor to NAN also because if OPUS TSSeq is >999 then consecutive esitmates of 
    nitrate will not be valid from that sensor. Likewise, if the exo salinity or turbidity is less than 0.011 the exo is 
    likely in air or >4000, 80 respectivly it has extreme fouling or comms error. 

    Parameters:
        data (pd.DataFrame): Input DataFrame containing sensor data.

    Returns:
        pd.DataFrame: DataFrame with values replaced by NaN based on specified conditions.
    """
    data_cleaned = data.copy()
    
    # Condition 1: Check for 'OPUS1016 TSSeq_mg/l' column
    if 'OPUS1016 TSSeq_mg/l' in data_cleaned.columns:
        opus_condition = (data_cleaned['OPUS1016 TSSeq_mg/l'] > 999)
        opus_columns = data_cleaned.columns[data_cleaned.columns.str.contains('OPUS')]
        data_cleaned.loc[opus_condition, opus_columns] = np.nan
    
    # Condition 2: Check for 'EXO TurbFNU_FNU' column
    if 'EXO TurbFNU_FNU' in data_cleaned.columns:
        exo_turb_condition = ((data_cleaned['EXO TurbFNU_FNU'] < 0.011) | (data_cleaned['EXO TurbFNU_FNU'] > 4000))
        exo_columns = data_cleaned.columns[data_cleaned.columns.str.contains('EXO ')]
        data_cleaned.loc[exo_turb_condition, exo_columns] = np.nan
    
    # Condition 3: Check for 'EXO Salpsu_psu' column
    if 'EXO Salpsu_psu' in data_cleaned.columns:
        exo_salpsu_condition = ((data_cleaned['EXO Salpsu_psu'] < 0.011) | (data_cleaned['EXO Salpsu_psu'] > 80))
        data_cleaned.loc[exo_salpsu_condition, 'EXO Salpsu_psu'] = np.nan
    
    # Condition 4: Check for 'ARG SigNoise Avg_dB' column
    if 'ARG SigNoise Avg_dB' in data_cleaned.columns:
        arg_condition = np.abs(data_cleaned['ARG SigNoise Avg_dB'].diff()) > 15
        arg_columns = data_cleaned.columns[data_cleaned.columns.str.contains('ARG ')]
        data_cleaned.loc[arg_condition, arg_columns] = np.nan
        
    ## add condition 5 for CDOM sensor limits - todo
    
    ## add condition 6 for DO sensor limits - toDo
    
    return data_cleaned
####################################################################
def remove_zero_values_and_flatlines(data):
    """
    Remove zero values and flatlines from float columns of the DataFrame. '0' values and repeated measurments of the same value
    are not reflective of an actual measurment results in a tidal system from the IN WATER sensors. Flatlines indicate fouling or error. 
    
    Parameters:
        data (pd.DataFrame): Input DataFrame.
        
    Returns:
        pd.DataFrame: DataFrame with zero values and flatlines removed.
    """
    data_cleaned = data.copy()
    
    float_columns = [col for col in data_cleaned.columns if col != 'Timestamp']
    data_cleaned[float_columns] = data_cleaned[float_columns].apply(pd.to_numeric, errors='coerce')
    
    data_cleaned[float_columns] = data_cleaned[float_columns].replace(0.0, np.nan)
    
    for column in float_columns:
        mask = data_cleaned[column].diff() == 0
        data_cleaned.loc[mask, column] = np.nan
    
    return data_cleaned

#################################################################
def remove_spikes(data, columns_to_process=None, window_size=40, threshold=3):
    """
    basic remove spikes and outliers from specified columns of the DataFrame 
    by replacing outlier values with NaN using a rolling average.
    
    Parameters:
        data (pd.DataFrame): Input DataFrame.
        columns_to_process (list): List of column names to process. 
                                   If None, all numeric columns are processed.
        window_size (int): Size of the moving average window.
        threshold (float): Threshold for identifying outliers.
        
    Returns:
        pd.DataFrame: DataFrame with outliers replaced by NaN (a copy of the input DataFrame).
    """
    
    data_cleaned = data.copy()

    if columns_to_process is None:
        columns_to_process = data_cleaned.select_dtypes(include=np.number).columns.tolist()

    for column in columns_to_process:
        if column in data_cleaned.columns:
            moving_avg = data_cleaned[column].rolling(window=window_size, min_periods=2).mean()
            residuals = data_cleaned[column] - moving_avg
            outliers_mask = abs(residuals) > threshold * residuals.std()
            data_cleaned.loc[outliers_mask, column] = np.nan

    return data_cleaned

#This ultamitly needs to be specified for each parameter differently but is acceptable for a start at smoothing

#####################################################
def inter_sensor_QA(data, columns_dict, threshold=0.4):
    """
    Perform inter-sensor (different sensors, same sit) quality assurance checks on specified sensor measurements in the DataFrame.

    Parameters:
        data (pd.DataFrame): Input DataFrame containing sensor measurements.
        columns_dict (dict): Dictionary specifying the columns to compare as {'type': [col1, col2]}.
                             'type' can be 'tss_ntu' or 'temperature'.
        threshold (float): Threshold ratio for considering a sensor's value as potentially incorrect (default: 0.4).

    Returns:
        pd.DataFrame: DataFrame with potentially incorrect sensor values replaced by NaN.
    """
    data_cleaned = data.copy()
    
 
    def identify_and_replace_bad_sensor_values(data, col1, col2):
        """
        Identify and replace potentially incorrect sensor values between two columns by comparing differences.

        Parameters:
            data (pd.DataFrame): Input DataFrame containing sensor measurements.
            col1 (str): Name of the first column.
            col2 (str): Name of the second column.

        Returns:
            pd.DataFrame: DataFrame with potentially incorrect sensor values replaced by NaN.
        """
        # Calculate absolute differences between the two columns
        diff = abs(data[col1] - data[col2])
        
        # Calculate ratios of differences to the corresponding values
        ratio_col1 = diff / data[col1]
        ratio_col2 = diff / data[col2]
        
        # Determine which sensor's value is potentially incorrect based on the ratio
        mask_col1_bad = (ratio_col1 > threshold)
        mask_col2_bad = (ratio_col2 > threshold)
        
        # Replace values with NaN where a sensor's reading is potentially incorrect
        data.loc[mask_col1_bad, col1] = np.nan
        data.loc[mask_col2_bad, col2] = np.nan
        
        return data

    # Iterate through the specified column pairs based on 'type' in columns_dict
    for sensor_type, columns in columns_dict.items():
        if sensor_type == 'tss_ntu' or sensor_type == 'temperature':
            # Check sensor measurements and identify potentially incorrect values
            col1, col2 = columns
            data_cleaned = identify_and_replace_bad_sensor_values(data_cleaned, col1, col2)
            
    # Apply additional NaN replacements based on specific conditions
    # Replace 'EXO TSSmgL_mg/L' and 'EXO TurbFNU_FNU' with NaN where 'EXO TurbNTU_NTU' is NaN
    data_cleaned.loc[data_cleaned['EXO TurbNTU_NTU'].isnull(), ['EXO TSSmgL_mg/L', 'EXO TurbFNU_FNU']] = np.nan

    # Columns to set NaN if 'OPUS1016 TSSeq_mg/l' is NaN
    opus_columns_to_set_nan = [
        'OPUS1000 N NO3_mg/l', 'OPUS1004 CODeq_mg/l', 'OPUS1008 DOCeq_mg/l',
        'OPUS1012 Salinity_psu', 'OPUS1014 TOCeq_mg/l', 'OPUS1016 TSSeq_mg/l',
        'OPUS1032 SAC254_1/m', 'OPUS1034 Abs360_nan', 'OPUS1036 Abs210_nan',
        'OPUS1038 FitError_nan', 'OPUS1042 Abs254_nan', 'OPUS1046 NO3_mg/l',
        'OPUS1052 COD SACeq_mg/l', 'OPUS1060 SQI_nan', 'OpusECmScm_mS/cm'
    ]
    data_cleaned.loc[data_cleaned['OPUS1016 TSSeq_mg/l'].isnull(), opus_columns_to_set_nan] = np.nan        
            
    return data_cleaned
#todo: improve/develop further

##############################################################

def basic_nitrate_scaling(data, lab_data):
    """
    Scale nitrate measurements based on linear regression between two columns.

    Parameters:
        data (pd.DataFrame): Input DataFrame containing sensor data.
        Lab_data (pd.DataFrame): DataFrame containing lab measurements.

    Returns:
        pd.DataFrame: DataFrame with scaled nitrate measurements.
    """
    data_cleaned = data.copy()

    # Convert Timestamp columns to datetime objects
    data_cleaned['Timestamp'] = pd.to_datetime(data_cleaned['Timestamp'])
    # Add formatting for QLD timezone and output column as UTC
    lab_data['date_time_qld'] = pd.to_datetime(lab_data['date_time_qld']+"+10",utc=True)
    
    # Merge data_cleaned with Lab_data based on matching timestamps
    merged_data = pd.merge(data_cleaned, lab_data, left_on='Timestamp', right_on='date_time_qld', how='left')
    
    # Extract relevant columns into a subset for calculation of a scaling factor
    selected_data = merged_data[['NO3mg/L', 'OPUS1046 NO3_mg/l', 'Timestamp']]
    
    # Drop rows with NaN values in either column of the subset
    selected_data_cleaned = selected_data.dropna(subset=['NO3mg/L', 'OPUS1046 NO3_mg/l'])
    
    # Convert 'OPUS1046 NO3_mg/l' to numeric, handling errors
    selected_data_cleaned['OPUS1046 NO3_mg/l'] = pd.to_numeric(selected_data_cleaned['OPUS1046 NO3_mg/l'], errors='coerce')
    
    # Take only the first three rows of the cleaned subset for demonstration
    selected_data_cleaned = selected_data_cleaned.head(3)
    
    # Extract X (OPUS1046 NO3_mg/l) and y (NO3mg/L) for linear regression
    X = selected_data_cleaned['OPUS1046 NO3_mg/l'].values.reshape(-1, 1)
    y = selected_data_cleaned['NO3mg/L'].values
    
    # Create and fit the linear regression model
    model = LinearRegression()
    model.fit(X, y)

    # Get the scaling factor (slope) from the linear regression model
    scaling_factor = model.coef_[0]

    # Scale the 'OPUS1046 NO3_mg/l' data and calculate 'OPUS1000 N NO3_mg/l'
    data_cleaned.loc[:, 'OPUS1046 NO3_mg/l'] *= scaling_factor
    # TODO: Link to literature here for the magic number
    data_cleaned['OPUS1000 N NO3_mg/l'] = data_cleaned['OPUS1046 NO3_mg/l'] / 4.43
    
    # Print the scaling factor for examination
    print("Scaling Factor (Slope):", scaling_factor)
    
    # Return the scaled data
    return data_cleaned, scaling_factor


#To do: improve scaling factor determination to be dynamic for each quater of the year and update as lab samples come in..?
#Add a chlorophyll scaling factor if ug/l is indended to be used by aquawatch. For now use RFU only!

###########################################################

def create_scatter_plot(data, data_cleaned, y_column):
    """
    Create a scatter plot comparing the specified column between the original and cleaned data.

    Parameters:
        data (pd.DataFrame): Original DataFrame.
        data_cleaned (pd.DataFrame): Cleaned DataFrame.
        y_column (str): Column name for the y-axis.

    Returns:
        None
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Timestamp'], y=data[y_column],
                             mode='markers', name='Original', marker=dict(color='red', size=8, opacity=0.5)))
    fig.add_trace(go.Scatter(x=data_cleaned['Timestamp'], y=data_cleaned[y_column],
                             mode='markers', name='Cleaned', marker=dict(color='blue', size=8, opacity=0.5)))
    fig.update_layout(
        xaxis_title='Time',
        yaxis_title=y_column,
        
    )
    fig.show()



###########################################################
def create_scatter_plot2(data_cleaned, Lab_data, contains):
    """
    Create a scatter plot comparing columns containing specified substrings between
    the cleaned data and Lab data.

    Parameters:
        data_cleaned (pd.DataFrame): Cleaned DataFrame.
        Lab_data (pd.DataFrame): Lab DataFrame.
        contains (list of str): List of substrings to match in column names.

    Returns:
        None
    """
    fig = go.Figure()

    # Plot cleaned data
    for substr in contains:
        data_cleaned_columns = [col for col in data_cleaned.columns if substr.lower() in col.lower()]
        for column in data_cleaned_columns:
            fig.add_trace(go.Scatter(x=data_cleaned['Timestamp'], y=data_cleaned[column],
                                     mode='markers', name=column + ' (Cleaned)',
                                     marker=dict(size=8, opacity=0.4)))

    # Plot Lab data
    for substr in contains:
        Lab_data_columns = [col for col in Lab_data.columns if substr.lower() in col.lower()]
        for column in Lab_data_columns:
            fig.add_trace(go.Scatter(x=Lab_data['date_time_qld'], y=Lab_data[column],
                                     mode='markers', name=column + ' (Lab)',
                                     marker=dict(size=8, opacity=1)))

    fig.update_layout(
        xaxis_title='Time',
        yaxis_title= 'specified variable' ,  # Modify y-axis title based on your preference
        title=f'Scatter Plot of Columns containing specified substrings ({", ".join(contains)}) between Cleaned and Lab Data'
    )
    fig.show()