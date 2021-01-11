import pandas as pd
import numpy as np

from unidecode import unidecode

#########################
# Data loading
#########################

from google.cloud import bigquery

def load_data_from_bq(data_names,project_name):
    """
    Basic structure to load data from GBQ.
    params:
    - data_names : table names.
    - project_name :: project name.
    returns:
    - dfs : list of BQ tables loaded as dataframes.
    """
    client = bigquery.Client(location="US")

    queries = [f"""SELECT * FROM `{project_name}.{name}`""" for name in data_names]

    dfs=[]
    for i, _ in enumerate(data_names): dfs.append(client.query(queries[i],location="US").to_dataframe())
    return(dfs)

#########################
# Data manipulation
#########################

def create_lags(df, cols, n):
    """
    Create lagged version of a df.
    params:
    - df : a dataframe.
    - cols : a list of columns.
    - n : number of lags.
    returns:
    - lagged_df : a lagged dataframe without the missing data at the start.
    """
    assert df.shape[0] >= n+1, f'DataFrame is big enough for {n} lags.'
    dfs_to_concat = []

    for col in cols:
        lagged_cols = [f'{col}_lag_{i+1}' for i in range(n)]
        lagged_df_col = pd.DataFrame(columns=lagged_cols)
        
        for i, lagged_col in enumerate(lagged_cols):
            lagged_df_col[lagged_col]=df[col].shift(i+1)
        dfs_to_concat.append(lagged_df_col)
        
    lagged_df=pd.concat(dfs_to_concat,axis=1).iloc[n:]
    return(lagged_df)

def standardize_data(df):
  """
  Standardize strings by striping spaces, making them uppercase and removing accents marks.
  params:
  - df : a dataframe
  returns:
  - std_df : a standardized dataframe.
  """
  df_std=df.select_dtypes('object')
  for col in df_std.columns:
      df_std[col]=df_std[col].str.upper().str.rstrip().str.lstrip().astype('str').apply(unidecode)
  return(df_std)
  
def create_date_features(df,date_col):
  """
  Create some datetime features from a dataframe.
  params:
  - df : a dataframe.
  - date_col : name of the datetime column.
  returns:
  - date_df : a dataframe with datetime features created.
  """
  assert pd.api.types.is_datetime64_ns_dtype(df[date_col]) == True, "This column is not a datetime64 column."
  date_df=df[date_col].copy()
  date_df['Year']=date_df[date_col].dt.year
  date_df['Month']=date_df[date_col].dt.month
  date_df['Weekday']=date_df[date_col].dt.weekday
  date_df['Quarter']=date_df.dt.quarter
  date_df['Semester'] = np.where(date_df['Quarter'].isin([1,2]),1,2)
  return(date_df)
  
