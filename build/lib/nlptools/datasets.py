"""
    This module contains the function to load any dataset.
"""

import os, pickle
import pkgutil
import pandas as pd
from io import StringIO


def _bytes_to_pandas(bytes_data):
    
    to_string = str(bytes_data, 'utf-8')
    data = StringIO(to_string)
    df = pd.read_csv(data)
    del to_string, data
    return df


def _get_source(data_name:str) -> pd.DataFrame:
    """ Loads a data source for internal usage.

    Parameters
    ----------
    data_name : str, {'calles', 'companies', 'persons'}
        Name of the dataset intended to load.

    Returns
    -------
    pd.DataFrame
        A pandas.DataFrame with the source data required.
    """
    if data_name in {'calles', 'companies', 'persons'}:
        source =  pkgutil.get_data('nlptools', f'data/{data_name}.csv')
        result = _bytes_to_pandas(source)
        del source
    else:
        result = None
        print(f'No dataset found with name {data_name}.')

    return result


def load_dataset(data_name:str):
    """ Function to call datasets for model training.

    Parameters
    ----------
    data_name : str, {'estatutos'}
        Name of the dataset intended to load.

    Returns
    -------
    [type]
        Every dataset has a different output. Check the type.
    """
    if data_name == 'estatutos':
        with open('data/estatutos/tagged/spacy_dataset_2020-5-6.pkl', 'rb') as file:
            result = pickle.load(file)

    return result