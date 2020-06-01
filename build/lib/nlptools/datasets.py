"""
    This module contains the function to load any dataset.
"""

import os, pickle
import pandas as pd
import pkgutil

def _get_source(data_name:str) -> pd.DataFrame:
    """ Loads a data source for internal usage.

    Parameters
    ----------
    data_name : str, {'streets', 'companies', 'people'}
        Name of the dataset intended to load.

    Returns
    -------
    pd.DataFrame
        A pandas.DataFrame with the dataset required.
    """
    if data_name == 'streets':
        result =  pd.read_csv('data/calles.csv', dtype=str)
    elif data_name == 'companies':
        result = pd.read_csv('data/companies.csv', dtype=str)
    elif data_name == 'people':
        result = pd.read_csv('data/persons.csv', dtype=str)    
    
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


test_data = pkgutil.get_data('sources', 'calles.csv')