from spa2num.converter import to_number
from typing import Optional


def words_to_numbers(text:str) -> Optional[str, int]:
    """ Ported from the spa2num library. Uses spa2num.converter.to_number
    to perform the conversion.

    DOCSTRING: >>> Extracted from spa2num <<<
    Allows you to translate to integer numerical words spelled in spanish.

    Text must be previously cleaned & removed extraneous words or symbols. 
    Quantities MUST be written in a correct Spanish. The upper limit is up to the millions range. 
    No decimal supported.

    Parameters
    ----------
    text : str
        A properly written and cleaned number in spanish words.

    Returns
    -------
    str or int
        A string or integer containing the number.
    """
    return to_number(text)

