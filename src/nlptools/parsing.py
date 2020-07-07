from spa2num.converter import to_number
from typing import Union
from num2words import num2words


def words_to_numbers(text:str) -> Union[str, int]:
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

    Examples
    -------
    >>> from nlptools.parsing import words_to_numbers

    >>> words_to_numbers('ciento dieciocho mil quinientos cuarenta y uno')
    118541
    >>> words_to_numbers('dos mil quinientos millones')
    2500000000
    """
    return to_number(text)


def number_to_words(
        number,
        lang: str = 'es',
        to: str = 'cardinal',
        **kwargs
    ) -> str:
    """ Ported from the num2words library. Uses num2words.num2words
        under the hood.
        
        Converts a number into a phrase containing the words that
        spell the number provided. Supports different languages.
        Can return an cardinal or ordinal representation of 
        the number.

    Parameters
    ----------
    number : {str, int, float}
        A number with or without decimals.
    lang : str, optional
        Output language, by default 'es'.
    to : str, optional
        Type of grammatical expression. Can be either cardinal
        or ordinal, by default 'cardinal'.
    **kargs :
        Uknown. Please refer to the num2words library to find out.

    Returns
    -------
    str
        A phrase with the number spelled into the desired language.

    Examples
    -------
    >>> from nlptools.parsing import number_to_words
    
    >>> number_to_words(1541, lang='es')
    'mil quinientos cuarenta y uno'
    
    >>> number_to_words(1541, lang='en')
    'one thousand, five hundred and forty-one'
    
    >>> number_to_words(154.1, lang='es')
    'ciento cincuenta y cuatro punto uno'
    
    >>> number_to_words(154.1, lang='en')
    'one hundred and fifty-four point one'
    
    >>> number_to_words("2345", lang='es')
    'dos mil trescientos cuarenta y cinco'
    
    >>> number_to_words(9, lang='es', to='cardinal')
    'nueve'
    
    >>> number_to_words(9, lang='es', to='ordinal')
    'noveno'
    """
    return num2words(number=number, lang=lang, to=to, **kwargs)
    