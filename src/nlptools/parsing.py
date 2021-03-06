import re
from copy import deepcopy
from typing import Union
from spa2num.converter import to_number
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
    


def remove_tildes(string:str) -> str:
    text = deepcopy(string)
    tildes = {
            '??': 'A', '??': 'E', '??': 'I', '??': 'O', '??': 'U',
            '??': 'A', '??': 'E', '??': 'I', '??': 'O', '??': 'U',
            '??': 'A', '??': 'E', '??': 'I', '??': 'O', '??': 'U',
            '??': 'a', '??': 'e', '??': 'i', '??': 'o', '??': 'u',
            '??': 'a', '??': 'e', '??': 'i', '??': 'o', '??': 'u',
            '??': 'a', '??': 'e', '??': 'i', '??': 'o', '??': 'u'
            }

    for k,v in tildes.items():
        text = text.replace(k, v)

    return text



def retokenizer(text:str, pattern=None, style:str=None) -> list:
    flag = False
    if not pattern:
        if style == 'alphanumeric':
            pattern = re.compile(r'\b(\w?\w+)', re.IGNORECASE)
            flag = True
        elif style == 'alphabetic':
            pattern = re.compile(r'\b([A-Za-z??-????-????-??]?[A-Za-z??-????-????-??]+)', re.IGNORECASE)
            flag = True
        elif style == 'numeric':
            pattern = re.compile(r'([0-9]+[.,]?[0-9]+([.,]?[0-9]+)?([.,]?[0-9]+)?([.,][0-9]+)?)')
            flag = True
        elif style == 'Name':
            pattern = re.compile(''.join(
                    [r'(?:[A-Z??-??][A-Za-z??-????-????-??]+\s?)',
                     r'+(?:(?:[a-z??-??]{1,4}\s)?',
                     r'(?:[A-Z??-??][A-Za-z??-????-????-??]+\s?)+)'])
                    )
            flag = True
        else:
            result = text.split()
    if flag or pattern:
        result = re.findall(pattern, text)
    
    return result