""" 
    A suite of functions to evaluate word similarity. If you're interested
    in knowing how similar words are, please use the rapidfuzz library.
"""

from rapidfuzz import fuzz
from typing import Optional, List


def is_similar_word(
        word1: str, 
        word2: str, 
        threshold: float = 0.8,
        ratio_func: str = 'ratio'
    ) -> bool:
    """Evaluates if two given words are similar acording to a threshold.

    Parameters
    ----------
    - word1 : `str`
        A word intended to evaluate.
    - word2 : `str`
        A second word intended to compare to the first.
    - threshold : `float`, optional
        The level of similarity required for considering them equivalent, by default 0.8.
    - ratio_func : `str`, {'ratio', 'QRatio'}, optional.
        Choose which function to do the evaluation, by default 'ratio'.

    Returns
    -------
    - bool
        `True` if the similarity ratio is over the threshold.

    Examples 
    --------
    >>> from nlptools.comparisson import is_similar_word
    >>> is_similar_word('Apple','Banana')
    False
    >>> is_similar_word('Apple','Apple')
    True    
    >>> is_similar_word('Apple','Aple', 0.7)
    True

    """
    if ratio_func == 'ratio':
        score = fuzz.ratio(word1, word2)
    elif ratio_func == 'QRatio':
        score = fuzz.QRatio(word1, word2)
        
    return score / 100 > threshold


def is_similar_sentence(
    sentence1: str, 
    sentence2: str, 
    threshold: float = 0.8, 
    ratio_func: str = 'ratio'
    ) -> bool:
    """
    Compare the given words and returns true or false if their coincidence
    are upper the compare_threshold param.

    Arguments:
    ---------
    - sentence1: str
        First given sentence.
    - sentence2: str
        Second given sentence.
    - threshold: float, optional
        The level of similarity required for considering them equivalent, by default 0.8.
    -ratio_func: str
        Function to evaluate similarity. Can be `ratio` or `QRatio`, by default `ratio`.
    Return:
    ------
    - Bool
        True if similarity between words is above the threshold.

    Examples
    --------
    >>> from nlptools.comparisson import is_similar_sentence
    >>> is_similar_sentence('Luke, I am your Father....','Noooooooooooo!')
    False
    >>> is_similar_sentence('Luke, I am your Father....','Luke, I am probably your Father', 0.4)
    True
    >>> is_similar_sentence('Luke, I am your Father....','Luke, I am not your Father',0.5, ratio_func='QRatio')
    True

    """
    
    if sentence1 and sentence2:
        if ratio_func == 'ratio':
            score = fuzz.ratio(sentence1, sentence2)
        elif ratio_func == 'QRatio':
            score = fuzz.QRatio(sentence1, sentence2)

        return score / 100 > threshold


def similar_word_in_sentence(
    word: str, 
    sentence_list: list, 
    threshold: float = 0.8, 
    ratio_func: str = 'ratio'
) -> bool:
    """ Searches if a certain word is similar to other contained in a sentence.

    Parameters
    ----------
    word : str
        The word intended to find in sentence.
    sentence_list : list
        The list of words of a sentences where the words might be.
    threshold : float, optional
        The level of similarity required for considering them equivalent, by default 0.8.
    ratio_func : str, optional
        The function that will evaluate the similarity. Can be `ratio` or `QRatio`, by default 'ratio'.

    Returns
    -------
    bool
        True if there is a similar word in the sentence.
    Examples
    -------
    >>> similar_word_in_sentence('cat',['The','cat','is','under','the','table'])
    True
    >>> similar_word_in_sentence('cat',['The','catd','is','under','the','table'],0.9)
    False

    """
    words_in_sentence = [
        True 
        for w in sentence_list 
        if is_similar_sentence(w.lower(), word.lower(), threshold, ratio_func)
    ]
    return any(words_in_sentence)


def is_sentence_contained_in_longer_sentence(
        short_sentence: str, 
        long_sentence: str, 
        threshold: int = 0.8
    ) -> bool:
    """ Checks if a shorter sentence is within a longer sentence.

    Parameters
    ----------
    short_sentence : str
        The sentence that might be within the other string.
    long_sentence : str
        The sentence that might contain the short sentence.
    threshold : int, optional
        The level of similarity required for considering them equivalent, by default 0.8.

    Returns
    -------
    bool
        True if short sentence is partially contained in the long sentence.

    Examples
    -------
    >>> is_sentence_contained_in_longer_sentence('the best of you','Is someone getting the best, the best, the best of you?')
    True
    >>> is_sentence_contained_in_longer_sentence('the worst of you','Is someone getting the best, the best, the best of you?')
    False
    >>> is_sentence_contained_in_longer_sentence('the best of you','Is someone getting the best, the best, the best of you?',0.9)
    True
    """
    
    compare_value = fuzz.token_set_ratio(short_sentence, long_sentence)
    
    return compare_value > threshold * 100


def get_similar_word_in_sentence(
        word: str, 
        list_of_words: list, 
        threshold: float = 0.8, 
        ratio_func: str = 'ratio'
    ) -> Optional[str]:
    """ Returns the word in the sentence that is similar to the word provided.

    Parameters
    ----------
    word : str
        Word intended to be found in the list of words.
    list_of_words : list
        List of words of a sentence.
    threshold : float, optional
        The level of similarity required for considering them equivalent, by default 0.8.
    ratio_func : str {'ratio', 'QRatio'}, optional
        The function that will evaluate the similarity. Can be `ratio` or `QRatio`, by default 'ratio'.

    Returns
    -------
    Optional[str]
        Returns the word in the sentence that is similar to the one provided. If there are not similar
        words to the one provided, will return None.
    Example
    ------
    >>> from nlptools.comparisson import get_similar_word_in_sentence
    >>> get_similar_word_in_sentence('Apple',['The','Apple','is','red'])
    'Apple'
    
    """

    similar_list = [
        w
        for w in list_of_words
        if is_similar_sentence(w.lower(), word.lower(), threshold, ratio_func)
    ]
    
    if similar_list:
        return similar_list[0]


def any_word_in_sentence(
        list_of_words: List[str], 
        list_of_words_in_sentence: List[str], 
        threshold: float = 0.8, 
        ratio_func: str = 'ratio'
    ) -> bool:
    """Analizes if any word in your list is contained in a sentence. Make sure you provide lists for both.

    Parameters
    ----------
    list_of_words : List[str]
        List containing the words in string format, that you hope to find in the sentence.
    list_of_words_in_sentence : List[str]
        A list of strings generated from a sentence. For best results, 
        use a tokenizer to generate the list of words.
    threshold : float (0,1), optional
        The level of similarity required for considering them equivalent. 0 would be 
        completely different and 1 exactly the same, by default 0.8.
    ratio_func : str {'ratio', 'QRatio'}, optional
        The function that will evaluate the similarity. Can be `ratio` or `QRatio`, by default 'ratio'.

    Returns
    -------
    bool
        True if any of the keywords provided is similar 
        to any of the ones in the sentence. Otherwise will 
        return False.

    Examples
    -------
    >>> any_word_in_sentence(['hola', 'vieja'], ['tu', 'vieja', 'esta', 'en', 'bolas'], 0.8, 'QRatio')
    True
    >>> any_word_in_sentence(['keyword'], ['I', 'am', 'trying', 'to', 'find', 'a', 'keyword', 'in', 'this', 'sentence'], 0.8, 'QRatio')
    True
    >>> any_word_in_sentence(['tokenizer'], ['Please', 'do', 'not', 'use', '.split()', 'to', 'generate', 'the', 'list', 'of', 'words'], 0.8, 'ratio')
    False
    >>> any_word_in_sentence(['tokenizer'], ['Use', 'a', 'tokenizer', 'instead'], 0.99, 'ratio')
    True
    """
    result = False
    for word in list_of_words:  
        for other_word in list_of_words_in_sentence:
            result = is_similar_word(word, other_word, threshold=threshold, ratio_func=ratio_func)
            if result:
                break

        if result == True:
            break
    
    return result