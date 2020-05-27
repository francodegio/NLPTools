""" 
    A suite of functions to evaluate word similarity. If you're interested
    in knowing how similar words are, please use the rapidfuzz library.
"""

from rapidfuzz import fuzz
from typing import Optional


def is_similar_word(
        word1: str, 
        word2: str, 
        threshold: float = 0.8
    ) -> bool:
    """
    Compare the given words and returns true or false if their coincidence
    are upper the compare_threshold param.

    Arguments:
    ---------
    -word1: First given word
    -word2: Second given word
    -compare_threshold: Comparition range. 0:no similarity, 1:complete similarity

    Return:
    ------
    -True or false if the comparission number is bigger or lower than compare_threslhold
    """
    if word1 and word2:
        similarity = fuzz.QRatio(word1, word2)
        return similarity / 100 > threshold


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
        Level required to be similar, by default 0.8.
    -ratio_func: str
        Function to evaluate similarity. Can be `ratio` or `QRatio`, by default `ratio`.
    Return:
    ------
    - Bool
        True if similarity between words is above the threshold.
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
    ratio_func : str, optional
        The function that will evaluate the similarity. Can be `ratio` or `QRatio`, by default 'ratio'.

    Returns
    -------
    Optional[str]
        Returns the word in the sentence that is similar to the one provided. If there are not similar
        words to the one provided, will return None.
    """
    similar_list = [
        w
        for w in list_of_words
        if is_similar_sentence(w.lower(), word.lower(), threshold, ratio_func)
    ]
    
    if similar_list:
        return similar_list[0]