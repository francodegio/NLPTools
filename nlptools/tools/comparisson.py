from rapidfuzz import fuzz


def is_similar_word(word1, word2, compare_threshold=0.8):
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
        return similarity/100


def is_similar_sentence(sentence1, sentence2, compare_threshold=0.8):
    """
    Compare the given words and returns true or false if their coincidence
    are upper the compare_threshold param.

    Arguments:
    ---------
    -sentence1: First given sentence
    -sentence2: Second given sentence
    -compare_threshold: Comparition range. 0:no similarity, 1:complete similarity

    Return:
    ------
    -True or false if the comparission number is bigger or lower than compare_threslhold
    """
    if sentence1 and sentence2:
        sentence_compare = fuzz.ratio(sentence1, sentence2)
        return sentence_compare / 100 > compare_threshold


def similar_word_in_sentence(word, sentence_list, similarity=0.8):
    words_in_sentence = [True for w in sentence_list if is_similar_sentence(w.lower(), word.lower(), similarity)]
    return any(words_in_sentence)


def is_sentence_contained_in_sentence(short_sentence: str, long_sentence: str, compare_threshold: int = 0.8) -> bool:
    """Receives a short and a long sentence and checks (with a compare threshold between 0 and 1) if the short sentence is contained inside the long sentence

    Arguments:
        short {str} -- Short sentence to look for
        long {str} -- Long sentence to look inside
        compare_threshold {int} -- Compare threshold to detect acceptable similarities

    Returns:
        bool -- True if the short sentence is contained inside the long one
    """
    compare_value = fuzz.token_set_ratio(short_sentence, long_sentence)
    
    return compare_value > compare_threshold * 100


def get_similar_word_in_sentence(word, sentence_list, similarity=0.8):
    similar_list = [w for w in sentence_list if is_similar_sentence(w.lower(), word.lower(), similarity)]
    if similar_list:
        return similar_list[0]