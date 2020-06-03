import pytest
from unittest import mock
from rapidfuzz import fuzz
from nlptools.comparisson import is_similar_word, is_similar_sentence, similar_word_in_sentence, is_sentence_contained_in_longer_sentence, \
    get_similar_word_in_sentence


class TestCompareFunctions:
    def test_is_similar_word(self):

        assert isinstance(is_similar_word('Apple', 'Banana'), bool)
        assert is_similar_word('Apple', 'Banana') == False

        assert isinstance(is_similar_word('Apple', 'Apple'), bool)
        assert is_similar_word('Apple', 'Apple') == True

        assert isinstance(is_similar_word('Apple', 'Aple', 0.7), bool)
        assert is_similar_word('Apple', 'Aple', 0.7) == True

    def test_is_similar_sentence(self):

        assert isinstance(is_similar_sentence(
            'Luke, I am your Father....', 'Noooooooooooo!'), bool)
        assert is_similar_sentence(
            'Luke, I am your Father....', 'Noooooooooooo!') == False

        assert isinstance(is_similar_sentence(
            'Luke, I am your Father....', 'Luke, I am probably your Father', 0.4), bool)
        assert is_similar_sentence(
            'Luke, I am your Father....', 'Luke, I am probably your Father', 0.4) == True

        assert isinstance(is_similar_sentence('Luke, I am your Father....',
                                              'Luke, I am not your Father', 0.5, ratio_func='QRatio'), bool)
        assert is_similar_sentence(
            'Luke, I am your Father....', 'Luke, I am not your Father', 0.5, ratio_func='QRatio') == True

    def test_is_sentence_contained_in_longer_sentence(self):

        assert isinstance(is_sentence_contained_in_longer_sentence(
            'the best of you', 'Is someone getting the best, the best, the best of you?'), bool)
        assert is_sentence_contained_in_longer_sentence(
            'the best of you', 'Is someone getting the best, the best, the best of you?') == True

        assert isinstance(is_sentence_contained_in_longer_sentence(
            'the worst of you', 'Is someone getting the best, the best, the best of you?'), bool)
        assert is_sentence_contained_in_longer_sentence(
            'the worst of you', 'Is someone getting the best, the best, the best of you?') == False

        assert isinstance(is_sentence_contained_in_longer_sentence(
            'the best of you', 'Is someone getting the best, the best, the best of you?', 0.9), bool)
        assert is_sentence_contained_in_longer_sentence(
            'the best of you', 'Is someone getting the best, the best, the best of you?', 0.9) == True

    def test_get_similar_word_in_sentence(self):
        assert isinstance(get_similar_word_in_sentence(
            'Apple', ['The', 'Aple', 'is', 'red'], 0.5), type('Apple'))
        assert get_similar_word_in_sentence(
            'Apple', ['The', 'Aple', 'is', 'red'], 0.5) == 'Aple'
