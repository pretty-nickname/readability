# -*- encoding: utf-8 -*-

import string

from features.base_extractor import BaseExtractor

from nltk.tokenize import sent_tokenize
from preprocess.basic_preprocess import tokenize

import statistics


class SentenceLengthExtractor(BaseExtractor):
    def get_features(self, text):
        features = dict()
        features.update(self._count_words_in_sent(text=text))
        return features

    @staticmethod
    def _count_words_in_sent(text):
        sents = sent_tokenize(text)
        words_number = [len(tokenize(x)) for x in sents]
        return {
            'average_sent_length': statistics.mean(words_number),
            'median_sent_length': statistics.median(words_number)
        }


class WordLengthExtractor(BaseExtractor):
    def get_features(self, tokenized_text):
        features = dict()
        features.update(self._count_chars_in_word(tokens=tokenized_text))
        features.update(self._count_long_words(tokens=tokenized_text))
        return features

    @staticmethod
    def _count_chars_in_word(tokens):
        return {
            'average_word_length': statistics.mean(map(float, [len(x) for x in tokens])),
            'median_word_length': statistics.median(map(float, [len(x) for x in tokens]))
        }

    def _count_long_words(self, tokens):
        if tokens:
            return {'long_words_ratio': len([x for x in tokens if self._if_long_word(x)]) / len(tokens)}
        else:
            return {'long_words_ratio': 0}

    @staticmethod
    def _if_long_word(token):
        token = token.lower()
        vowels_en = "aeiouy"
        vowels_ru = 'аяуюэеоёиы'

        if any([x in token for x in vowels_en]):
            count = 0
            if token[0] in vowels_en:
                count += 1
            for index in range(1, len(token)):
                if token[index] in vowels_en and token[index - 1] not in vowels_en:
                    count += 1
            if token.endswith("e"):
                count -= 1
            if count == 0:
                count += 1
            return count > 4
        else:
            return len([x for x in token if x in vowels_ru]) > 4


class VarietyExtractor(BaseExtractor):
    def get_features(self, lemmatized_text):
        features = dict()
        # features['unique_ttr'] = self._count_variety(lemmatized_text=lemmatized_text)
        features['nav'] = self._count_nav(lemmatized_text=lemmatized_text)
        return features

    @staticmethod
    def _count_variety(lemmatized_text):
        return len(set([x['lemma'] for x in lemmatized_text])) / len(lemmatized_text)

    @staticmethod
    def _count_nav(lemmatized_text):
        all_nouns = list()
        all_adj = list()
        all_verbs = list()

        for x in lemmatized_text:
            if x['pos'] in {'NOUN', 'n'}:
                all_nouns.append(x['lemma'])
            elif x['pos'] in {'VERB', 'INFN', 'v'}:
                all_verbs.append(x['lemma'])
            elif x['pos'] in {'ADJF', 'ADJS', 'a'}:
                all_adj.append(x['lemma'])

        ttr_n = len(set(all_nouns)) / len(all_nouns) if len(all_nouns) > 0 else -1
        ttr_a = len(set(all_adj)) / len(all_adj) if len(all_adj) > 0 else -1
        ttr_v = len(set(all_verbs)) / len(all_verbs) if len(all_verbs) > 0 else -1

        if ttr_n > 0:
            return ttr_a * ttr_v / ttr_n
        else:
            return -1

