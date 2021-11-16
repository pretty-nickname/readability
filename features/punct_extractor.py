# -*- encoding: utf-8 -*-

import string

from features.base_extractor import BaseExtractor


class PunctuationExtractor(BaseExtractor):
    def get_features(self, raw_text):
        features = dict()
        features['punctuation_number'] = self._count_punct(raw_text=raw_text)
        features['semicolons_number'] = self._count_semicolons(raw_text=raw_text)
        return features

    @staticmethod
    def _count_punct(raw_text):
        count = lambda l1, l2: sum([1 for x in l1 if x in l2])
        return count(raw_text, set(string.punctuation))

    @staticmethod
    def _count_semicolons(raw_text):
        return raw_text.count(';')
