# -*- encoding: utf-8 -*-

import string

from features.base_extractor import BaseExtractor


class OldFormsExtractor(BaseExtractor):
    def get_features(self, lemmatized_text):
        features = dict()
        features['-oyu'] = self._count_oyu(lemmatized_text=lemmatized_text)
        return features

    @staticmethod
    def _count_oyu(lemmatized_text):
        return len([x for x in lemmatized_text if x['pos'] == 'NOUN' and x['gender'] == 'femn' and x['form'].endswith('ою')])
