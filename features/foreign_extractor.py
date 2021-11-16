# -*- encoding: utf-8 -*-

from pathlib import Path

from features.base_extractor import BaseExtractor

import unicodedata


def strip_accents(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')


class ForeignExtractor(BaseExtractor):
    def __init__(self):
        super().__init__()
        self.foreign_list = self._load_foreign_list()

    def _load_foreign_list(self):
        list_path = Path(__file__).parent.parent.absolute() / 'resources/foreign_list.txt'
        if list_path.is_file():
            with open(list_path, 'r') as f:
                foreign_list = [x.strip() for x in f.readlines()]
            return foreign_list
        else:
            return self._gain_foreign_list(list_path=list_path)

    @staticmethod
    def _gain_foreign_list(list_path):
        dictionary_path = Path(__file__).parent.parent.absolute() / 'resources/Учебный-сл.-ин.-сл.-15.txt'
        foreign_list = list()
        with open(dictionary_path, 'r') as f:
            lines = [x.strip() for x in f.readlines()]
        for line in lines:
            if not line:
                continue
            first_word = line.split()[0].rstrip(',').replace('¹', '').replace('²', '')
            if first_word == first_word.upper() and len(first_word.replace('.', '')) > 1:
                value = strip_accents(first_word.lower())
                foreign_list.append(value)
        with open(list_path, 'w') as f:
            for word in foreign_list:
                f.write(f"{word}\n")
        return foreign_list

    def get_features(self, lemmatized_text):
        features = dict()
        features['foreign_ratio'] = self._get_foreign_ratio(lemmatized_text=lemmatized_text)
        return features

    def _get_foreign_ratio(self, lemmatized_text):
        return len([x for x in lemmatized_text if x['lemma'] in self.foreign_list]) / len(lemmatized_text)
