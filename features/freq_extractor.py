# -*- encoding: utf-8 -*-

from pathlib import Path
import pandas as pd

import statistics

from features.base_extractor import BaseExtractor
from utils.pos_translation import POS_PYMORPHY2FREQ, POS_NLTK2FREQ, POS_FREQ2BNC


class FrequencyExtractor(BaseExtractor):
    def __init__(self):
        super().__init__()
        self.freq_df = pd.DataFrame
        self.freq_dictionary = dict()
        self.freq_ratings = dict()

    def get_features(self, lemmatized_text):
        features = dict()
        features['average_frequency'] = self._get_average_frequency(lemmatized_text=lemmatized_text)
        features['median_frequency'] = self._get_median_frequency(lemmatized_text=lemmatized_text)
        for category, ratings in self.freq_ratings.items():
            features[category] = dict()
            for rating_name, rating in ratings.items():
                if category == 'common':
                    features[category][rating_name] = self._get_ratio_from_rating(lemmatized_text=lemmatized_text,
                                                                                  rating=rating)
                else:
                    features[category][rating_name] = self._get_ratio_from_rating(lemmatized_text=lemmatized_text,
                                                                                  rating=rating,
                                                                                  pos=category)
        return features

    @staticmethod
    def _get_ratio_from_rating(lemmatized_text, rating, pos=None):
        return 0

    def _get_average_frequency(self, lemmatized_text):
        return statistics.mean([self.freq_dictionary.get(x['lemma'], 0.) for x in lemmatized_text])

    def _get_median_frequency(self, lemmatized_text):
        return statistics.median([self.freq_dictionary.get(x['lemma'], 0.) for x in lemmatized_text])


class EnFreqExtractor(FrequencyExtractor):
    def __init__(self):
        super().__init__()
        dictionary_path = Path(__file__).parent.parent.absolute() / 'resources/freq_ratings/en/all_words.txt'
        self.freq_df = pd.read_csv(dictionary_path, sep='\t')
        self.freq_df = self.freq_df[self.freq_df['Lemma'] != '@']
        self.freq_dictionary = dict(zip(self.freq_df['Lemma'], self.freq_df['Freq']))
        self.freq_ratings = self._load_freq_ratings()

    def _load_freq_ratings(self):
        ratings_directory = Path(__file__).parent.parent.absolute() / 'resources/freq_ratings/en'
        all_ratings = [fp for fp in ratings_directory.iterdir() if fp.suffix == '.csv']
        if not all_ratings:
            return self._gain_freq_ratings(ratings_directory=ratings_directory)
        else:
            ratings = {'common': {},
                       's': {},
                       'v': {},
                       'a': {},
                       'adv': {}}
            for path in all_ratings:
                df = pd.read_csv(path)
                df_name = path.stem
                params = df_name.split('_')
                if len(params) == 2:
                    ratings['common'][params[1]] = df
                elif len(params) == 3:
                    ratings[params[0]][params[2]] = df
            return ratings

    def _gain_freq_ratings(self, ratings_directory):
        ratings = {}
        df_sorted = self.freq_df.sort_values(by=['Freq'], ascending=False)
        ratings['common'] = {}
        for i in range(10):
            df_cutted = df_sorted.iloc[i * 100:(i + 1) * 100]
            ratings['common'][str((i + 1) * 100)] = df_cutted
            df_cutted.to_csv(f'{ratings_directory}/{i * 100}_{(i + 1) * 100}.csv', index=False)

        for pos in ['s', 'v', 'a', 'adv']:
            ratings[pos] = {}
            df_pos = self.freq_df[self.freq_df['PoS'] == POS_FREQ2BNC[pos]]
            df_sorted = df_pos.sort_values(by=['Freq'], ascending=False)
            for i in range(10):
                df_cutted = df_sorted.iloc[i * 100:(i + 1) * 100]
                ratings[pos][str((i + 1) * 100)] = df_cutted
                df_cutted.to_csv(f'{ratings_directory}/{pos}_{i * 100}_{(i + 1) * 100}.csv', index=False)
        return ratings

    @staticmethod
    def _get_ratio_from_rating(lemmatized_text, rating, pos=None):
        if pos is not None:
            lemmas = [x['lemma'] for x in lemmatized_text if POS_NLTK2FREQ.get(x['pos'], '') == pos]
        else:
            lemmas = [x['lemma'] for x in lemmatized_text]
        try:
            return len([x for x in lemmas if x in rating['Lemma'].to_list()]) / len(lemmas)
        except ZeroDivisionError:
            return -1


class RuFreqExtractor(FrequencyExtractor):
    def __init__(self):
        super().__init__()
        dictionary_path = Path(__file__).parent.parent.absolute() / 'resources/freqrnc2011.csv'
        self.freq_df = pd.read_csv(dictionary_path, sep='\t')
        self.freq_dictionary = dict(zip(self.freq_df['Lemma'], self.freq_df['Freq(ipm)']))
        self.freq_ratings = self._load_freq_ratings()

    def _load_freq_ratings(self):
        ratings_directory = Path(__file__).parent.parent.absolute() / 'resources/freq_ratings/ru'
        all_ratings = [fp for fp in ratings_directory.iterdir() if fp.suffix == '.csv']
        if not all_ratings:
            return self._gain_freq_ratings(ratings_directory=ratings_directory)
        else:
            ratings = {'common': {},
                       's': {},
                       'v': {},
                       'a': {},
                       'adv': {}}
            for path in all_ratings:
                df = pd.read_csv(path)
                df_name = path.stem
                params = df_name.split('_')
                if len(params) == 2:
                    ratings['common'][params[1]] = df
                elif len(params) == 3:
                    ratings[params[0]][params[2]] = df
            return ratings

    def _gain_freq_ratings(self, ratings_directory):
        ratings = {}
        df_sorted = self.freq_df.sort_values(by=['Freq(ipm)'], ascending=False)
        ratings['common'] = {}
        for i in range(10):
            df_cutted = df_sorted.iloc[i * 100:(i + 1) * 100]
            ratings['common'][str((i + 1) * 100)] = df_cutted
            df_cutted.to_csv(f'{ratings_directory}/{i * 100}_{(i + 1) * 100}.csv', index=False)
        for pos in ['s', 'v', 'a', 'adv']:
            ratings[pos] = {}
            df_pos = self.freq_df[self.freq_df['PoS'] == pos]
            df_sorted = df_pos.sort_values(by=['Freq(ipm)'], ascending=False)
            for i in range(10):
                df_cutted = df_sorted.iloc[i * 100:(i + 1) * 100]
                ratings[pos][str((i + 1) * 100)] = df_cutted
                df_cutted.to_csv(f'{ratings_directory}/{pos}_{i * 100}_{(i + 1) * 100}.csv', index=False)
        return ratings

    @staticmethod
    def _get_ratio_from_rating(lemmatized_text, rating, pos=None):
        if pos is not None:
            lemmas = [x['lemma'] for x in lemmatized_text if POS_PYMORPHY2FREQ.get(x['pos'], '') == pos]
        else:
            lemmas = [x['lemma'] for x in lemmatized_text]
        try:
            return len([x for x in lemmas if x in rating['Lemma'].to_list()]) / len(lemmas)
        except ZeroDivisionError:
            return -1

