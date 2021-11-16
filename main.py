# -*- encoding: utf-8 -*-

import pandas as pd

from pathlib import Path

# from features.freq_extractor import RuFreqExtractor, EnFreqExtractor
from features.foreign_extractor import ForeignExtractor
# from features.punct_extractor import PunctuationExtractor
from features.old_forms_extractor import OldFormsExtractor
# from features.traditional_extractor import VarietyExtractor, SentenceLengthExtractor, WordLengthExtractor
from features.syntax_extractor import SyntaxExtractor

from preprocess.basic_preprocess import tokenize, lemmatize


from joblib import Parallel, delayed


def flat_features(features):
    flatten_features = dict()
    for k, v in features.items():
        if isinstance(v, dict):
            for kk, vv in v.items():
                flatten_features[f"{k}_{kk}"] = vv
        else:
            flatten_features[k] = v
    return flatten_features


def construct_features(text, language):
    features = {'text': text['text'],
                'filename': text['filename'],
                'category': text['category']}
    text = text['text']

    tokens = tokenize(text, language=language)
    lemmas = lemmatize(tokens, language=language)

    if language == 'ru':
        features.update(RuFreqExtractor().get_features(lemmas))
    elif language == 'en':
        features.update(EnFreqExtractor().get_features(lemmas))
    # features.update(ForeignExtractor().get_features(lemmas))
    features.update(PunctuationExtractor().get_features(text))
    # features.update(OldFormsExtractor().get_features(lemmas))
    features.update(VarietyExtractor().get_features(lemmas))
    features.update(WordLengthExtractor().get_features(tokens))
    features.update(SentenceLengthExtractor().get_features(text))

    return flat_features(features)


def add_features(text_with_features, language):
    raw_text = text_with_features['text']

    # tokens = tokenize(raw_text, language=language)
    # lemmas = lemmatize(tokens, language=language)

    text_with_features.update(SyntaxExtractor(language=language).get_features(raw_text))
    # text_with_features.update(VarietyExtractor().get_features(lemmas))
    # text_with_features.update(WordLengthExtractor().get_features(tokens))
    # text_with_features.update(SentenceLengthExtractor().get_features(raw_text))

    return text_with_features


def process_texts(texts, mode='parallel', language='ru'):
    if mode == 'parallel':
        texts = Parallel(n_jobs=15)(
            delayed(add_features)(text_features, language)
            for text_features in texts
        )
    else:
        texts = [add_features(text_features, language) for text_features in texts]

    df = pd.DataFrame(texts)
    return df


if __name__ == "__main__":
    language = 'ru1 '
    # data_dict = pd.read_csv('data/onestop/test.csv').to_dict('records')

    # df = pd.read_csv('data/fiction_previews/ratings_test.csv')
    # categories = dict(zip(df['filename'], df['category']))
    #
    # data_folder = Path(__file__).parent.absolute() / f"data/fiction_previews/test"
    # book_paths = [d for d in data_folder.iterdir() if d.suffix == '.txt' and d.name in df['filename'].unique()]
    # data_dict = list()
    # for path in book_paths:
    #     with open(path, 'r') as f:
    #         text = f.read()
    #     data_dict.append({'text': text,
    #                       'filename': path.name,
    #                       'category': categories[path.name]})


    # with open('data/onestop/train.csv', 'r') as f:
    #     test_text = f.read()

    for table in [
        'fiction_read_test_features.csv',
        'fiction_read_train_features.csv',
        'fiction_recommended_test_features.csv',
        'fiction_recommended_train_features.csv',
        'fiction_previews_test_features.csv',
        'fiction_previews_train_features.csv'
        # 'common_core_test_features.csv',
        # 'common_core_train_features.csv',
        # 'commonlit_test_features.csv',
        # 'commonlit_train_features.csv',
        # 'onestop_test_features.csv',
        # 'onestop_train_features.csv'
    ]:

        texts_features = pd.read_csv(table).to_dict(orient='records')

        df = process_texts(texts_features, mode='parallel', language=language)

        df.to_csv(table, index=False)

        print(f'Finished {table}')

