# -*- encoding: utf-8 -*-
import pymorphy2

from deeppavlov.models.tokenizers.ru_tokenizer import RussianTokenizer
from deeppavlov.models.tokenizers.nltk_tokenizer import NLTKTokenizer

import nltk
# nltk.download('wordnet')
# nltk.download('averaged_perceptron_tagger')
from nltk.corpus import wordnet


def tokenize(text, language='ru'):
    if language == 'ru':
        tokenizer = RussianTokenizer()
        return tokenizer([text])[0]
    elif language == 'en':
        tokenizer = NLTKTokenizer()
        return tokenizer([text])[0]


def get_wordnet_pos(pos_tag):
    """Map POS tag to first character lemmatize() accepts"""
    tag = pos_tag[0].upper()
    tag_dict = {
        "J": wordnet.ADJ,
        "N": wordnet.NOUN,
        "V": wordnet.VERB,
        "M": wordnet.VERB,
        "R": wordnet.ADV
    }

    return tag_dict.get(tag)


def lemmatize(tokens, language='ru'):
    lemmas = list()
    if language == 'ru':
        morph = pymorphy2.MorphAnalyzer()
        for token in tokens:
            parsed = morph.parse(token)[0]
            lemmas.append({'lemma': parsed.normal_form.replace('ё', 'е'),
                           'pos': parsed.tag.POS,
                           'gender': parsed.tag.gender,
                           'form': token})
    elif language == 'en':
        morph = nltk.WordNetLemmatizer()
        pos_tags = nltk.pos_tag(tokens)
        for token, pos_tag in zip(tokens, pos_tags):
            pos = get_wordnet_pos(pos_tag[1])
            if pos:
                lemma = morph.lemmatize(token, pos)
            else:
                lemma = morph.lemmatize(token, wordnet.NOUN)
            lemmas.append({'lemma': lemma.lower(),
                           'pos': pos,
                           'form': token})
    return lemmas
