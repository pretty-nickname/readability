import spacy
from spacy import displacy

from features.base_extractor import BaseExtractor

import statistics


class SyntaxExtractor(BaseExtractor):
    def __init__(self, language):
        super().__init__()
        if language == 'ru':
            self.nlp = spacy.load("ru_core_news_lg")
        else:
            self.nlp = spacy.load("en_core_web_sm")

    def get_features(self, text):
        features = dict()
        doc = self.nlp(text)
        features.update(self._get_tree_depth(doc))
        features.update(self._get_max_distance(doc))
        features.update(self._count_clauses(doc))
        features.update(self._count_adverbal_clauses(doc))
        features.update(self._count_adnominal_clauses(doc))
        features.update(self._count_complement_clauses(doc))
        features.update(self._count_open_complement_clauses(doc))
        features.update(self._count_nominal_modifiers(doc))
        features.update(self._count_nmod_depth(doc))
        return features

    def _get_tree_depth(self, doc):
        tree_depths = [self._walk_tree_depth(sent.root, 0) for sent in doc.sents]
        return {
            'average_tree_depth': statistics.mean(tree_depths),
            'median_tree_depth': statistics.median(tree_depths),
            'max_tree_depth': max(tree_depths)
        }

    def _walk_tree_depth(self, node, depth):
        if node.n_lefts + node.n_rights > 0:
            return max(self._walk_tree_depth(child, depth + 1) for child in node.children)
        else:
            return depth

    @staticmethod
    def _get_max_distance(doc):
        max_distances = list()
        for sent in doc.sents:
            max_distance = 0
            for token in sent:
                max_distance = max(abs(token.i - token.head.i), max_distance)
            max_distances.append(max_distance)

        return {
            'average_max_distance': statistics.mean(max_distances),
            'median_max_distance': statistics.median(max_distances),
            'max_max_distance': max(max_distances)
        }

    def _count_clauses(self, doc):
        verbs_number = [len(self._find_verbs_with_nsubj(sent)) for sent in doc.sents]
        return {
            'average_clauses': statistics.mean(verbs_number),
            'median_clauses': statistics.median(verbs_number),
            'max_clauses': max(verbs_number)
        }

    @staticmethod
    def _find_verbs_with_nsubj(sent):
        verbs = list()

        root_token = None
        for token in sent:
            if token.dep_ == "ROOT":
                root_token = token

        if any([x.dep_ in {"nsubj", "nsubj:pass"} for x in root_token.children]):
            verbs.append(root_token)

        for token in sent:
            ancestors = list(token.ancestors)
            if token.pos_ == "VERB" and len(ancestors) == 1 and ancestors[0] == root_token:
                if any([x.dep_ == "nsubj" for x in token.children]):
                    verbs.append(token)
        return verbs

    @staticmethod
    def _count_adverbal_clauses(doc):
        advcl_number = list()

        for sent in doc.sents:
            advcl = list()
            for token in sent:
                if token.dep_ == "advcl":
                    advcl.append(token)
            advcl_number.append(len(advcl))

        return {
            'average_advcl': statistics.mean(advcl_number),
            'median_advcl': statistics.median(advcl_number),
            'max_advcl': max(advcl_number)
        }

    @staticmethod
    def _count_adnominal_clauses(doc):
        acl_number = list()

        for sent in doc.sents:
            acl = list()
            for token in sent:
                if token.dep_ == "acl":
                    acl.append(token)
            acl_number.append(len(acl))

        return {
            'average_acl': statistics.mean(acl_number),
            'median_acl': statistics.median(acl_number),
            'max_acl': max(acl_number)
        }

    @staticmethod
    def _count_complement_clauses(doc):
        ccomp_number = list()

        for sent in doc.sents:
            ccomp = list()
            for token in sent:
                if token.dep_ == "ccomp":
                    ccomp.append(token)
            ccomp_number.append(len(ccomp))

        return {
            'average_ccomp': statistics.mean(ccomp_number),
            'median_ccomp': statistics.median(ccomp_number),
            'max_ccomp': max(ccomp_number)
        }

    @staticmethod
    def _count_open_complement_clauses(doc):
        xcomp_number = list()

        for sent in doc.sents:
            xcomp = list()
            for token in sent:
                if token.dep_ == "xcomp":
                    xcomp.append(token)
            xcomp_number.append(len(xcomp))

        return {
            'average_xcomp': statistics.mean(xcomp_number),
            'median_xcomp': statistics.median(xcomp_number),
            'max_xcomp': max(xcomp_number)
        }

    @staticmethod
    def _count_nominal_modifiers(doc):
        nmod_number = list()

        for sent in doc.sents:
            nmod = list()
            for token in sent:
                if token.dep_ in {"nmod", "nmod:poss"}:
                    nmod.append(token)
            nmod_number.append(len(nmod))

        return {
            'average_nmod': statistics.mean(nmod_number),
            'median_nmod': statistics.median(nmod_number),
            'max_nmod': max(nmod_number)
        }

    def _count_nmod_depth(self, doc):
        nmod_depths = list()

        for sent in doc.sents:
            nmod_depth = list()
            for token in sent:
                if token.dep_ in {"nmod", "nmod:poss"}:
                    nmod_depth.append(self._walk_tree_nmod_depth(token, 1))
            nmod_depths.append(max(nmod_depth) if nmod_depth else 0)

        return {
            'average_nmod_depth': statistics.mean(nmod_depths),
            'median_nmod_depth': statistics.median(nmod_depths),
            'max_nmod_depth': max(nmod_depths)
        }

    def _walk_tree_nmod_depth(self, node, depth):
        if node.n_lefts + node.n_rights > 0:
            try:
                return max(
                    self._walk_tree_depth(child, depth + 1)
                    for child in node.children
                    if child.dep_ in {"nmod", "nmod:poss"}
                )
            except ValueError:
                return 0
        else:
            return depth


if __name__ == "__main__":
    # nlp = spacy.load("ru_core_news_lg")
    # nlp = spacy.load("en_core_web_sm")

    text = "Я шёл домой. На улице начался дождь."

    # sentence = "He eats cheese, but he won't eat ice cream."
    # sentence = "Смеркалось, я встал, оделся и пошёл в сад, он встал из-за стола, а она зажгла свечу."
    # sentence = "Я решил пойти, поскольку на улице уже смеркалось."
    # sentence = "Я смеялся до слёз, читая неправильные контексты для игры."
    # sentence = "Школьники, начавшие играть в нашу игру, перестанут делать домашние задания"
    # sentence = "Когда я разрабатываю игру, я забываю обо всём"
    # sentence = "То, что она сказала, может быть полезно."
    # sentence = "He said that you like cheese."
    # sentence = "Он сказал, что ты любишь сыр."
    # sentence = "Я начал копать."
    # sentence = "Это определяет невозможность приобретения способности."

    # doc = nlp(sentence)
    # print(SyntaxExtractor(language='ru')._count_nmod_depth(doc))

    print(SyntaxExtractor(language='en').get_features(text))

    # displacy.serve(doc)





