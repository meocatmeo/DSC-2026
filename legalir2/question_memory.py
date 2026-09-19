"""Retrieve documents through similar labelled training questions.

The corpus retrievers match a question to a statute. This complementary channel
matches it to other questions and follows their document labels. Query IDs and
normalised duplicate text are always excluded, including during training.
"""
from collections import Counter, defaultdict

import numpy as np

from .cache import normalise_question
from .lexical import tokens

MEMORY_FEATURES = ("memory_best", "memory_vote", "memory_rr", "memory_support",
                   "memory_margin", "memory_top1", "memory_confidence")


def terms(text):
    words = tokens(text)
    counts = Counter("w:" + word for word in words)
    counts.update("b:" + a + " " + b for a, b in zip(words, words[1:]))
    # Character fragments make inflections, typos and Vietnamese syllable
    # boundaries less brittle without discarding accents or legal numbers.
    for word in words:
        padded = "^" + word + "$"
        counts.update("c:" + padded[i:i + 3] for i in range(len(padded) - 2))
    return counts


class QuestionMemory:
    def __init__(self, records):
        from scipy.sparse import csr_matrix

        if not records:
            raise ValueError("Question memory requires labelled training records")
        self.records = records
        self.ids = sorted(records)
        self.texts = [normalise_question(records[q]["question"]) for q in self.ids]
        if any(not records[q].get("answer") for q in self.ids):
            raise ValueError("Question memory has an unlabelled record")
        bags = [terms(records[q]["question"]) for q in self.ids]
        frequency = Counter(term for bag in bags for term in bag)
        self.vocabulary = {term: i for i, term in enumerate(sorted(frequency))}
        self.idf = np.asarray([np.log((1 + len(bags)) / (1 + frequency[term])) + 1
                               for term in self.vocabulary], dtype=np.float32)
        data, indices, indptr = [], [], [0]
        for bag in bags:
            for term, count in bag.items():
                indices.append(self.vocabulary[term])
                data.append(self.weight(term, count))
            indptr.append(len(data))
        matrix = csr_matrix((data, indices, indptr), shape=(len(bags), len(self.vocabulary)),
                            dtype=np.float32).multiply(self.idf).tocsr()
        norm = np.sqrt(np.asarray(matrix.multiply(matrix).sum(axis=1)).ravel())
        self.matrix = matrix.multiply(1 / np.maximum(norm, 1e-12)[:, None]).tocsr()

    @staticmethod
    def weight(term, count):
        scale = .35 if term.startswith("c:") else 1.
        return scale * (1 + np.log(count))

    def search(self, query, qid=None, neighbors=32, exclude_ids=()):
        from scipy.sparse import csr_matrix

        bag = terms(query)
        pairs = [(self.vocabulary[t], self.weight(t, n)) for t, n in bag.items()
                 if t in self.vocabulary]
        if not pairs:
            return {}, []
        indices, values = zip(*pairs)
        vector = csr_matrix((values, ([0] * len(indices), indices)),
                            shape=(1, len(self.vocabulary)), dtype=np.float32).multiply(self.idf).tocsr()
        norm = np.sqrt(vector.multiply(vector).sum())
        vector /= max(float(norm), 1e-12)
        similarities = (self.matrix @ vector.T).toarray().ravel()
        excluded = set(exclude_ids)
        text = normalise_question(query)
        for i, key in enumerate(self.ids):
            if key == qid or key in excluded or self.texts[i] == text:
                similarities[i] = -1
        picks = np.argsort(-similarities, kind="stable")[:neighbors]
        matches = [(self.ids[i], float(similarities[i])) for i in picks if similarities[i] > 0]
        if not matches:
            return {}, []
        evidence = defaultdict(lambda: [0., 0., 0., 0., 0., 0., matches[0][1]])
        mass = sum(s ** 4 for _, s in matches)
        for rank, (key, score) in enumerate(matches, 1):
            answers = set(self.records[key]["answer"])
            for doc in answers:
                features = evidence[doc]
                features[0] = max(features[0], score)
                features[1] += score ** 4 / len(answers) / max(mass, 1e-12)
                features[2] = max(features[2], 1. / rank)
                features[3] += 1. / neighbors
                features[5] = max(features[5], float(rank == 1))
        for doc, features in evidence.items():
            opposition = max((score for key, score in matches
                              if doc not in self.records[key]["answer"]), default=0.)
            features[4] = features[0] - opposition
        ranking = sorted(evidence, key=lambda d: (-evidence[d][1], -evidence[d][0], d))
        return dict(evidence), ranking

    def features(self, query, docs, qid=None, exclude_ids=()):
        evidence, ranking = self.search(query, qid, exclude_ids=exclude_ids)
        confidence = max((v[6] for v in evidence.values()), default=0.)
        absent = [0., 0., 0., 0., -confidence, 0., confidence]
        return np.asarray([evidence.get(d, absent) for d in docs], dtype=np.float64), ranking
