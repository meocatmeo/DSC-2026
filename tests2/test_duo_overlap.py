"""Exact behavioral equivalence of the faster contiguous-evidence matcher."""
import random
import unicodedata
import unittest

from legalir2.duo_evidence import overlap, tokens


def reference(query, text):
    q, body = tokens(query), tokens(text)
    if not q or not body:
        return 0, 0.
    coverage = len(set(q) & set(body)) / len(set(q))
    for size in range(min(len(q), 24), 1, -1):
        phrases = {tuple(body[i:i + size]) for i in range(len(body) - size + 1)}
        if any(tuple(q[i:i + size]) in phrases for i in range(len(q) - size + 1)):
            return size, coverage
    return 0, coverage


class TestOverlap(unittest.TestCase):
    def test_repeated_words_unicode_and_boundaries(self):
        for query, body in [('', 'abc'), ('!!!', 'abc'), ('one', 'one'),
                            ('a b', 'b a'), ('a b c', 'a b x b c'),
                            ('a b a b a', 'a b a x b a b a'),
                            ('x ' * 40, 'x ' * 60),
                            ('THUẾ miễn 03-1/MGTH', unicodedata.normalize('NFD', 'thuế miễn 03-1/MGTH'))]:
            self.assertEqual(overlap(query, body), reference(query, body))

    def test_randomized_against_original_algorithm(self):
        rng = random.Random(2026)
        vocabulary = ['thuế', 'miễn', 'và', 'a', 'b', 'điều', '03', 'MGTH', 'ngoại']
        for _ in range(600):
            query = rng.choices(vocabulary, k=rng.randrange(1, 60))
            body = rng.choices(vocabulary, k=rng.randrange(1, 250))
            if rng.random() < .5:
                start = rng.randrange(len(body))
                body[start:start] = query[:rng.randrange(1, len(query) + 1)]
            q, text = ' '.join(query), ' '.join(body)
            self.assertEqual(overlap(q, text), reference(q, text))


if __name__ == '__main__':
    unittest.main()
