from copy import deepcopy
import unittest
from legalir2.passage_order import reorder, token_hash, changed_edges, rescore
from tests2.test_duo_agreement import fixture


class Codec:
    separator = [500]

    def encode(self, text):
        return list(map(ord, text))


def trace():
    codec = Codec()
    packed = [{'rowid': 1, 'text': 'unrelated', 'tokens': 9},
              {'rowid': 2, 'text': 'relevant', 'tokens': 8}]
    ids = codec.encode('law\n\nunrelated')+codec.separator+codec.encode('relevant')
    return {'label': 'law', 'packed': packed, 'input_ids': ids,
            'sha256': token_hash(ids), 'budget': 500}


class TestPassageOrder(unittest.TestCase):
    def test_reorders_without_dropping_tokens_or_mutating_input(self):
        original = trace(); saved = deepcopy(original)
        result = reorder(original, [-9., -4.], Codec())
        self.assertEqual([p['rowid'] for p in result['packed']], [2, 1])
        self.assertCountEqual(result['input_ids'], original['input_ids'])
        self.assertEqual(original, saved)
        self.assertEqual(result['ordering']['indices'], [1, 0])

    def test_stable_ties_and_already_ordered_are_byte_identical(self):
        for values in ([1., 1.], [2., 1.]):
            self.assertEqual(reorder(trace(), values, Codec())['input_ids'], trace()['input_ids'])

    def test_corruption_and_score_mismatch_fail(self):
        for values in ([1.], [1., float('nan')]):
            with self.assertRaises(ValueError):
                reorder(trace(), values, Codec())
        bad = trace(); bad['packed'][0]['text'] = 'different'
        with self.assertRaises(ValueError):
            reorder(bad, [1., 2.], Codec())
        bad = trace(); bad['input_ids'][0] += 1
        with self.assertRaises(ValueError):
            reorder(bad, [1., 2.], Codec())

    def test_only_incident_edges_are_rescored(self):
        before = dict(a=[1], b=[2], c=[3], d=[4]); after = deepcopy(before); after['b'] = [5]
        self.assertEqual(changed_edges(list('abcd'), before, after), [(0, 1), (1, 2), (1, 3)])
        self.assertEqual(changed_edges(list('abcd'), before, before), [])
        with self.assertRaises(ValueError):
            changed_edges(list('abc'), before, after)

    def test_zero_changed_edges_never_load_or_call_model(self):
        row = fixture()
        result = rescore(None, None, row, {}, [], 4)
        self.assertEqual(result.tolist(), row['probabilities'])

    def test_both_orientations_are_averaged_and_other_edges_preserved(self):
        import torch
        from types import SimpleNamespace
        row = fixture()

        class PairCodec:
            labels = [0, 1]

            def pair(self, query, first, second):
                return [first[0][0], second[0][0]]

        class Tokenizer:
            def pad(self, value, **kwargs):
                return {'input_ids': torch.tensor(value['input_ids'])}

        def model(input_ids, logits_to_keep):
            self.assertEqual(input_ids.tolist(), [[0, 1], [1, 0]])
            # P(A beats B)=.8; P(B beats A)=.3 -> symmetric preference .75.
            return SimpleNamespace(logits=torch.log(torch.tensor([[[.8, .2]], [[.3, .7]]])))

        codec = PairCodec(); codec.tokenizer = Tokenizer()
        local = SimpleNamespace(torch=torch, model=model, device='cpu')
        encoded = {doc: [i] for i, doc in enumerate(row['docs'])}
        result = rescore(local, codec, row, encoded, [(0, 1)], 4)
        self.assertAlmostEqual(result[0, 1], .75, places=6)
        self.assertAlmostEqual(result[1, 0], .25, places=6)
        for i in range(6):
            for j in range(6):
                if (i, j) not in ((0, 1), (1, 0)):
                    self.assertEqual(result[i, j], row['probabilities'][i][j])


if __name__ == '__main__':
    unittest.main()
