"""Corpus citation ownership and training-only candidate proposals."""
import unittest

from legalir2.related_documents import Relations, SCHEMA, numbers


class TestRelations(unittest.TestCase):
    def setUp(self):
        self.graph = {'schema': SCHEMA, 'documents': list('abcdef'),
            'numbers': {'3/2014/TT-BTTTT': 'f'},
            'outgoing': {'a': ['f']}, 'incoming': {'f': ['a']}}
        self.training = {str(i): {'question': f'train {i}', 'answer': ['a', 'e']} for i in range(4)}

    def test_document_numbers_are_canonical_and_bounded(self):
        self.assertEqual(numbers('Thông tư số 03 / 2014 / TT – BTTTT, quy định'), {'3/2014/TT-BTTTT'})
        self.assertEqual(numbers('day 03/2014 and not-a-document'), set())

    def test_proposals_use_relationships_and_never_drop_existing(self):
        model = Relations(self.graph, self.training)
        current = list('abcd')
        proposal = model.propose(['a'], current, mode='training')
        self.assertEqual([p['doc'] for p in proposal], ['e'])
        self.assertEqual(current, list('abcd'))
        self.assertEqual(proposal[0]['reasons'][0]['support'], 4)

    def test_local_citation_outweighs_global_reference_without_labels(self):
        model = Relations(self.graph, {})
        plain = model.propose(['a'], ['a'], mode='citations')
        local = model.propose(['a'], ['a'], {'a': {'packed': [{'text': '03/2014/TT-BTTTT'}]}}, mode='citations')
        self.assertEqual(local[0]['doc'], 'f')
        self.assertGreater(local[0]['score'], plain[0]['score'])
        self.assertEqual(model.propose(['a'], ['a', 'f'], mode='citations'), [])

    def test_unknown_training_documents_and_anchors_fail(self):
        with self.assertRaises(ValueError):
            Relations(self.graph, {'q': {'answer': ['unknown']}})
        with self.assertRaises(ValueError):
            Relations(self.graph, {}).propose(['unknown'], [])


if __name__ == '__main__':
    unittest.main()
