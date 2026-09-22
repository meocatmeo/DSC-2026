"""Fourth-voter cache reuse cannot cross input or model identities."""
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from legalir2.fourth_voter import main, passages, pool, validate_result
from legalir2.h100 import digest
from legalir2.store import file_hash, read_json, write_json


class TestFourthVoter(unittest.TestCase):
    def setUp(self):
        self.row = {'id': 'q', 'rankings': {'v9': list('abcdef'), 'duo': list('fghija'), 'fulltext_duo': list('klmnoa')}}

    def test_full_union_and_cache_identity(self):
        docs = pool(self.row)
        self.assertEqual(docs, list('abcdefghijklmno'))
        result = dict(id='q', input=digest(self.row), run='model1', docs=docs,
                      scores=list(range(len(docs), 0, -1)), ordering=docs,
                      passage_sha256=['hash']*len(docs))
        validate_result(result, self.row, 'model1')
        with self.assertRaises(ValueError):
            validate_result(result, self.row, 'model2')
        with self.assertRaises(ValueError):
            validate_result(dict(result, ordering=docs[::-1]), self.row, 'model1')

    def test_passages_are_document_local_and_gold_free(self):
        class Evidence:
            def search(self, query, doc, limit):
                self.request = query, doc, limit
                return [(1, doc, 0, 'label', 'body', -1)]
        evidence = Evidence()
        self.assertEqual(passages(evidence, 'question', 'doc'), ['label\nbody'])
        self.assertEqual(evidence.request, ('question', 'doc', 2))

    def test_labeled_scoring_resume_without_loading_gpu_and_stale_cache_rejection(self):
        class Evidence:
            def __init__(self, *args):
                self.store = SimpleNamespace(doc_position=dict.fromkeys('abcde'))
            def search(self, query, doc, limit):
                return [(1, doc, 0, doc, 'body', -1.)]
            def close(self):
                pass

        class Model:
            def parameters(self):
                return [SimpleNamespace(is_floating_point=lambda: True, dtype='bf16')]

        class Reranker:
            def __init__(self, *args, **kwargs):
                self.parameters = 7
                self.local = SimpleNamespace(model=Model(), torch=SimpleNamespace(bfloat16='bf16'), close=lambda: None)
            def score(self, query, texts, batch, length):
                return list(range(len(texts), 0, -1))

        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'store.sqlite').write_bytes(b'fixture')
            (root/'fts.sqlite').write_bytes(b'fixture index')
            row = dict(id='q', question='question', gold=['a'], rankings={n: list('abcde') for n in ('v9', 'duo', 'fulltext_duo')})
            write_json(root/'rankings.json', {'rows': [row], 'inputs': 'fixture'})
            write_json(root/'questions.json', {'q': {'question': 'question', 'answer': ['a']}})
            write_json(root/'audit.json', {'components': {'baseline': 3}, 'total': 3})
            argv = ['fourth-voter', '--rankings', str(root/'rankings.json'), '--questions', str(root/'questions.json'),
                    '--base-audit', str(root/'audit.json'), '--model', str(root/'model'), '--index', str(root),
                    '--evidence-index', str(root/'fts.sqlite'), '--out', str(root/'out')]
            provenance = {'repository': 'namdp-ptit/ViRanker', 'commit': 'fixture',
                          'files': {'config.json': {'sha256': 'fixture'}}}
            with patch('sys.argv', argv), patch('legalir2.fourth_voter.verify', return_value=provenance), \
                 patch('legalir2.fourth_voter.index_metadata', return_value={'store_sha256': file_hash(root/'store.sqlite')}), \
                 patch('legalir2.fourth_voter.Evidence', Evidence), patch('legalir2.rerank.Reranker', side_effect=Reranker) as model:
                main()
                self.assertEqual(model.call_count, 1)
                self.assertEqual(read_json(root/'out/audit.json')['total'], 10)
                main()
                self.assertEqual(model.call_count, 1, 'Complete resume must not load GPU weights')
                cached = read_json(root/'out/scores/part-00000.json')
                write_json(root/'out/scores/part-00000.json', dict(cached, input='wrong'))
                with self.assertRaises(ValueError):
                    main()


if __name__ == '__main__':
    unittest.main()
