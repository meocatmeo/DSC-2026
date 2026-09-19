"""Private input isolation, cached two-model inference, and complete submission ZIPs."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import zipfile

import numpy as np

from legalir2 import private_infer as private
from legalir2.store import file_hash, read_json, write_json


class TestPrivateInference(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.args = SimpleNamespace(questions=self.root/'private.json', out=self.root/'private-out',
            duo_run=self.root/'trained-duo', index=self.root/'index', infer_batch=8, shard_size=2)
        self.docs = list('abcdef')
        self.weights = [1.] + [0.] * 19
        self.signature = {'known_docs': self.docs, 'runtime': {'batch':8, 'attention':'sdpa'},
            'weights':self.weights, 'memory':{}, 'max_length':2048, 'attention':'flash_attention_2'}
        write_json(self.args.questions, {'p1':{'question':'first', 'answer':None},
                                        'p2':{'question':'second', 'answer':None},
                                        'p3':{'question':'third', 'answer':None}})
        self.records = private.question_records(self.args.questions)
        self.marker = {'questions':private.h100.digest(self.records), 'strength':.25}

    def row(self, qid, record):
        features = [[float(6-i)]+[0.]*19 for i in range(6)]
        return {'id':qid, 'question':record['question'], 'gold':[], 'docs':self.docs,
                'features1':features, 'features2':features}

    def collect(self):
        model = Mock()
        def score(model, qid, record, mode):
            self.assertEqual(record['answer'], [])
            return self.row(qid, record)
        with patch.object(private.h100, 'pipeline', return_value=model) as load, \
             patch.object(private.h100, 'score_row', side_effect=score) as scoring:
            result = private.collect_v9(self.args, self.records, self.signature, self.marker)
        return result, model, load, scoring

    def test_null_answers_are_normalized_only_in_scoring_copy(self):
        before = file_hash(self.args.questions)
        self.collect()
        self.assertEqual(before, file_hash(self.args.questions))
        self.assertIsNone(read_json(self.args.questions)['p1']['answer'])
        write_json(self.args.questions, {'bad':{'question':'q', 'answer':['a']}})
        with self.assertRaises(ValueError):
            private.question_records(self.args.questions)

    def test_v9_commits_partial_last_shard_and_resumes_without_gpu(self):
        rows, model, load, scoring = self.collect()
        self.assertEqual(scoring.call_count, 3)
        model.close.assert_called_once()
        self.assertEqual(len(list((self.args.out/'cache/v9').glob('part-*.json'))), 2)
        again, _, load, scoring = self.collect()
        self.assertEqual(rows, again)
        load.assert_not_called()
        scoring.assert_not_called()

    def test_new_questions_or_corrupt_shards_cannot_reuse_cache(self):
        self.collect()
        with self.assertRaises(ValueError):
            private.collect_v9(self.args, self.records, self.signature, dict(self.marker, questions='different'))
        path = self.args.out/'cache/v9/part-00000.json'
        rows = read_json(path)
        rows[0]['question'] = 'changed text'
        write_json(path, rows)
        with self.assertRaises(ValueError):
            self.collect()

    def test_duo_reuses_private_v9_and_saved_attention_and_resumes(self):
        base, _, _, _ = self.collect()
        local = Mock()
        evidence = Mock()
        memory = Mock()
        memory.search.return_value = ({}, [])
        def load_model(*a, **kw):
            self.assertEqual(os.environ['LEGALIR_CAUSAL_ATTN'], 'flash_attention_2')
            return local
        with patch.object(private, 'Local', side_effect=load_model) as load, \
             patch.object(private.duo, 'Evidence', return_value=evidence), \
             patch.object(private.duo, 'PairTokenizer'), \
             patch.object(private, 'QuestionMemory', return_value=memory), \
             patch.object(private.duo, 'compare_documents', return_value=np.full((6,6), .5)) as compare, \
             patch.object(private.h100, 'pipeline') as pipeline:
            rows = private.collect_duo(self.args, self.records, self.signature, self.marker)
            self.assertEqual(compare.call_count, 3)
            self.assertEqual([r['baseline'] for r in rows], [private.h100.order(r,self.weights,1) for r in base])
            pipeline.assert_not_called()
            load.reset_mock()
            again = private.collect_duo(self.args, self.records, self.signature, self.marker)
            self.assertEqual(rows, again)
            load.assert_not_called()
        local.close.assert_called_once()
        evidence.close.assert_called_once()

    def test_submission_has_all_private_ids_and_only_submission_json(self):
        result = {q:{'answer':self.docs[:5]} for q in self.records}
        private.write_submission(self.args,result,'v9',set(self.docs),self.marker)
        with zipfile.ZipFile(self.args.out/'submission-private-v9.zip') as archive:
            self.assertEqual(archive.namelist(), ['submission.json'])
            self.assertEqual(json.loads(archive.read('submission.json')),result)
        for invalid in [dict(list(result.items())[:2]),
                        dict(result,p1={'answer':['a']*5}),
                        dict(result,p1={'answer':list('abcdz')})]:
            with self.assertRaises(ValueError):
                private.write_submission(self.args,invalid,'duo',set(self.docs),self.marker)

    def test_fulltext_policy_reaches_comparator_and_records_visible_evidence(self):
        self.collect()
        self.args.duo_policy = 'fulltext-v2'
        self.args.evidence_index = self.root/'evidence.sqlite'
        evidence = Mock()
        evidence.expand.side_effect = lambda query, docs, base, known: (docs, base)
        evidence.encode.return_value = {'a': [[101, 102]]}
        evidence.trace = {'schema': 'duo-fulltext-v2', 'additions': [], 'documents': {}}
        memory = Mock()
        memory.search.return_value = ({}, [])
        with patch.object(private, 'Local'), patch.object(private.duo, 'PairTokenizer'), \
             patch('legalir2.duo_evidence.Evidence', return_value=evidence), \
             patch.object(private.duo, 'Evidence') as legacy, \
             patch.object(private, 'QuestionMemory', return_value=memory), \
             patch.object(private.duo, 'compare_documents', return_value=np.full((6,6), .5)) as compare:
            rows = private.collect_duo(self.args, self.records, self.signature, self.marker)
        legacy.assert_not_called()
        self.assertEqual(evidence.expand.call_count, 3)
        self.assertEqual(compare.call_args.args[3], evidence.encode.return_value)
        self.assertEqual(rows[0]['evidence']['schema'], 'duo-fulltext-v2')
        evidence.close.assert_called_once()

    def test_asset_hash_mismatch_is_fatal(self):
        directory = Path(private.__file__).parent
        files = {'legalir2/'+name:file_hash(directory/name) for name in private.SOURCE_MODULES}
        names = ['store.sqlite','lexical/unigram.npz','lexical/bigram.npz',
                 'dense/vectors.npy','dense/chunk_ids.npy','dense/manifest.json']
        self.args.models = self.root/'models'
        self.args.baseline = self.root/'baseline'
        for name in names:
            p = self.args.index/name
            p.parent.mkdir(parents=True,exist_ok=True)
            p.write_bytes(b'fixture')
            files['artifacts-ranking-h100/'+name] = file_hash(p)
        for name in private.h100.MODELS:
            p = self.args.models/name/'config.json'
            write_json(p,{})
            files['models2/'+name+'/config.json'] = file_hash(p)
        signature = {'source_inputs':{'files':files},'code':{}}
        private.verify_assets(self.args,signature)
        (self.args.index/'store.sqlite').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'asset missing/changed'):
            private.verify_assets(self.args,signature)


class TestPrivateShell(unittest.TestCase):
    def test_setup_stops_on_flash_failure_and_reaches_inference_on_success(self):
        root = Path(__file__).resolve().parents[1]
        bash = Path('C:/Program Files/Git/bin/bash.exe') if os.name=='nt' else Path('/bin/bash')
        if not bash.exists():
            self.skipTest('Bash unavailable')
        import hashlib
        for fail in ('', 'flash'):
            with tempfile.TemporaryDirectory() as directory:
                fixture = Path(directory)
                def put(name, text):
                    p = fixture/name
                    p.parent.mkdir(parents=True, exist_ok=True)
                    p.write_bytes(text.encode())
                    p.chmod(0o755)
                put('scripts/gpu/gpu-private-setup.sh', (root/'scripts/gpu/gpu-private-setup.sh').read_text())
                put('config/requirements-private-h100.txt', (root/'config/requirements-private-h100.txt').read_text())
                put('private-files.sha256', hashlib.sha256(b'fixture').hexdigest()+'  checked.txt\n')
                put('checked.txt', 'fixture')
                for command in ('rclone','flock','setsid','nvidia-smi','nvcc','g++'):
                    put('bin/'+command, '#!/usr/bin/env bash\nexit 0\n')
                put('.venv-h100/bin/python', '''#!/usr/bin/env bash
printf '%s\\n' "$*" >> "$ROOT/calls"
if [[ "$*" == *flash-attn==2.8.3* ]] && [ "${FAIL_SETUP:-}" = flash ]; then exit 19; fi
exit 0
''')
                put('scripts/gpu/gpu-private-h100.sh', '#!/usr/bin/env bash\nprintf "inference %s\\n" "$1" >> "$ROOT/calls"\n')
                put('scripts/gpu/gpu-duo-runtime.sh', '''step() { echo "$*"; }
stage_now() { echo "stage=$1"; }
ship_once() { :; }
start_worker() { :; }
stop_worker() { :; }
run_cmd() { if [ "$1" = timeout ]; then shift 4; fi; "$@"; }
finish_run() { local code=$?; trap - EXIT; exit "$code"; }
''')
                env = dict(os.environ, FIXTURE=fixture.as_posix(), FAIL_SETUP=fail)
                result = subprocess.run([str(bash), '-c',
                    'export PATH="$(cd "$FIXTURE/bin" && pwd):$PATH"; '
                    'export PYTHON_BIN="$FIXTURE/.venv-h100/bin/python"; '
                    'bash "$FIXTURE/scripts/gpu/gpu-private-setup.sh" all'],
                    env=env, capture_output=True, text=True)
                self.assertEqual(result.returncode, 19 if fail else 0, result.stdout+result.stderr)
                calls = (fixture/'calls').read_text()
                self.assertIn('torch==2.7.1+cu128', calls)
                self.assertIn('--no-build-isolation --no-deps flash-attn==2.8.3', calls)
                self.assertEqual('inference all' in calls, not fail)

    def test_all_reaches_submit_and_scoring_failure_stops_it(self):
        root = Path(__file__).resolve().parents[1]
        bash = Path('C:/Program Files/Git/bin/bash.exe') if os.name=='nt' else Path('/bin/bash')
        if not bash.exists():
            self.skipTest('Bash unavailable')
        for fail in ('', 'duo'):
            with tempfile.TemporaryDirectory() as directory:
                fixture = Path(directory)
                def put(name, text):
                    p = fixture/name
                    p.parent.mkdir(parents=True, exist_ok=True)
                    p.write_bytes(text.encode())
                    p.chmod(0o755)
                put('scripts/gpu/gpu-private-h100.sh',(root/'scripts/gpu/gpu-private-h100.sh').read_text())
                put('data/private-official.json','{}')
                for command in ('rclone','flock','setsid','nvidia-smi'):
                    put('bin/'+command,'#!/usr/bin/env bash\nexit 0\n')
                put('.venv-h100/bin/python','''#!/usr/bin/env bash
if [ "$2" = legalir2.private_restore ]; then exit 0; fi
printf '%s\\n' "$4" >> "$ROOT/calls"
if [ "$4" = "${FAIL_STAGE:-}" ]; then exit 17; fi
if [ "$4" = v9 ]; then touch "$OUT/submission-private-v9.zip"; fi
if [ "$4" = duo ]; then touch "$OUT/submission-private-duo.zip"; fi
''')
                put('scripts/gpu/gpu-duo-runtime.sh','''step() { echo "$*"; }
stage_now() { echo "stage=$1"; }
ship_once() { :; }
rclone_run() { :; }
start_worker() { :; }
run_cmd() { "$@"; }
finish_run() { local code=$?; trap - EXIT; exit "$code"; }
''')
                env = dict(os.environ,FIXTURE=fixture.as_posix(),FAIL_STAGE=fail)
                for name in ('PY','PRIVATE_QUESTIONS','PRIVATE_RUN_ID','SOURCE_RUN_ID','DUO_RUN_ID'):
                    env.pop(name,None)
                result = subprocess.run([str(bash),'-c',
                    'export PATH="$(cd "$FIXTURE/bin" && pwd):$PATH"; bash "$FIXTURE/scripts/gpu/gpu-private-h100.sh" all'],
                    env=env,capture_output=True,text=True)
                self.assertEqual(result.returncode,17 if fail else 0,result.stdout+result.stderr)
                expected = ['preflight','v9','duo'] + ([] if fail else ['submit'])
                self.assertEqual((fixture/'calls').read_text().splitlines(),expected)

    def test_parse_and_help(self):
        root = Path(__file__).resolve().parents[1]
        bash = Path('C:/Program Files/Git/bin/bash.exe') if os.name=='nt' else Path('/bin/bash')
        if not bash.exists():
            self.skipTest('Bash unavailable')
        for name in ('scripts/gpu/gpu-private-h100.sh', 'scripts/gpu/gpu-private-setup.sh'):
            for arguments in [('-n',), ()]:
                command = [str(bash), *arguments, str(root/name)]
                if not arguments:
                    command += ['--help']
                result = subprocess.run(command,capture_output=True,text=True)
                self.assertEqual(result.returncode,0,result.stderr)
                if not arguments:
                    self.assertIn('submit',result.stdout)
                    self.assertIn('no retraining',result.stdout.lower())


if __name__ == '__main__':
    unittest.main()
