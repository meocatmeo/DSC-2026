"""Exercise the lazy training imports from the actual release, outside the repository."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT/path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


dependencies = module('release_dependencies', 'scripts/package/release_dependencies.py')
repair = module('repair_retrain_dependencies', 'scripts/gpu/repair_retrain_dependencies.py')


class TestRetrainDependencies(unittest.TestCase):
    def test_closure_includes_function_local_and_transitive_imports(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'legalir2').mkdir()
            (root/'legalir2/second.py').write_text('from .third import helper\n')
            (root/'legalir2/third.py').write_text('def helper(): pass\n')
            payload = {'legalir2/first.py': b'def train():\n    from .second import helper\n'}
            self.assertEqual(dependencies.complete_imports(payload, root, ['legalir2/first.py']),
                             ['legalir2/second.py', 'legalir2/third.py'])

    def test_packaged_train_reproduces_failure_then_reaches_cuda_guard(self):
        archive = ROOT/'releases/retrain/20260921-fulltext-retrain-4090-v1/fulltext-retrain.zip'
        source = ROOT/'releases/tools/20260921-retrain-training-deps-v1'
        if not archive.exists() or not source.exists():
            self.skipTest('Build the repair release to run the isolated archive regression')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            with zipfile.ZipFile(archive) as bundle:
                for name in bundle.namelist():
                    if name.endswith('.py') or name == 'private-files.sha256':
                        bundle.extract(name, root)
            before = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
                      for path in root.rglob('*') if path.is_file()}
            smoke = '''
import json, pathlib, sys, types
root = pathlib.Path(sys.argv[1]).resolve()
sys.path.insert(0, str(root))
# PEFT is unavailable on the local CPU environment. No PEFT function may run.
peft = types.ModuleType('peft')
def unavailable(*args, **kwargs):
    raise AssertionError('Training must stop before PEFT execution')
for name in ('LoraConfig', 'get_peft_model', 'get_peft_model_state_dict', 'set_peft_model_state_dict'):
    setattr(peft, name, unavailable)
sys.modules['peft'] = peft
import torch
torch.cuda.is_available = lambda: False
from legalir2 import duo
assert pathlib.Path(duo.__file__).resolve() == root/'legalir2/duo.py'
settings = dict(max_length=2048, epochs=2, batch=2, accumulate=8, learning_rate=2e-5)
out = root/'smoke-output'
out.mkdir(exist_ok=True)
(out/'inputs.json').write_text(json.dumps(settings))
try:
    duo.train(types.SimpleNamespace(out=out, **settings))
except (ModuleNotFoundError, ValueError) as error:
    print(type(error).__name__+': '+str(error))
else:
    raise AssertionError('Expected the CPU guard')
'''
            def run():
                result = subprocess.run([sys.executable, '-I', '-c', smoke, str(root)],
                                        cwd=root, text=True, capture_output=True, timeout=90)
                self.assertEqual(result.returncode, 0, result.stderr)
                return result.stdout
            self.assertIn("ModuleNotFoundError: No module named 'legalir2.train_ce'", run())
            repair.install(root, source)
            repair.install(root, source)
            self.assertIn('ValueError: Comparator training requires BF16 CUDA', run())
            for name, expected in before.items():
                self.assertEqual(hashlib.sha256((root/name).read_bytes()).hexdigest(), expected, name)
            (root/'legalir2/mine.py').write_text('user modification')
            with self.assertRaisesRegex(RuntimeError, 'Refusing to replace'):
                repair.install(root, source)
            self.assertEqual((root/'legalir2/mine.py').read_text(), 'user modification')


if __name__ == '__main__':
    unittest.main()
