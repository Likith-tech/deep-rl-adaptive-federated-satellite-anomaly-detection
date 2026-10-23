"""Run isolated tests and persist their exact output without touching artifacts."""
from pathlib import Path
import json
import os
import subprocess
import sys
import uuid

root = Path(__file__).resolve().parent
run_id = uuid.uuid4().hex
temporary = root / '.test_runs' / run_id
temporary.parent.mkdir(exist_ok=True)
(root / 'logs').mkdir(exist_ok=True)
command = [sys.executable, '-B', '-m', 'pytest', '-q', str(root / 'tests'),
           '--basetemp', str(temporary), '-p', 'no:cacheprovider', '--tb=short',
           '--junitxml', str(root / 'logs' / f'tests-{run_id}.xml')]
env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
result = subprocess.run(command, cwd=root, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
(root / 'logs' / f'tests-{run_id}.log').write_text(result.stdout, encoding='utf8')
(root / 'logs' / f'tests-{run_id}.json').write_text(json.dumps({'command': command, 'exit_code': result.returncode}, indent=2), encoding='utf8')
print(result.stdout)
print('Test record: ' + str(root / 'logs' / f'tests-{run_id}.json'))
sys.exit(result.returncode)
