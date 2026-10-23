"""Default is read-only preflight. No experiment runs without --enable-training."""
from pathlib import Path
import sys

root = Path(__file__).resolve().parent
sys.path.insert(0, str(root / 'src'))
sys.path.insert(0, str(root / 'detector_training'))
import orbitshield
if Path(orbitshield.__file__).resolve().parent != root / 'src/orbitshield':
    raise RuntimeError('Wrong orbitshield package; refuse legacy package collision')
from orbitshield_detector.cli import main

if __name__ == '__main__':
    main()
