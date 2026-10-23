"""Small logging setup used by command-line workflows."""
import logging
from pathlib import Path

def configure_logging(path: Path) -> logging.Logger:
    path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(filename=path, level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    return logging.getLogger("fedstar_clean")
