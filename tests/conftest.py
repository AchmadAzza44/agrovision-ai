import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))


@pytest.fixture(scope="session")
def synthetic_dir(tmp_path_factory):
    import numpy as np
    from make_synthetic_dataset import CLASSES, make_leaf

    root = tmp_path_factory.mktemp("leaves")
    rng = np.random.default_rng(1)
    for label in CLASSES:
        (root / label).mkdir()
        for i in range(24):
            make_leaf(label, rng).save(root / label / f"{i}.png")
    return root
