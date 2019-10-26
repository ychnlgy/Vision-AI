import os

import tqdm

import vision_ai


DIR = os.path.dirname(__file__)
DATA = os.path.join(DIR, "sample.mp4")


def test_parse():
    with vision_ai.utils.VideoManager(fps=1) as manager:
        with manager.parse(DATA) as parser:
            data = parser.read()
            assert data.shape == (4, 4, 4, 3)
