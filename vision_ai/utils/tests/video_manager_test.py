import os

import tqdm
import torch

import vision_ai


DIR = os.path.dirname(__file__)
DATA = os.path.join(DIR, "sample.mp4")
REPEATS = 3


def check0(data):
    return (data == 0).all()


def check1(data):
    return (
        (data == 0).long().sum() == 11 and
        data[2, 1, 1] == 133
    )


def check2(data):
    return (
        (data == 0).long().sum() == 10,
        data[2, 1, 1] == 133,
        data[1, 0, 0] == 255
    )


def check3(data):
    return (
        (data == 0).long().sum() == 9,
        data[2, 1, 1] == 133,
        data[1, 0, 0] == 255,
        data[0, 1, 0] == 80
    )


def check4(data):
    return (
        (data == 0).long().sum() == 6,
        data[2, 1, 1] == 133,
        data[1, 0, 0] == 255,
        data[0, 1, 0] == 80,
        (data[:, 0, 1] == 120).all()
    )


def test_parse():
    with vision_ai.utils.VideoManager(fps=1) as manager:
        for i in range(REPEATS):
            with manager.parse(DATA) as parser:
                data = parser.read()
                assert data.shape == (5, 3, 2, 2)
                assert check0(data[0])
                assert check1(data[1])
                assert check2(data[2])
                assert check3(data[3])
                assert check4(data[4])


def test_skip_2_frames():
    with vision_ai.utils.VideoManager(fps=2) as manager:
        for i in range(REPEATS):
            with manager.parse(DATA) as parser:
                data = parser.read()
                assert data.shape == (3, 3, 2, 2)
                assert check0(data[0])
                assert check2(data[1])
                assert check4(data[2])


def test_skip_3_frames():
    with vision_ai.utils.VideoManager(fps=3) as manager:
        for i in range(REPEATS):
            with manager.parse(DATA) as parser:
                data = parser.read()
                assert data.shape == (2, 3, 2, 2)
                assert check0(data[0])
                assert check3(data[1])
