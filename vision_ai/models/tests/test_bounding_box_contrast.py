import numpy
import torch

import vision_ai


def test_no_boxes():
    bbox_coords = [
        [],
        []
    ]

    pred_xh = torch.zeros(2, 2, 4, 4)
    embeddings_xh = torch.zeros(2, 3, 4, 4)
    assert vision_ai.models.bounding_box_contrast.batch_loss(
        pred_xh,
        embeddings_xh,
        bbox_coords
    ) == 0.0


def test_one_box_same():
    bbox_coords = [
        [(0, 0, 2, 2)],
        [(0, 0, 2, 2)]
    ]

    pred_xh = torch.zeros(2, 2, 4, 4)
    embeddings_xh = torch.zeros(2, 3, 4, 4)
    for i, bboxes in enumerate(bbox_coords):
        x, y, w, h = bboxes[0]
        pred_xh[i, 1, x:x+w, y:y+h] = 1
        embeddings_xh[i, :, x:x+w, y:y+h] = torch.Tensor(
            [1, 2, 3]
        ).view(3, 1, 1)

    eps = 1e-6
    assert (vision_ai.models.bounding_box_contrast.batch_loss(
        pred_xh,
        embeddings_xh,
        bbox_coords,
    ) - (-1.0)).abs() < eps


def test_one_box_diff():
    numpy.random.seed(5)
    bbox_coords = [
        [(0, 0, 2, 2)],
        [(0, 0, 2, 2)]
    ]

    pred_xh = torch.zeros(2, 2, 4, 4)
    embeddings_xh = torch.zeros(2, 3, 4, 4)
    for i, bboxes in enumerate(bbox_coords):
        x, y, w, h = bboxes[0]
        pred_xh[i, 1, x:x+w, y:y+h] = 1
        for dx in range(w):
            for dy in range(h):
                s = int((dx + dy) > 1)
                embeddings_xh[i, :, x+dx, y+dy] = torch.Tensor(
                    [dx*(1-s), dy*(1-s), s]
                )

    assert vision_ai.models.bounding_box_contrast.batch_loss(
        pred_xh,
        embeddings_xh,
        bbox_coords,
    ) < 0.5


def test_two_boxes_same():
    numpy.random.seed(5)
    bbox_coords = [
        [(0, 0, 2, 2), (2, 0, 2, 2)]
    ]
    pred_xh = torch.zeros(1, 2, 4, 4)
    pred_xh[0, 1] = 1
    embeddings_xh = torch.ones(1, 3, 4, 4)
    assert (vision_ai.models.bounding_box_contrast.batch_loss(
        pred_xh,
        embeddings_xh,
        bbox_coords
    ) - (-2.0)) < 1e-4
