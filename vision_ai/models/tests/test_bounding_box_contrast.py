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
        bbox_coords,
        frac_compare=0.5
    ) == 0.0
