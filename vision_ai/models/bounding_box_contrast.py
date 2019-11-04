import random

import torch


def batch_loss(
    pred_xh,
    embeddings_xh,
    bbox_coords,
    frac_compare
):
    assert len(pred_xh) == len(embeddings_xh) == len(bbox_coords)

    sum_loss = n = 0.0
    for n, args in enumerate(zip(pred_xh, embeddings_xh, bbox_coords), 1):
        sum_loss += _single_bounding_box_contrastive_loss(
            *args, frac_compare
        )
    return sum_loss / n


# === PRIVATE ===


def _single_bounding_box_contrastive_loss(
    pred_xh,
    embeddings_xh,
    bbox_coords,
    frac_compare
):
    if not bbox_coords:
        return 0
    elif len(bbox_coords) == 1:
        return _bounding_box_similarity_loss(
            pred_xh,
            embeddings_xh,
            bbox_coords[0],
            frac_compare
        )
    else:
        i, j = _obtain_two_rand_diff_bbox_coords(bbox_coords)
        return _bounding_box_similarity_loss(
            pred_xh,
            embeddings_xh,
            bbox_coords[i],
            frac_compare
        ) + _bounding_box_difference_loss(
            pred_xh,
            embeddings_xh,
            bbox_coords[i],
            bbox_coords[j],
            frac_compare
        )


def _obtain_two_rand_diff_bbox_coords(bbox_coords):
    i = random.randint(0, len(bbox_coords) - 1)
    j = i
    while j == i:
        j = random.randint(0, len(bbox_coords) - 1)
    return i, j


def _bounding_box_similarity_loss(
    pred_xh,
    embeddings_xh,
    bbox_coord,
    frac_compare,
    eps = 1e-12
):
    selection = pred_xh[:, :, 1] > pred_xh[:, :, 0]
    mask = torch.zeros_like(selection)
    x, y, w, h = bbox_coord
    mask[x:x+w, y:y+h] = 1
    correct_selection = selection & mask

    part1 = torch.rand(correct_selection.size()) < frac_compare
    part2 = torch.rand(correct_selection.size()) < frac_compare

    select_part1 = correct_selection & part1
    select_part2 = correct_selection & part2

    emb_part1 = embeddings_xh[select_part1].mean(dim=0)
    emb_part2 = embeddings_xh[select_part2].mean(dim=0)
    # we wish to maximize their average cosine similarity
    return -emb_part1.dot(emb_part2)/(emb_part1.norm()*emb_part2.norm() + eps)


def _bounding_box_difference_loss(
    pred_xh,
    embeddings_xh,
    bbox_coord1,
    bbox_coord2,
    frac_compare,
    eps = 1e-12
):
    selection = pred_xh[:, :, 1] > pred_xh[:, :, 0]
    mask1 = torch.zeros_like(selection)
    mask2 = torch.zeros_like(selection)
    x1, y1, w1, h1 = bbox_coord1
    x2, y2, w2, h2 = bbox_coord2
    mask1[x1:x1+w1, y1:y1+h1] = 1
    mask2[x2:x2+w2, y2:y2+h2] = 1
    correct_selection1 = selection & mask1
    correct_selection2 = selection & mask2

    part1 = torch.rand(correct_selection1.size()) < frac_compare
    part2 = torch.rand(correct_selection2.size()) < frac_compare

    select_part1 = correct_selection1 & part1
    select_part2 = correct_selection2 & part2

    emb_part1 = embeddings_xh[select_part1].mean(dim=0)
    emb_part2 = embeddings_xh[select_part2].mean(dim=0)
    # we wish to minimize their average cosine similarity
    return emb_part1.dot(emb_part2)/(emb_part1.norm()*emb_part2.norm() + eps)
