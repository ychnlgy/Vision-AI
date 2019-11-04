import random

import numpy
import torch


def batch_loss(
    pred_xh,
    embeddings_xh,
    bbox_coords,
    mean=True
):
    """Contrastive loss for learning to differentiate between bounding boxes.
    
    Parameters :
    ============
    pred_xh : torch.FloatTensor of size (N, 2, W, H). Pixel-wise bounding box
        prediction by the model, using its final layer on the pixel embeddings.
    embeddings_xh : embeddings prior to the model's final layer.
    bbox_coords : list of list of (x, y, w, h) of bounding boxes. Corresponds
        to the indices of pred_xh and embeddings_xh.
    mean : bool indicator of whether to use mean of batch or not.

    Output :
    ========
    loss : single tensor value that can backpropogate gradient in respect to loss
    """
    assert len(pred_xh) == len(embeddings_xh) == len(bbox_coords)

    sum_loss = n = 0.0
    for n, args in enumerate(zip(pred_xh, embeddings_xh, bbox_coords), 1):
        sum_loss += _single_bounding_box_contrastive_loss(*args)
    return sum_loss / n**int(mean)


# === PRIVATE ===


def _single_bounding_box_contrastive_loss(
    pred_xh,
    embeddings_xh,
    bbox_coords
):
    if not bbox_coords:
        return 0
    elif len(bbox_coords) == 1:
        return _bounding_box_similarity_loss(
            pred_xh,
            embeddings_xh,
            bbox_coords[0]
        )
    else:
        i, j = _obtain_two_rand_diff_bbox_coords(bbox_coords)
        sim_loss = _bounding_box_similarity_loss(
            pred_xh,
            embeddings_xh,
            bbox_coords[i]
        )
        dif_loss = _bounding_box_difference_loss(
            pred_xh,
            embeddings_xh,
            bbox_coords[i],
            bbox_coords[j]
        )
        return sim_loss + dif_loss


def _obtain_two_rand_diff_bbox_coords(bbox_coords):
    i = random.randint(0, len(bbox_coords) - 1)
    j = i
    while j == i:
        j = random.randint(0, len(bbox_coords) - 1)
    return i, j


def _extract_avg_embedding_select(emb, select):
    d, w, h = emb.size()
    select = select.unsqueeze(0).repeat(d, 1, 1)
    data = emb[select].view(d, -1).T
    numpy.random.shuffle(data)
    return data


def _extract_avg_embedding_selections(emb, select1, select2):
    emb1 = _extract_avg_embedding_select(emb, select1)
    emb2 = _extract_avg_embedding_select(emb, select2)
    n = min(len(emb1), len(emb2))
    assert n > 0
    return emb1[:n], emb2[:n]


def cosine_sim(m1, m2, eps=1e-12):
    return ((m1*m2).sum(dim=1)/(m1.norm(dim=1)*m2.norm(dim=1) + eps)).mean()


def _bounding_box_similarity_loss(
    pred_xh,
    embeddings_xh,
    bbox_coord
):
    selection = pred_xh[1] > pred_xh[0]
    mask = torch.zeros_like(selection)
    x, y, w, h = bbox_coord
    mask[x:x+w, y:y+h] = 1
    correct_selection = selection & mask

    emb_part1, emb_part2 = _extract_avg_embedding_selections(
        embeddings_xh, correct_selection, correct_selection
    )

    # we wish to maximize their average cosine similarity
    return -cosine_sim(emb_part1, emb_part2)


def _bounding_box_difference_loss(
    pred_xh,
    embeddings_xh,
    bbox_coord1,
    bbox_coord2
):
    selection = pred_xh[1] > pred_xh[0]
    mask1 = torch.zeros_like(selection)
    mask2 = torch.zeros_like(selection)
    x1, y1, w1, h1 = bbox_coord1
    x2, y2, w2, h2 = bbox_coord2
    mask1[x1:x1+w1, y1:y1+h1] = 1
    mask2[x2:x2+w2, y2:y2+h2] = 1
    select_part1 = selection & mask1
    select_part2 = selection & mask2

    emb_part1, emb_part2 = _extract_avg_embedding_selections(
        embeddings_xh, select_part1, select_part2
    )

    # we wish to minimize their average cosine similarity
    return cosine_sim(emb_part1, emb_part2)
