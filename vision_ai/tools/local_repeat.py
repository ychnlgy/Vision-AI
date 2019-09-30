def local_repeat(X, factor):
    """Given X (N, C, W, H), return X' (N, C, factor*W, factor*H).

    Repeat each channel vector <factor> times along the W and H axis.
    """
    N, C, W, H = X.size()
    if type(factor) is int:
        factor = (factor, factor)
    factor_x, factor_y = factor
    return X.repeat(
        1, factor_x*factor_y, 1, 1
    ).view(
        N, factor_x, factor_y, C, W, H
    ).permute(
        0, 3, 4, 1, 5, 2
    ).contiguous().view(N, C, factor_x*W, factor_y*H)
