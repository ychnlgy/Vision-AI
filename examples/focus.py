import numpy
import scipy.interpolate

def extract_object(depths, resolution=100, min_dist=100):
    low, hgh = extract_object_depths(depths, resolution, min_dist)
    depths = depths.copy()
    depths[(depths < low) | (depths > hgh)] = 0
    return depths

def extract_object_depths(depths, resolution=100, min_dist=100):
    flat = depths.reshape(-1)
    hist, xs = numpy.histogram(flat, bins=resolution)
    xs = xs[1:]
    hist[xs <  min_dist] = 0
    rbf_interp = scipy.interpolate.Rbf(xs, hist, epsilon=resolution, smooth=1)
    interp = rbf_interp(xs)
    low, hgh = extract_peak_gaussian(interp, xs)
    return xs[low], xs[hgh]

# === PRIVATE ===

def extract_peak_gaussian(interp, xs):
    p = numpy.argmax(interp)
    low = find_valley(interp, p, xs, -1)
    hgh = find_valley(interp, p, xs, +1)
    return (low, hgh)

def find_valley(interp, p, xs, step):
    out = p + step
    while 0 < out < len(xs):
        if interp[p] < interp[out]:
            return p
        else:
            p = out
            out += step
    return p
