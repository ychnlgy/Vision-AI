import sys
import pathlib

import vision_ai


WIDTH = 480
HEIGHT = 854


def collect_mp4s(root):
    return sorted(list(map(str, pathlib.Path(root).rglob("*.mp4"))))


def process_single_file(fpath, manager, skip_x, skip_y):
    with manager.parse(fpath) as slicer:
        data = slicer.read().numpy()
    assert data.shape[1:] == (3, WIDTH, HEIGHT)
    return data[:, :, ::skip_x, ::skip_y]


def process_all_files(root, skip_frames, skip_x, skip_y):
    with vision_ai.utils.VideoManager(skip_frames) as manager:
        for fpath in collect_mp4s(root):
            yield process_single_file(fpath, manager, skip_x, skip_y)


def main(root, skip_frames, skip_x, skip_y, savef):
    with vision_ai.utils.ChunkFile(savef, "wb") as sfile:
        for data in process_all_files(root, skip_frames, skip_x, skip_y):
            sys.stderr.write("Chunk size: %d x %d x %d x %d\n" % data.shape)
            sys.stderr.flush()
            sfile.save(data)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--root", required=True)
    parser.add_argument("--skip_frames", type=int, default=50)
    parser.add_argument("--skip_x", type=int, default=2)
    parser.add_argument("--skip_y", type=int, default=3)
    parser.add_argument("--savef", default="sliced_cholec80.pkl")
    
    args = parser.parse_args()
    
    main(args.root, args.skip_frames, args.skip_x, args.skip_y, args.savef)
