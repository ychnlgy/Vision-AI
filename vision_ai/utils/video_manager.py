import math

import cv2
import numpy
import tqdm
import torch


class VideoManager:

    def __init__(self, fps):
        self._fps = fps

    def __enter__(self):
        return self

    def __exit__(self, *args):
        cv2.destroyAllWindows()

    def parse(self, fname):
        return VideoParser(fname, self._fps)


class VideoParser:

    def __init__(self, fname, fps):
        self._pth = fname
        self._fps = fps
        self._cap = cv2.VideoCapture(fname)
        self._len = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.destroy()

    def destroy(self):
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __len__(self):
        return math.ceil(self._len / self._fps)

    def __iter__(self):
        ret = True
        state = 0
        out = []
        while ret and self._cap is not None:
            ret, frame = self._cap.read()
            if ret and not state % self._fps:
                yield torch.from_numpy(
                    frame
                ).permute(
                    2, 0, 1
                ).unsqueeze(
                    0
                )  # 1, C, W, H
            state += int(ret)
        self.destroy()

    def read(self):
        data = list(
            tqdm.tqdm(
                iter(self),
                ncols=80,
                total=len(self),
                desc="Slicing %s per %d frames" % (self._pth, self._fps)
            )
        )
        return torch.cat(data, axis=0)
