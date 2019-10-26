import cv2
import numpy
import tqdm


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
        return self._len

    def __iter__(self):
        ret = True
        state = 0
        out = []
        while ret and self._cap is not None:
            ret, frame = self._cap.read()
            if not state % self._fps:
                yield frame
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
        return numpy.stack(data, dim=0)
