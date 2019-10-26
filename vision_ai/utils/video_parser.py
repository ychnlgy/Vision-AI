import cv2


class VideoParser:
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        cv2.destroyAllWindows()
    
    def parse(self, fname, fps):
        cap = cv2.VideoCapture(fname)
        ret = True
        state = 0
        out = []
        while ret:
            ret, frame = cap.read()
            if not state % fps:
                yield frame
            state += int(ret)
        cap.release()
