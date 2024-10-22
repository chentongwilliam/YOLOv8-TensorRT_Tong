import os
import numpy as np
import cv2
from pathlib import Path
import re
import logging
import traceback
import glob

logger = logging.getLogger(__name__)

img_formats = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng', 'webp', 'mpo']  # acceptable image suffixes
vid_formats = ['mov', 'avi', 'mp4', 'mpg', 'mpeg', 'm4v', 'wmv', 'mkv']  # acceptable video suffixes

def sorted_nicely(l): 
    """ Sort the given iterable in the way that humans expect.""" 
    convert = lambda text: int(text) if text.isdigit() else text 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(l, key = alphanum_key)


class LoadImages:  # for inference
    def __init__(self, path):
        p = str(Path(path).absolute())  # os-agnostic absolute path
        if '*' in p:
            files = sorted_nicely(glob.glob(p, recursive=True))  # glob
        elif os.path.isdir(p):
            files = sorted_nicely(glob.glob(os.path.join(p, '*.*')))  # dir
        elif os.path.isfile(p):
            files = [p]  # files
        else:
            raise Exception(f'ERROR: {p} does not exist')

        images = [x for x in files if x.split('.')[-1].lower() in img_formats]
        videos = [x for x in files if x.split('.')[-1].lower() in vid_formats]
        ni, nv = len(images), len(videos)

        self.files = images + videos
        self.nf = ni + nv  # number of files
        self.video_flag = [False] * ni + [True] * nv
        self.mode = 'image'
        if any(videos):
            self.new_video(videos[0])  # new video
        else:
            self.cap = None
        assert self.nf > 0, f'No images or videos found in {p}. ' \
                            f'Supported formats are:\nimages: {img_formats}\nvideos: {vid_formats}'

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        if self.count == self.nf:
            raise StopIteration
        path = self.files[self.count]

        if self.video_flag[self.count]:
            # Read video
            self.mode = 'video'
            ret_val, img0 = self.cap.read()
            if not ret_val:
                self.count += 1
                self.cap.release()
                if self.count == self.nf:  # last video
                    raise StopIteration
                else:
                    path = self.files[self.count]
                    self.new_video(path)
                    ret_val, img0 = self.cap.read()

            self.frame += 1
            print(f'video {self.count + 1}/{self.nf} ({self.frame}/{self.nframes}) {path}: ', end='')

        else:
            # Read image
            self.count += 1
            img0 = cv2.imread(path)  # BGR
            assert img0 is not None, 'Image Not Found ' + path
            #print(f'image {self.count}/{self.nf} {path}: ', end='')

        img = np.ascontiguousarray(img0)

        return img, path

    def new_video(self, path):
        self.frame = 0
        self.cap = cv2.VideoCapture(path)
        self.nframes = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def __len__(self):
        return self.nf  # number of files


class LoadWebcam:  # for inference
    def __init__(
        self, 
        pipe='0', 
        cam_w=640,
        cam_h=480,
        et = 300,
        wb = 1200,
        ):
        

        if pipe.isnumeric():
            pipe = eval(pipe)  # local camera
        # pipe = 'rtsp://192.168.1.64/1'  # IP camera
        # pipe = 'rtsp://username:password@192.168.1.64/1'  # IP camera with login
        # pipe = 'http://wmccpinetop.axiscam.net/mjpg/video.mjpg'  # IP golf camera

        self.pipe = pipe
        # camera config
        self.cap = None
        self.cam_w = cam_w
        self.cam_h = cam_h
        self.et = et
        self.wb = wb
        self.setCamera()
        # self.cap = cv2.VideoCapture(pipe)  # video capture object
        # self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)  # set buffer size

    def setCamera(self):
        camera_id = "/dev/video0"
        self.cap = cv2.VideoCapture(camera_id, cv2.CAP_V4L2)
        # How to set video capture properties using V4L2:
        # Full list of Video Capture Properties for OpenCV: https://docs.opencv.org/3.4/d4/d15/group__videoio__flags__base.html
        #Select Pixel Format:
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

        # Select frame size, FPS:
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cam_w)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cam_h)

        # Set Exposure mode:
        if self.et == 0:
            self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)
        else:
            self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1) #1 is manual mode,  3 is auto
            self.cap.set(cv2.CAP_PROP_EXPOSURE, self.et)

        # Set White Balance:
        if self.wb == 0:
            self.cap.set(cv2.CAP_PROP_AUTO_WB, 1)
        else:
            self.cap.set(cv2.CAP_PROP_AUTO_WB, 0)
            self.cap.set(cv2.CAP_PROP_WB_TEMPERATURE, self.wb)

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        try:
            self.count += 1
            if cv2.waitKey(1) == ord('q'):  # q to quit
                self.cap.release()
                cv2.destroyAllWindows()
                raise StopIteration

            # Read frame
            if self.pipe == 0:  # local camera
                ret_val, img0 = self.cap.read()
                # img0 = cv2.flip(img0, 1)  # flip left-right
            else:  # IP camera
                n = 0
                while True:
                    n += 1
                    self.cap.grab()
                    if n % 30 == 0:  # skip frames
                        ret_val, img0 = self.cap.retrieve()
                        if ret_val:
                            break

            # Print
            assert ret_val, f'Camera Error {self.pipe}'
            # print(f'webcam {self.count}: ', end='')

            img = np.ascontiguousarray(img0)

            return img, ""

        except:
            traceback.print_exc()
            logger.error(traceback.format_exc())
            StopIteration

    def __len__(self):
        return 0

