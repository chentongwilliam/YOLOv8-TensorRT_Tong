import argparse
from pathlib import Path

import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
from models.imageloader import LoadWebcam

from config import CLASSES, COLORS
from models.utils import blob, det_postprocess, letterbox, path_to_list


def main(args: argparse.Namespace) -> None:
    if args.method == 'cudart':
        from models.cudart_api import TRTEngine
    elif args.method == 'pycuda':
        from models.pycuda_api import TRTEngine

    Engine = TRTEngine(args.engine)
    H, W = Engine.inp_info[0].shape[-2:]

    if args.imgs == '/dev/video0':  # test for usb cam in jetson
        images = LoadWebcam(
                pipe='0',
                cam_w=3840,
                cam_h=2160,
                et=0,
                wb=0
            )
    else:
        images = path_to_list(args.imgs)
    save_path = Path(args.out_dir)

    if not args.show and not save_path.exists():
        save_path.mkdir(parents=True, exist_ok=True)
        
    frame_count = 0
    start_time = time.time()
    
    # init times
    loading_times = []
    preprocessing_times = []
    inference_times = []
    postprocessing_times = []
    saving_times = []
    loop_times = []

    for image in images:
        loop_start_time = time.time()
        
        if args.imgs == '/dev/video0':
            bgr = image[0]
        else:
            save_image = save_path / image.name
            # load image
            start_loading_time = time.time()
            bgr = cv2.imread(str(image))
            end_loading_time = time.time()
            loading_times.append(end_loading_time - start_loading_time)
        
        # preprocessing
        start_preprocessing_time = time.time()
        draw = bgr.copy()
        bgr, ratio, dwdh = letterbox(bgr, (W, H))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        tensor = blob(rgb, return_seg=False)
        dwdh = np.array(dwdh * 2, dtype=np.float32)
        tensor = np.ascontiguousarray(tensor)
        end_preprocessing_time = time.time()
        preprocessing_times.append(end_preprocessing_time - start_preprocessing_time)
        
        # inference
        start_inference_time = time.time()
        data = Engine(tensor)
        end_inference_time = time.time()
        inference_times.append(end_inference_time - start_inference_time)
        
        # postprocessing
        start_postprocessing_time = time.time()
        bboxes, scores, labels = det_postprocess(data)
        if bboxes.size != 0:
            bboxes -= dwdh
            bboxes /= ratio

            for (bbox, score, label) in zip(bboxes, scores, labels):
                bbox = bbox.round().astype(np.int32).tolist()
                cls_id = int(label)
                cls = CLASSES[cls_id]
                color = COLORS[cls]
                cv2.rectangle(draw, bbox[:2], bbox[2:], color, 2)
                cv2.putText(draw,
                            f'{cls}:{score:.3f}', (bbox[0], bbox[1] - 2),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.75, [225, 255, 255],
                            thickness=2)
            
        end_postprocessing_time = time.time()
        postprocessing_times.append(end_postprocessing_time - start_postprocessing_time)
        
        # save image
        start_saving_time = time.time()
        if args.show:
            draw = cv2.resize(draw, (1280, 720))
            cv2.imshow('result', draw)
            # cv2.waitKey(0)
            
            keyCode = cv2.waitKey(11) & 0xFF
            # Stop the program on the ESC key or 'q'
            if keyCode == 27 or keyCode == ord('q'):
                break
        else:
            cv2.imwrite(str(save_image), draw)
            
        end_saving_time = time.time()
        saving_times.append(end_saving_time - start_saving_time)
            
        frame_count += 1
        loop_end_time = time.time()
        loop_times.append(loop_end_time - loop_start_time)
        loop_fps = 1.0 / (loop_end_time - loop_start_time)
        print(f"Loop FPS: {loop_fps:.2f}")
        
    current_time = time.time()
    fps = frame_count / (current_time - start_time)
    print(f"Total FPS: {fps:.2f}")
    
    # count avg time
    if not args.imgs == '/dev/video0':
        avg_loading_time = np.array(loading_times).mean()
    avg_preprocessing_time = np.array(preprocessing_times).mean()
    avg_inference_time = np.array(inference_times).mean()
    avg_postprocessing_time = np.array(postprocessing_times).mean()
    avg_loop_time = np.array(loop_times).mean()
    if not args.show:
        avg_saving_time = np.array(saving_times).mean()

    if not args.imgs == '/dev/video0':
        print(f"Average Loading Time: {avg_loading_time:.3f} seconds")
    print(f"Average Preprocessing Time: {avg_preprocessing_time:.3f} seconds")
    print(f"Average Inference Time: {avg_inference_time:.3f} seconds")
    print(f"Average Postprocessing Time: {avg_postprocessing_time:.3f} seconds")
    if not args.show:
        print(f"Average Saving Time: {avg_saving_time:.3f} seconds")
    print(f"Average Loop Time: {avg_loop_time:.3f} seconds")

    # use matplotlib plot all times in one plot, using different color
    # plt.plot(loading_times, label='Loading Time', color='blue')
    # plt.plot(preprocessing_times, label='Preprocessing Time', color='orange')
    # plt.plot(inference_times, label='Inference Time', color='green')
    # plt.plot(postprocessing_times, label='Postprocessing Time', color='red')
    # plt.plot(saving_times, label='Saving Time', color='purple')
    # plt.plot(loop_times, label='Loop Time', color='olive')
    # plt.title('Time per Image for Each Step')
    # plt.xlabel('Image Index')
    # plt.ylabel('Time (seconds)')
    # plt.legend()
    # plt.savefig("time_consume_seg_without_torch.png", dpi=300)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--engine', type=str, help='Engine file')
    parser.add_argument('--imgs', type=str, help='Images file')
    parser.add_argument('--show',
                        action='store_true',
                        help='Show the detection results')
    parser.add_argument('--out-dir',
                        type=str,
                        default='./output',
                        help='Path to output file')
    parser.add_argument('--method',
                        type=str,
                        default='cudart',
                        help='CUDART pipeline')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    main(args)
