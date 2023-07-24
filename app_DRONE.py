from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QMenu, QAction
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QObject, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
from UI.ui import Ui_MainWindow
import sys
from time import sleep
app = QtWidgets.QApplication(sys.argv)
MainWindow = QtWidgets.QMainWindow()

import numpy as np
import argparse
import os
import platform
import sys
from pathlib import Path

import torch
import time

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from models.experimental import attempt_load
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode


class detectWorker(QThread):
    # Send image after detection
    send_img = pyqtSignal(np.ndarray)
    # Stop Detection
    stop_det = pyqtSignal()
    # Send result string
    send_result_text = pyqtSignal(list)
    def __init__(self) -> None:
        super(detectWorker, self).__init__()
        self.weights = ROOT / 'trainmodel/drone.pt'
        # self.source = '0'
        self.source = [None, None]
        self.source_det = None
        self.conf_thres = 0.25
        self.pause_sys = False
        self.jump_out = False
        self.percent_length = 1000 
        

    @torch.no_grad()
    def run(
        self,
        weights=ROOT / 'yolov5s.pt',  # model path or triton URL
        source=ROOT / 'data/images',  # file/dir/URL/glob/screen/0(webcam)
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='0',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS 
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        vid_stride=1,  # video frame-rate stride
):
        print('Started')
        source = str(self.source_det)
        save_img = not nosave and not source.endswith('.txt')  # save inference images
        is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
        is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
        webcam = source.isnumeric() or source.endswith('.streams') or (is_url and not is_file)
        screenshot = source.lower().startswith('screen')
        if is_url and is_file:
            source = check_file(source)  # download

        # Directories
        # save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
        # (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir
        save_dir = Path(project)

        # Load model
        device = select_device(device)
        model = DetectMultiBackend(self.weights, device=device, dnn=dnn, data=data, fp16=half)
        stride, names, pt = model.stride, model.names, model.pt
        imgsz = check_img_size(imgsz, s=stride)  # check image size

        # Dataloader
        bs = 1  # batch_size
        if webcam:
            view_img = check_imshow(warn=True)
            dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
            bs = len(dataset)
        elif screenshot:
            dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
        else:
            dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        vid_path, vid_writer = [None] * bs, [None] * bs

        # Run inference
        # model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
        seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
        # initial framme count
        count = 0
        percent = 0
        start_time = time.time()
        # iterate dataset
        iter(dataset)
        # Start detect loop
        while True:
            drone = mouse_bite = short = 0
            if not self.jump_out:
                if not self.pause_sys:
                    try:
                        path, im, im0s, vid_cap, s = next(dataset)
                        # Calculate FPS and count frame
                        count += 1
                        if count % 30 == 0 and count >= 30:
                            fps = int(30 / (time.time() - start_time))
                            print(f'FPS: {fps}')
                            start_time = time.time()
                            percent = int(count / vid_cap.get(cv2.CAP_PROP_FRAME_COUNT) * self.percent_length)
                        with dt[0]:
                            im = torch.from_numpy(im).to(model.device)
                            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
                            im /= 255  # 0 - 255 to 0.0 - 1.0
                            if len(im.shape) == 3:
                                im = im[None]  # expand for batch dim

                        # Inference
                        with dt[1]:
                            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
                            pred = model(im, augment=augment, visualize=visualize)

                        # NMS
                        with dt[2]:
                            pred = non_max_suppression(pred, self.conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

                        # Second-stage classifier (optional)
                        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

                        # Process predictions
                        for i, det in enumerate(pred):  # per image
                            seen += 1
                            if webcam:  # batch_size >= 1
                                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                                s += f'{i}: '
                            else:
                                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

                            p = Path(p)  # to Path
                            save_path = str(save_dir / p.name)  # im.jpg
                            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
                            s += '%gx%g ' % im.shape[2:]  # print string
                            s_result = 'Result: '
                            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                            imc = im0.copy() if save_crop else im0  # for save_crop
                            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
                            if len(det):
                                # Rescale boxes from img_size to im0 size
                                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                                # Print results
                                for c in det[:, 5].unique():
                                    n = (det[:, 5] == c).sum()  # detections per class
                                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                                    s_result += f"{n} {names[int(c)]}{'s' * (n > 1)}, "
                                    if int(c) == 2:
                                        drone = n
                                    elif int(c) == 1:
                                        mouse_bite = n
                                    else:
                                        short = n

                                # Write results
                                for *xyxy, conf, cls in reversed(det):
                                    if save_txt:  # Write to file
                                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                                        with open(f'{txt_path}.txt', 'a') as f:
                                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                                    if save_img or save_crop or view_img:  # Add bbox to image
                                        c = int(cls)  # integer class
                                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                                        annotator.box_label(xyxy, label, color=colors(c, True))
                                    if save_crop:
                                        save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

                            # Stream results
                            try:
                                # Stream results
                                self.im0 = annotator.result()
                                self.send_img.emit(self.im0)
                            except:
                                pass
                        # String result
                        resultList = [drone, mouse_bite, short]
                        self.send_result_text.emit(resultList)

                        if percent == self.percent_length:
                            print(count)
                            vid_cap.release()
                    except:
                        self.stop_det.emit()
                else:
                    continue
            else:
                self.stop_det.emit()


class myApp(Ui_MainWindow):
    def __init__(self):
        super().setupUi(MainWindow)
        self.detectWorker = detectWorker()
        # show result images
        self.detectWorker.send_img.connect(lambda x: self.UpdateImg(x, self.resultimg))
        # show result string
        self.detectWorker.send_result_text.connect(lambda x: self.UpdateResultDisplay(x, self.hole_dp, self.mouse_bite_dp, self.short_dp))
        self.detectWorker.stop_det.connect(self.StopDetect)
        self.setupSignalMainWindows()
        # initial
        self.conf_slider.setValue(50)
        self.detectWorker.conf_thres = self.conf_slider.value() / 100
        # Initial Button
        self.InitialButton()
        # Change text of show result
        self.label.setText('Drone')
        self.label_4.setText('')
        self.label_5.setText('')

    # initial button
    def InitialButton(self):
        if self.detectWorker.source is None:
            self.run_bt.setEnabled(False)
            self.pause_bt.setEnabled(False)
            self.stop_bt.setEnabled(False)

    def setupSignalMainWindows(self):
        self.conf_slider.sliderReleased.connect(self.UpdateConf)
        # Pause system
        self.pause_bt.clicked.connect(self.PauseDetect)
        # Continue system
        self.run_bt.clicked.connect(self.RunDetect)
        # Stop detect
        self.stop_bt.clicked.connect(self.StopDetect)
        # Selected image
        self.selected_video.clicked.connect(self.SelectedSourceFunc)
        # Close Program
        self.actionExit.triggered.connect(self.closeEvent)

    # Selected source
    def SelectedSourceFunc(self):
        file = QFileDialog.getOpenFileName(caption = 'Choose image/video', filter = 'Image files (*.jpg *.png *.jpeg *.bmp *.mp4)', directory = str(ROOT / 'trainmodel/testDRONE'))
        try:
            if file[0] != '':
                # Change source
                if self.detectWorker.source[-1] is None:
                    self.detectWorker.source[-1] = str(file[0])
                    self.detectWorker.source_det = self.detectWorker.source[-1]
                else:
                    # Move source[-1] to [0] and add new source to index [-1]
                    self.detectWorker.source[0] = self.detectWorker.source[-1]
                    self.detectWorker.source[-1] = str(file[0])
                    # check user add new source that not duplicate 
                    if self.detectWorker.source[-1] != self.detectWorker.source[0]:
                        self.detectWorker.source_det = self.detectWorker.source[-1]
                self.run_bt.setEnabled(True)
        except:
            pass

    # update confidence threshold
    def UpdateConf(self):
        self.detectWorker.conf_thres = self.conf_slider.value() / 100
        
    # Run detect Process
    def RunDetect(self):
        if not self.detectWorker.isRunning():
            self.detectWorker.start()
        else:
            self.detectWorker.pause_sys = False
        # set button
        self.run_bt.setEnabled(False)
        self.stop_bt.setEnabled(True)
        self.pause_bt.setEnabled(True)
        self.selected_video.setEnabled(False)
        
    # Pause detect
    def PauseDetect(self):
        self.detectWorker.pause_sys = True
        # set button
        self.run_bt.setEnabled(True)
        self.stop_bt.setEnabled(True)
        self.pause_bt.setEnabled(False)

    # Stop detect button
    def StopDetect(self):
        if self.detectWorker.isRunning():
            # terminate QThread
            try:
                self.detectWorker.terminate()
            except:
                pass
            # set button
            self.run_bt.setEnabled(True)
            self.stop_bt.setEnabled(False)
            self.pause_bt.setEnabled(False)
            self.selected_video.setEnabled(True)
        # Clear result image
        self.resultimg.clear()
        # clear result display widget
        self.hole_dp.display(0)
        self.mouse_bite_dp.display(0)
        self.short_dp.display(0)
        
    @staticmethod
    def UpdateImg(img_src, label):
        img_src = cv2.cvtColor(img_src, cv2.COLOR_BGR2RGB)
        height, width, channel = img_src.shape
        bytesPerLine = channel * width
        # Resize 
        img_src = cv2.resize(img_src, (int(width * 0.7), int(height * 0.7)), interpolation = cv2.INTER_AREA)
        height, width, channel = img_src.shape
        bytesPerLine = channel * width
        qImg = QImage(img_src.data, width, height, bytesPerLine, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qImg)
        label.setPixmap(pixmap)

    @staticmethod
    def UpdateResultDisplay(result, misshole_dp, bite_dp, short_dp):
        misshole_dp.display(int(result[0]))
        # bite_dp.display(int(result[1]))
        # short_dp.display(int(result[2]))

    def closeEvent(self):
        # Stop Thread
        self.detectWorker.jump_out = True
        # Emit signal for stop detection in Thread
        self.detectWorker.stop_det.emit()
        sys.exit(0)

if __name__ == '__main__':
    obj = myApp()
    MainWindow.show()
    sys.exit(app.exec_())