# coding=<utf-8>

from PyQt5 import QtCore, QtGui, QtWidgets

# 检测界面 （必备）
from ui.detect_ui import Ui_MainWindow  # 导入 UI目录下的 detect_ui
import detect

# 检测功能所需要的工具包 （必备）
import argparse
import os
import sys
from pathlib import Path
import cv2 as cv

# import win32gui

from utils.general import (increment_path)


# 创建主函数类
class Ui_logic_window(QtWidgets.QMainWindow):

    # 基本
    def __init__(self, parent=None):
        super(Ui_logic_window, self).__init__(parent)

        self.opt = None
        # self.timer_video = QtCore.QTimer()  # 创建定时器 视频
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.init_slots()  # 绑定控件

        # self.cap = cv2.VideoCapture()
        self.num_stop = 1  # 暂停与播放辅助信号，note：通过奇偶来控制暂停与播放
        self.output_folder = 'output/'
        self.vid_writer = None

        # 权重初始文件名
        self.openfile_name_model = None
        self.vid_name = None
        self.img_name = None
        self.cap_statue = None
        self.save_dir = None
        self.img_over = None

        # 视频相关
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.show_video)
        #        self.Button_open_cam.clicked.connect(self.video_button)
        self.cap_video = 0
        self.flag = 0
        self.img = []

    # 控件绑定相关操作
    def init_slots(self):
        self.ui.Button_load_model.clicked.connect(self.load_model)

        self.ui.Button_open_img.clicked.connect(self.open_img)
        self.ui.Button_open_vid.clicked.connect(self.open_vid)
        self.ui.Button_open_cam.clicked.connect(self.open_cam)
        self.ui.Button_detect.clicked.connect(self.detect)
        self.ui.Button_stop.clicked.connect(self.stop_detect)
        # self.ui.pushButton_9.clicked.connect(self.save_ss)
        # self.timer_video.timeout.connect(self.show_video_frame)

        self.ui.Button_open_img.setDisabled(True)
        self.ui.Button_open_vid.setDisabled(True)
        self.ui.Button_open_cam.setDisabled(True)
        self.ui.Button_detect.setDisabled(True)
        self.ui.Button_stop.setDisabled(True)

    # 模型
    def load_model(self):
        try:
            self.openfile_name_model, _ = QtWidgets.QFileDialog.getOpenFileName(self, '选择模型(以.pt结尾)', 'many/weights/', "*.pt")
        except OSError as reason:
            print(str(reason))
        else:
            if self.openfile_name_model:
                QtWidgets.QMessageBox.warning(self, u"Ok!", u"载入完成！", buttons=QtWidgets.QMessageBox.Ok, defaultButton=QtWidgets.QMessageBox.Ok)
                self.ui.message_box.append("模型载入完成!")
                self.ui.Button_open_img.setDisabled(False)
                self.ui.Button_open_vid.setDisabled(False)
                self.ui.Button_open_cam.setDisabled(False)
            else:
                QtWidgets.QMessageBox.warning(self, u"Warning", u"无权重文件，请先选择权重文件，否则会发生未知错误。", buttons=QtWidgets.QMessageBox.Ok, defaultButton=QtWidgets.QMessageBox.Ok)
                self.ui.message_box.append("模型载入失败!")
        # self.model_init(self,  **self.openfile_name_model )

    # 图片
    def open_img(self):
        # try except
        try:
            # self.img_name 选择图片路径
            self.img_name, _ = QtWidgets.QFileDialog.getOpenFileName(self, "打开图片", "data/images", "*.jpg ; *.png ; All Files(*)")
        except OSError as reason:
            print(str(reason))
        # else
        else:  # 判断self.img_name  图像是否选选择，若未选择则弹出打开失败，若选择则执行else
            if not self.img_name:
                QtWidgets.QMessageBox.warning(self, u"⚠警告", u"打开图片失败", buttons=QtWidgets.QMessageBox.Ok, defaultButton=QtWidgets.QMessageBox.Ok)
                self.ui.message_box.append("图片载入失败。")
            # 执行段
            else:
                # 按钮状态
                self.ui.Button_detect.setDisabled(False)
                self.ui.Button_open_vid.setDisabled(True)
                self.ui.Button_open_cam.setDisabled(True)
                self.ui.Button_stop.setDisabled(False)
                # 图片链接Label方法
                # 旧的open_img_label 方法于2022/8/12注释
                # open_img_label = QtGui.QPixmap(self.img_name).scaled(self.ui.label_11.width(), self.ui.label_11.height())

                # 创建QImage图像
                img = QtGui.QImage(self.img_name)
                scale_1 = img.width() / self.ui.label_11.width()
                scale_2 = img.height() / self.ui.label_11.height()
                scale = max(scale_1, scale_2)
                # print(scale, img.width(), img.height(), "thescale")
                # 缩放率算法判断宽高是否超过框架 Label层
                if scale > 1:
                    w = int(img.width() // scale)
                    h = int(img.height() // scale)
                    img = img.scaled(int(img.width() / scale), int(img.height() / scale))
                    sordidness = "图像(width，height)=({},{});  缩放比例为:{:.3f}%".format(img.width(), img.height(), 1 / scale * 100)
                    # self.label_text.setText(str)
                    print(sordidness)
                else:
                    sordidness2 = "图像(width，height)=({},{});  缩放比例为:100%".format(img.width(), img.height())
                    # self.label_text.setText(str)
                    print(sordidness2)
                # 设定重置后的label的x，y，宽和高
                image_x = (self.ui.label_11.width() - img.width()) // 2 + self.ui.label_11.x()
                image_y = (self.ui.label_11.height() - img.height()) // 2 + self.ui.label_11.y()
                self.ui.image_label.setGeometry(QtCore.QRect(image_x, image_y, img.width(), img.height()))
                # size函数
                size = QtCore.QSize(img.width(), img.height())
                # 设定最后的图像
                opened_img = QtGui.QPixmap.fromImage(img.scaled(size, QtCore.Qt.IgnoreAspectRatio))
                # 设置图片显示
                self.ui.image_label.setPixmap(opened_img)
                # 提示信息
                self.ui.message_box.append("图片载入完成。")

    # 视频
    def open_vid(self):
        print("视频检测了")

        self.vid_name, _ = QtWidgets.QFileDialog.getOpenFileName(self, "打开视频", "data/videos", "*.mp4;*.mkv;All Files(*)")
        print(self.vid_name)
        # flag = self.cap.open(self.vid_name)
        if not self.vid_name:
            QtWidgets.QMessageBox.warning(self, u"Warning", u"打开视频失败", buttons=QtWidgets.QMessageBox.Ok, defaultButton=QtWidgets.QMessageBox.Ok)
            self.ui.message_box.append("视频载入失败。")
        else:
            vid = QtGui.QPixmap(self.vid_name).scaled(self.ui.label_11.width(), self.ui.label_11.height())
            self.ui.label_11.setPixmap(vid)
            print('视频地址为：' + str(self.vid_name))
            self.ui.message_box.append("视频载入完成。")
            self.ui.Button_detect.setDisabled(False)
            self.ui.Button_open_img.setDisabled(True)
            self.ui.Button_stop.setDisabled(False)
            self.ui.Button_open_cam.setDisabled(True)

    # 摄像头
    def open_cam(self):
        # 按钮状态
        self.ui.Button_detect.setDisabled(False)
        self.ui.Button_open_img.setDisabled(True)
        self.ui.Button_open_vid.setDisabled(True)
        self.ui.Button_open_cam.setDisabled(True)
        self.ui.Button_stop.setDisabled(False)

        #
        # self.cap_statue = 1
        # cap = cv.VideoCapture(0)
        #
        # while True:
        #
        #     # 开始用摄像头读数据，返回hx为true则表示读成功，frame为读的图像
        #     hx, frame = cap.read()
        #
        #     # 如果hx为Flase表示开启摄像头失败，那么就输出"read vido error"并退出程序
        #     if hx is False:
        #         # 打印报错
        #         print('read video error')
        #         # 退出程序
        #         exit(0)
        #
        #     # 显示摄像头图像，其中的video为窗口名称，frame为图像
        #     cv.imshow('video', frame)
        #
        #     cv.waitKey(1)
        #     # 监测键盘输入是否为q，为q则退出程序
        #     # if cv.waitKey(1) & 0xFF == ord('q'):  # 按q退出
        #     #     break
        #     if self.cap_statue == 0:  # 按q退出
        #         break
        #
        # # 释放摄像头
        # cap.release()
        #
        # # 结束所有窗口
        # cv.destroyAllWindows()
        self.cap_video = cv.VideoCapture(0)
        self.timer.start(50)
        self.flag += 1
        # self.Button_open_cam.setText("Close")

        # 提示信息
        print("打开摄像头")

    #
    def show_video(self):
        ret, self.img = self.cap_video.read()
        if ret:
            self.show_cv_img(self.img)

    # 摄像头-展示接受到的图像到label上
    def show_cv_img(self, img):
        shrink = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        print(shrink)
        QtImg = QtGui.QImage(shrink.data, shrink.shape[1], shrink.shape[0], shrink.shape[1] * 3, QtGui.QImage.Format_RGB888)
#----------
        img = QtGui.QImage(QtImg)
        scale_1 = img.width() / self.ui.label_11.width()
        scale_2 = img.height() / self.ui.label_11.height()
        scale = max(scale_1, scale_2)
        # print(scale, img.width(), img.height(), "thescale")
        # 缩放率算法判断宽高是否超过框架 Label层
        if scale > 1:
            w = int(img.width() // scale)
            h = int(img.height() // scale)
            img = img.scaled(int(img.width() / scale), int(img.height() / scale))
            sordidness = "图像(width，height)=({},{});  缩放比例为:{:.3f}%".format(img.width(), img.height(), 1 / scale * 100)
            # self.label_text.setText(str)
            print(sordidness)
        else:
            sordidness2 = "图像(width，height)=({},{});  缩放比例为:100%".format(img.width(), img.height())
            # self.label_text.setText(str)
            print(sordidness2)
        # 设定重置后的label的x，y，宽和高
        image_x = (self.ui.label_11.width() - img.width()) // 2 + self.ui.label_11.x()
        image_y = (self.ui.label_11.height() - img.height()) // 2 + self.ui.label_11.y()
        self.ui.image_label.setGeometry(QtCore.QRect(image_x, image_y, img.width(), img.height()))
        # size函数
        size = QtCore.QSize(img.width(), img.height())
        # 设定最后的图像
        opened_cam = QtGui.QPixmap.fromImage(img.scaled(size, QtCore.Qt.IgnoreAspectRatio))

 #-------
        # jpg_out = QtGui.QPixmap(QtImg).scaled(self.ui.label_11.width(), self.ui.label_11.height())

        self.ui.image_label.setPixmap(opened_cam)
        # 提示信息

    # 闭包
    def parse_opt(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model path(s)')
        parser.add_argument('--source', type=str, default='data/images', help='file/dir/URL/glob, 0 for webcam')
        parser.add_argument('--data', type=str, default='data/coco128.yaml', help='(optional) dataset.yaml path')
        parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640],
                            help='inference size h,w')
        parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
        parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
        parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
        parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
        parser.add_argument('--view-img', action='store_true', help='show results')
        parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
        parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
        parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
        parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
        parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
        parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
        parser.add_argument('--augment', action='store_true', help='augmented inference')
        parser.add_argument('--visualize', action='store_true', help='visualize features')
        parser.add_argument('--update', action='store_true', help='update all models')
        parser.add_argument('--project', default='runs/detect', help='save results to project/name')
        parser.add_argument('--name', default='exp', help='save results to project/name')
        parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
        parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
        parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
        parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
        parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
        parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
        self.opt = parser.parse_args()
        self.opt.imgsz *= 2 if len(self.opt.imgsz) == 1 else 1  # expand
        # print_args(FILE.stem, opt)
        return self.opt

    # 检测
    def detect(self):
        self.ui.Button_stop.setDisabled(False)
        opt = self.parse_opt()
        opt.weights = self.openfile_name_model
        if self.img_name:
            # 设定每次自动检测的保存文件夹如("exp3","exp4","exp5") 此地址会自动检测输出结果的位置
            self.save_dir = increment_path(Path('runs\\detect') / 'exp', exist_ok=False)
            # print(self.save_dir)
            # self.ui.message_box.append(str(self.save_dir))
            # 初始化输入参数 opt 把图像地址输入进初始化参数
            opt.source = self.img_name
            # Run detect.ui里的Run函数
            detect.run(**vars(opt))

            # 图片文件名读取
            m_file = os.path.basename(self.img_name)
            t_file = 'E:/K01/Documents/pythonProject1/yolov5-6.1/' + str(self.save_dir) + '/' + m_file
            # 反斜杠正规化 利用normpath
            i_file = os.path.normpath(t_file)
            # 图像居中规整化
            img = QtGui.QImage(i_file)
            scale_1 = img.width() / self.ui.label_5.width()
            scale_2 = img.height() / self.ui.label_5.height()
            scale = max(scale_1, scale_2)
            print(scale, img.width(), img.height(), "thescale")
            # 缩放率算法判断宽高是否超过框架 Label层
            if scale > 1:
                w = int(img.width() // scale)
                h = int(img.height() // scale)
                img = img.scaled(int(img.width() / scale), int(img.height() / scale))
                sordidness = "图像(width，height)=({},{});  缩放比例为:{:.3f}%".format(img.width(), img.height(), 1 / scale * 100)
                # self.label_text.setText(str)
                print(sordidness)
            else:
                sordidness2 = "图像(width，height)=({},{});  缩放比例为:100%".format(img.width(), img.height())
                # self.label_text.setText(str)
                print(sordidness2)
            # 设定重置后的label的x，y，宽和高
            image_x = (self.ui.label_5.width() - img.width()) // 2 + self.ui.label_5.x()
            image_y = (self.ui.label_5.height() - img.height()) // 2 + self.ui.label_5.y()
            self.ui.image_label2.setGeometry(QtCore.QRect(image_x, image_y, img.width(), img.height()))
            # size函数
            size = QtCore.QSize(img.width(), img.height())
            detected_img = QtGui.QPixmap.fromImage(img.scaled(size, QtCore.Qt.IgnoreAspectRatio))
            # detected_img = QtGui.QPixmap(t_file).scaled(self.ui.label_11.width(), self.ui.label_11.height())
            self.ui.image_label2.setPixmap(detected_img)

            # 提示信息
            # print(self.save_dir)
            # print(m_file)
            # print(t_file)
            # print(i_file)
            self.ui.message_box.append("图像检测开始")
            self.ui.message_box.append("检测完成")

        if self.vid_name:
            print("视频")

        if self.flag == 1:
            print("摄像头")
            # 代码段
            opt.source = 0
            detect.run(**vars(opt))
            if self.cap_statue == 0:  # 按q退出
                print("点击了按钮，但没法关闭窗口0")


    # 结束
    def stop_detect(self):
        # 结束检测并重置所有状态 True
        self.ui.Button_detect.setDisabled(True)  # 关闭检测
        self.ui.Button_stop.setDisabled(True)  # 关闭结束
        self.ui.Button_open_img.setDisabled(False)  # 激活图像
        self.ui.Button_open_vid.setDisabled(False)  # 激活视频
        self.ui.Button_open_cam.setDisabled(False)  # 激活摄像机按钮

        # 图像路径状态
        self.img_name = None
        # 视频路径清除
        self.vid_name = None
        # 摄像头关闭
        if self.flag == 1:
            self.timer.stop()
            self.cap_video.release()
            self.ui.label_11.clear()
            # self.pushButton.setText("Open")
            self.flag = 0
        # 两个画布清除
        self.ui.image_label.clear()
        self.ui.image_label2.clear()
        self.ui.label_5.clear()

    # 保存路径
    @staticmethod
    def save_ss():
        print("the save-path")


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    current_ui = Ui_logic_window()
    current_ui.show()
    sys.exit(app.exec_())
