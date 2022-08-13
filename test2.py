import sys

import ui.ui_test
from PyQt5.QtWidgets import *
import cv2
from PyQt5 import QtCore, QtGui, QtWidgets


class test_ui(QMainWindow, ui.ui_test.Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.show_viedo)
        self.pushButton.clicked.connect(self.video_button)
        self.cap_video = 0
        self.flag = 0
        self.img = []

    # 按钮响应
    def video_button(self):
        if self.flag == 0:
            self.cap_video = cv2.VideoCapture(0)
            self.timer.start(50);
            self.flag += 1
            self.pushButton.setText("Close")
        else:
            self.timer.stop()
            self.cap_video.release()
            self.label.clear()
            self.pushButton.setText("Open")
            self.flag = 0

    # 展示帧画面= 显示功能
    def show_viedo(self):
        ret, self.img = self.cap_video.read()
        if ret:
            self.show_cv_img(self.img)

    # 展示CV的图像
    def show_cv_img(self, img):
        shrink = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        QtImg = QtGui.QImage(shrink.data,shrink.shape[1],shrink.shape[0],shrink.shape[1] * 3,QtGui.QImage.Format_RGB888)
        jpg_out = QtGui.QPixmap(QtImg).scaled(self.label.width(), self.label.height())
        self.label.setPixmap(jpg_out)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = test_ui()
    win.show()
    sys.exit(app.exec_())
