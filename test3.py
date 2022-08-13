

def open_img(self):
    # try except
    try:
        # self.img_name 选择图片路径
        self.img_name, _ = QtWidgets.QFileDialog.getOpenFileName(self, "打开图片", "data/images", "*.jpg ; *.png;All Files(*)")
    except OSError as reason:
        print(str(reason))
    # else
    else:  # 判断图像是否选选择，若未选择则弹出打开失败，若选择则执行else
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
            open_img_label = QtGui.QPixmap(self.img_name).scaled(self.ui.label_11.width(), self.ui.label_11.height())
            # 2022-8-11 4:01 04点01分
            image = cv.imread(self.img_name)
            width, height, channels = image.shape
            # 04点01分
            self.ui.label_11.setPixmap(open_img_label)
            # 提示信息
            self.ui.message_box.append("图片载入完成。")
            print("image:" + str(width) + "+" + str(height))