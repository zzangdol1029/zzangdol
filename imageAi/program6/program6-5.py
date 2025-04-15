from PyQt5.QtWidgets import *
import cv2 as cv
import numpy as np
import sys
import os  # macOS/Linux에서 비프음을 대체하기 위한 시스템 명령어 사용


class Panorama(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('파노라마 영상')
        self.setGeometry(200, 200, 700, 200)

        # 버튼 및 라벨 초기화
        collectButton = QPushButton('영상 수집', self)
        self.showButton = QPushButton('영상 보기', self)
        self.stitchButton = QPushButton('파노라마 생성', self)
        self.saveButton = QPushButton('저장', self)
        quitButton = QPushButton('나가기', self)
        self.label = QLabel('환영합니다!', self)

        # 버튼 배치
        collectButton.setGeometry(10, 25, 100, 30)
        self.showButton.setGeometry(110, 25, 100, 30)
        self.stitchButton.setGeometry(210, 25, 100, 30)
        self.saveButton.setGeometry(310, 25, 100, 30)
        quitButton.setGeometry(450, 25, 100, 30)
        self.label.setGeometry(10, 60, 600, 170)

        # 버튼 비활성화 설정
        self.showButton.setEnabled(False)
        self.stitchButton.setEnabled(False)
        self.saveButton.setEnabled(False)

        # 버튼 클릭 이벤트 연결
        collectButton.clicked.connect(self.collectFunction)
        self.showButton.clicked.connect(self.showFunction)
        self.stitchButton.clicked.connect(self.stitchFunction)
        self.saveButton.clicked.connect(self.saveFunction)
        quitButton.clicked.connect(self.quitFunction)

        # 캡처 객체 초기화
        self.cap = None
        self.imgs = []

    def collectFunction(self):
        self.imgs = []  # 이전에 저장된 이미지 초기화

        # macOS/Linux에서 CAP_AVFOUNDATION 또는 기본값 0 사용
        self.cap = cv.VideoCapture(0, cv.CAP_AVFOUNDATION)
        if not self.cap.isOpened():
            self.label.setText('카메라를 열 수 없습니다.')
            return

        self.label.setText('스페이스 키를 누를 때마다 영상을 저장하고 ESC 키를 누르면 종료합니다.')

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            cv.imshow('Video Display', frame)

            key = cv.waitKey(1)
            if key == 32:  # Space bar
                self.imgs.append(frame)
                self.label.setText(f'{len(self.imgs)}장의 영상이 수집되었습니다.')
            elif key == 27:  # ESC 키
                break

        # 캡처 종료
        self.cap.release()
        cv.destroyWindow('Video Display')

        # 파노라마 처리 관련 버튼 활성화
        if len(self.imgs) > 1:
            self.showButton.setEnabled(True)
            self.stitchButton.setEnabled(True)
            self.saveButton.setEnabled(False)  # 새 파노라마 생성 필요

    def showFunction(self):
        if not self.imgs:
            self.label.setText('수집된 영상이 없습니다. 먼저 영상을 수집하세요.')
            return

        stack = cv.resize(self.imgs[0], dsize=(0, 0), fx=0.25, fy=0.25)
        for img in self.imgs[1:]:
            stack = np.hstack((stack, cv.resize(img, dsize=(0, 0), fx=0.25, fy=0.25)))

        cv.imshow('Image Collection', stack)
        self.label.setText(f'수집된 영상은 총 {len(self.imgs)}장입니다.')

    def stitchFunction(self):
        if len(self.imgs) < 2:
            self.label.setText('파노라마를 생성하려면 최소 2개의 영상이 필요합니다.')
            return

        self.label.setText('파노라마를 생성 중입니다. 잠시 기다려주세요...')
        stitcher = cv.Stitcher_create()
        status, self.img_stitched = stitcher.stitch(self.imgs)

        if status == cv.Stitcher_OK:
            cv.imshow('Image Stitched Panorama', self.img_stitched)
            self.label.setText('파노라마 영상이 성공적으로 생성되었습니다!')
            self.saveButton.setEnabled(True)
            self.beep()
        else:
            self.label.setText('파노라마 생성에 실패했습니다. 다시 시도하세요.')
            self.beep()

    def saveFunction(self):
        if not hasattr(self, 'img_stitched'):
            self.label.setText('저장할 파노라마가 없습니다.')
            return

        fname, _ = QFileDialog.getSaveFileName(self, '파일 저장', './', "Images (*.png *.jpg *.bmp)")
        if fname:
            cv.imwrite(fname, self.img_stitched)
            self.label.setText(f'파노라마를 "{fname}"에 저장했습니다.')
        else:
            self.label.setText('저장이 취소되었습니다.')

    def quitFunction(self):
        if self.cap and self.cap.isOpened():
            self.cap.release()
        cv.destroyAllWindows()
        self.close()

    def beep(self):
        try:
            # macOS/Linux에서는 시스템 알림음을 대체 실행
            os.system("/usr/bin/osascript -e 'beep'")
        except Exception as e:
            print("Beep 실패:", e)


# 프로그램 실행
if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = Panorama()
    win.show()
    app.exec_()
