import cv2 as cv
import numpy as np
from PyQt5.QtWidgets import *
import sys
import os  # 시스템 명령어 사용
import sounddevice as sd  # macOS에서 비프음을 대체하기 위한 라이브러리


class TrafficWeak(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('교통약자 보호')
        self.setGeometry(200, 200, 700, 200)

        signButton = QPushButton('표지판 등록', self)
        roadButton = QPushButton('도로 영상 불러오기', self)
        recognitionButton = QPushButton('표지판 인식', self)
        quitButton = QPushButton('나가기', self)
        self.label = QLabel('환영합니다!', self)

        signButton.setGeometry(10, 10, 100, 30)
        roadButton.setGeometry(110, 10, 130, 30)
        recognitionButton.setGeometry(270, 10, 130, 30)
        quitButton.setGeometry(510, 10, 100, 30)
        self.label.setGeometry(10, 60, 400, 170)

        # 버튼 이벤트 연결 (잘못된 연결 수정)
        signButton.clicked.connect(self.sign)  # 기존 signFunction -> 수정: sign
        roadButton.clicked.connect(self.roadFunction)
        recognitionButton.clicked.connect(self.recognitionFunction)
        quitButton.clicked.connect(self.quitFunction)

        self.signFiles = ['child.png', 'elder.png', 'disabled.png']
        self.signImgs = []

        self.sign()  # 표지판 로드 함수 호출 시 예외 처리 추가

    def sign(self):
        self.label.clear()
        self.label.setText('교통약자 표지판을 등록합니다...')
        for fname in self.signFiles:
            img = cv.imread(fname)
            if img is None:
                print(f"Warning: 파일 {fname}을(를) 찾을 수 없습니다.")
                continue
            self.signImgs.append(img)
            cv.imshow(fname, img)
        if not self.signImgs:
            self.label.setText("표지판을 하나도 로드하지 못했습니다. 파일을 확인하세요.")

    def roadFunction(self):
        if not self.signImgs:  # 표지판이 로드되지 않았으면 동작하지 않음
            self.label.setText('먼저 표지판을 등록하세요.')
            return

        fname, _ = QFileDialog.getOpenFileName(self, '파일 열기', './')
        if not fname:  # 사용자가 취소를 누를 경우
            self.label.setText('파일 선택이 취소되었습니다.')
            return

        self.roadImg = cv.imread(fname)
        if self.roadImg is None:
            self.label.setText('파일을 열 수 없습니다. 경로를 확인하세요.')
            return

        cv.imshow('Road scene', self.roadImg)

    def recognitionFunction(self):
        if not hasattr(self, 'roadImg'):  # 도로 영상이 없으면 알림
            self.label.setText('먼저 도로 영상을 입력하세요.')
            return

        sift = cv.SIFT_create()
        key_des_pairs = []

        for img in self.signImgs:
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            kp, des = sift.detectAndCompute(gray, None)
            key_des_pairs.append((kp, des))

        grayRoad = cv.cvtColor(self.roadImg, cv.COLOR_BGR2GRAY)
        kp_road, des_road = sift.detectAndCompute(grayRoad, None)

        matcher = cv.DescriptorMatcher_create(cv.DescriptorMatcher_FLANNBASED)
        good_matches = []

        for sign_kp, sign_des in key_des_pairs:
            knn_match = matcher.knnMatch(sign_des, des_road, 2)
            T = 0.7  # 임계값
            good_match = [
                nearest1 for nearest1, nearest2 in knn_match
                if nearest1.distance / nearest2.distance < T
            ]
            good_matches.append(good_match)

        if not good_matches:
            self.label.setText('어떤 표지판도 도로 영상에서 인식되지 않았습니다.')
            return

        best_match_idx = good_matches.index(max(good_matches, key=len))
        cv.imshow('Matches and Homography', self.signImgs[best_match_idx])
        self.label.setText(self.signFiles[best_match_idx] + ' 보호구역입니다! 30km 이하로 서행하세요.')

        self.beep()

    def beep(self):
        try:
            # macOS 기본 비프음
            os.system("/usr/bin/osascript -e 'beep'")
        except Exception as e:
            print("Beep 실패:", e)

    def quitFunction(self):
        cv.destroyAllWindows()
        self.close()


app = QApplication(sys.argv)
win = TrafficWeak()
win.show()
app.exec_()
