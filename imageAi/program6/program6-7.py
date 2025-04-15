import cv2 as cv
import numpy as np
from PyQt5.QtWidgets import *
from PyQt5.QtCore import Qt
import sys


class VideoSpecialEffect(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('비디오 특수 효과')
        self.setGeometry(200, 200, 400, 100)

        videoButton = QPushButton('비디오 재생', self)
        self.pickCombo = QComboBox(self)
        self.pickCombo.addItems(['엠보싱', '카툰', '연필 스케치(회색)', '연필 스케치(컬러)', '유화'])
        quitButton = QPushButton('나가기', self)

        videoButton.setGeometry(10, 10, 140, 30)
        self.pickCombo.setGeometry(160, 10, 180, 30)
        quitButton.setGeometry(270, 60, 100, 30)

        # 버튼 이벤트 연결
        videoButton.clicked.connect(self.videoSpecialEffectFunction)
        quitButton.clicked.connect(self.quitFunction)

        # 비디오 캡쳐 객체
        self.cap = None

    def videoSpecialEffectFunction(self):
        # 카메라 열기 (macOS 호환을 위해 기본값으로)
        self.cap = cv.VideoCapture(0)
        if not self.cap.isOpened():
            QMessageBox.critical(self, '에러', '카메라 장치를 열 수 없습니다.', QMessageBox.Ok)
            return

        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                QMessageBox.warning(self, '경고', '카메라 입력이 중단되었습니다.', QMessageBox.Ok)
                break

            # 현재 선택된 효과 인덱스를 가져옴
            pick_effect = self.pickCombo.currentIndex()

            # 각 효과별 처리
            if pick_effect == 0:  # 엠보싱
                kernel = np.array([[-1, -1, 0],
                                   [1, 0, 1],
                                   [0, 1, 1]])
                gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                gray16 = np.int16(gray)
                special_img = np.uint8(np.clip(cv.filter2D(gray16, -1, kernel) + 128, 0, 255))
            elif pick_effect == 1:  # 카툰 효과
                special_img = cv.stylization(frame, sigma_s=60, sigma_r=0.45)
            elif pick_effect == 2:  # 연필 스케치 (회색)
                special_img, _ = cv.pencilSketch(frame, sigma_s=60, sigma_r=0.07, shade_factor=0.02)
            elif pick_effect == 3:  # 연필 스케치 (컬러)
                _, special_img = cv.pencilSketch(frame, sigma_s=60, sigma_r=0.07, shade_factor=0.02)
            elif pick_effect == 4:  # 유화 효과
                special_img = cv.xphoto.oilPainting(frame, size=10, dynRatio=1, code=cv.COLOR_BGR2Lab)
            else:
                # 기본 프레임 출력 (효과 선택 없음)
                special_img = frame

            # 결과 이미지 출력
            cv.imshow('Special Effect', special_img)

            # 키 입력 처리 (ESC 키로 종료)
            if cv.waitKey(1) & 0xFF == 27:
                break

        # 자원 해제 및 창 닫기
        self.cap.release()
        cv.destroyAllWindows()

    def quitFunction(self):
        # 프로그램 종료 시 리소스 정리
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()
        cv.destroyAllWindows()
        self.close()


# 프로그램 실행
if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = VideoSpecialEffect()
    win.show()
    app.exec_()
