from PyQt5.QtWidgets import *
import sys
import cv2 as cv


class Video(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('비디오에서 프레임 수집')  # 윈도우 이름과 위치 지정
        self.setGeometry(200, 200, 500, 100)

        videoButton = QPushButton('비디오 켜기', self)  # 버튼 생성
        captureButton = QPushButton('프레임 잡기', self)
        saveButton = QPushButton('프레임 저장', self)
        quitButton = QPushButton('나가기', self)

        videoButton.setGeometry(10, 10, 100, 30)  # 버튼 위치와 크기 지정
        captureButton.setGeometry(110, 10, 100, 30)
        saveButton.setGeometry(210, 10, 100, 30)
        quitButton.setGeometry(310, 10, 100, 30)

        videoButton.clicked.connect(self.videoFunction)  # 콜백 함수 지정
        captureButton.clicked.connect(self.captureFunction)
        saveButton.clicked.connect(self.saveFunction)
        quitButton.clicked.connect(self.quitFunction)

    def videoFunction(self):
        # 수정된 부분: cv.CAP_DSHOW 플래그 제거
        self.cap = cv.VideoCapture(0)  # 플래그 없이 기본 카메라 연결
        if not self.cap.isOpened():
            self.close()

        # OpenCV 윈도우에서 비디오 스트림 표시
        while True:
            ret, self.frame = self.cap.read()
            if not ret:
                break
            cv.imshow('video display', self.frame)
            # 'q'를 누르면 종료
            if cv.waitKey(1) & 0xFF == ord('q'):
                break

    def captureFunction(self):
        # 프레임 캡처 후 표시
        self.capturedFrame = self.frame
        cv.imshow('Captured Frame', self.capturedFrame)

    def saveFunction(self):
        # 수정된 부분: macOS에서도 작동하는 파일 대화상자
        fname, _ = QFileDialog.getSaveFileName(self, '파일 저장', './', "Images (*.png *.xpm *.jpg *.jpeg *.bmp)")
        if fname:  # 파일 경로가 선택되었는지 확인
            cv.imwrite(fname, self.capturedFrame)

    def quitFunction(self):
        # 카메라 연결 및 모든 창 닫기
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()
        cv.destroyAllWindows()
        self.close()


app = QApplication(sys.argv)
win = Video()
win.show()
app.exec_()
