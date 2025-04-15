from PyQt5.QtWidgets import *
import sys
import numpy as np  # numpy로 사운드 데이터를 생성
import sounddevice as sd  # 비프음을 재생하기 위해 sounddevice 사용


class BeepSound(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('삑 소리 내기')  # 윈도우 이름과 위치 지정
        self.setGeometry(200, 200, 500, 100)

        shortBeepButton = QPushButton('짧게 삑', self)  # 버튼 생성
        longBeepButton = QPushButton('길게 삑', self)
        quitButton = QPushButton('나가기', self)
        self.label = QLabel('환영합니다!', self)

        shortBeepButton.setGeometry(10, 10, 100, 30)  # 버튼 위치와 크기 지정
        longBeepButton.setGeometry(110, 10, 100, 30)
        quitButton.setGeometry(210, 10, 100, 30)
        self.label.setGeometry(10, 40, 500, 70)

        shortBeepButton.clicked.connect(self.shortBeepFunction)  # 콜백 함수 지정
        longBeepButton.clicked.connect(self.longBeepFunction)
        quitButton.clicked.connect(self.quitFunction)

    # 비프음 재생을 위한 함수 (numpy를 사용해 사운드 신호 생성)
    def playBeep(self, frequency, duration, amplitude=0.5, samplerate=44100):
        """frequency: 주파수, duration: 길이(ms), amplitude: 크기"""
        duration_in_seconds = duration / 1000.0  # ms를 초 단위로 변환
        t = np.linspace(0, duration_in_seconds, int(samplerate * duration_in_seconds), endpoint=False)
        wave = amplitude * np.sin(2 * np.pi * frequency * t)  # 사인파 생성
        sd.play(wave, samplerate)  # sounddevice로 사운드 재생
        sd.wait()  # 사운드 재생이 끝날 때까지 대기

    def shortBeepFunction(self):
        self.label.setText('주파수 1000으로 0.5초 동안 삑 소리를 냅니다.')
        self.playBeep(1000, 500)  # 주파수 1000Hz, 0.5초 동안 비프음 재생

    def longBeepFunction(self):
        self.label.setText('주파수 1000으로 3초 동안 삑 소리를 냅니다.')
        self.playBeep(1000, 3000)  # 주파수 1000Hz, 3초 동안 비프음 재생

    def quitFunction(self):
        self.close()


app = QApplication(sys.argv)
win = BeepSound()  # BeepSound 클래스의 객체 생성
win.show()
app.exec_()
