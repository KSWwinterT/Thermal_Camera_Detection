from flask import Flask, render_template
import cv2
import numpy as np
import serial
import threading
import io

app = Flask(__name__)

# 시리얼 포트 설정
ser = serial.Serial('/dev/ttyUSB0', 9600, timeout=1)  # 시리얼 포트 경로와 속도를 알맞게 수정

# 카메라 초기화
cap = cv2.VideoCapture(0)

# 데이터를 저장할 전역 변수
camera_data = ""

def serial_reader():
    global camera_data
    while True:
        # 데이터 수신
        received_data = ser.readline().decode('utf-8').rstrip()
        if received_data:
            print(f"Received: {received_data}")
            camera_data = received_data

# 시리얼 데이터를 읽는 쓰레드 시작
serial_thread = threading.Thread(target=serial_reader)
serial_thread.start()

# 웹 페이지 렌더링
@app.route('/')
def index():
    return render_template('index.html', camera_data=camera_data)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
