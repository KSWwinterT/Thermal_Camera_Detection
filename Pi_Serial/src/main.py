import serial
import time

# 시리얼 포트와 속도 설정
ser = serial.Serial('/dev/ttyS0', 9600, timeout=1)

try:
    while True:
        # 데이터 전송
        data_to_send = "Hello, Serial!"
        ser.write(data_to_send.encode('utf-8'))
        print(f"Sent: {data_to_send}")

        # 데이터 수신
        received_data = ser.readline().decode('utf-8').rstrip()
        if received_data:
            print(f"Received: {received_data}")

        time.sleep(1)

except KeyboardInterrupt:
    print("프로그램이 종료되었습니다.")
    ser.close()
