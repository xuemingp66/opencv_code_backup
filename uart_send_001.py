import serial
import time

serial_port = serial.Serial(
    port="/dev/ttyUSB0",
    baudrate=115200,
    bytesize=serial.EIGHTBITS,
    parity=serial.PARITY_NONE,
    stopbits=serial.STOPBITS_ONE,
)

time.sleep(1)

i = 0

try:
    while True:
        serial_port.write("Hello World\r\n".encode())
        serial_port.write(i)
        i += 1
        time.sleep(2)
finally:
    serial_port.close()