import serial as ser
import struct, time

a = 'z'
c = a.encode('utf-8')
b = 678
se = ser.serial('dev/ttyTHS1', 115200, timeout = 0.5)
def recv(serial):
    while True:
        data = serial.read(64)
        if data == ' ':
            continue
        else:
            break
    return data
while True:
    data = recv(se)
    if data != ' ':
        print(data)
    se.write(str(b).encode('utf-8'))
    se.write(a.encode('utf-8'))
    time.sleep(1)