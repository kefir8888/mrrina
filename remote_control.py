import cv2
import numpy as np
import time
from IPython.display import clear_output
import math
import requests
import time


def get_middle_point(image):
    nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(image)
    if len(stats) > 1:
        i_max = 1
        for i in range(2, len(stats)):
            if stats[i, cv2.CC_STAT_AREA] > stats[i_max, cv2.CC_STAT_AREA]:
                i_max = i
        x = stats[i_max, cv2.CC_STAT_LEFT] + (stats[i_max, cv2.CC_STAT_WIDTH]) / 2
        y = stats[i_max, cv2.CC_STAT_TOP] + (stats[i_max, cv2.CC_STAT_HEIGHT]) / 2
        return x, y
    return -1, -1


# TODO: свитч на кнопки, словарь с активностями
# TODO: отрефакторить это произведение искусства
# TODO: поменять клавиатурный интерфейс хождения
# TODO: словарь на действия и кнопки, а то совсем плохо
# TODO: добавить хэндлинг русских букв
# TODO: добавить флаг на работу в автономном режиме, без робота
# TODO: добавить кнопки на: руку вперед, "привет", разжать/сжать пальцы, танец

WIND_X = 200
WIND_Y = 200

cv2.namedWindow("remote_controller", cv2.WINDOW_AUTOSIZE)
cv2.resizeWindow("remote_controller", (WIND_Y, WIND_X))

ip_prefix = "http://"

# ip_num = "10.197.241.216"
# ip_num = "10.0.0.103"
ip_num = "192.168.43.65"

# ip_num = "192.168.43.42"
ip_postfix = ":"

ip = ip_prefix + ip_num + ip_postfix

port = "9562"

canvas = np.ones((WIND_Y, WIND_X, 3), np.uint8) * 100

queue = []
queue_ = []

to_next_operation = False

curr_time = time.time()
time_of_prev_press = 0.0

cam1 = cv2.VideoCapture(1)

cv2.namedWindow("frame", cv2.WINDOW_NORMAL)

cv2.resizeWindow("frame", (960, 720))

while True:
    ret1, frame1 = cam1.read()

    cv2.waitKey(1)

    ker_sz = 19

    frame1_hsv = cv2.cvtColor(frame1, cv2.COLOR_BGR2HSV)

    lower_red = np.array([0, 120, 70])
    upper_red = np.array([10, 255, 255])

    mask_red = cv2.inRange(frame1_hsv, lower_red, upper_red)

    lower_blue = np.array([120, 120, 70])
    upper_blue = np.array([140, 255, 255])

    mask_blue = cv2.inRange(frame1_hsv, lower_blue, upper_blue)

    clear_output(wait=True)

    x_blue, y_blue = get_middle_point(mask_blue)

    x_red, y_red = get_middle_point(mask_red)

    mask = np.concatenate((mask_blue, mask_red), axis=1)
    cv2.imshow('mask', mask)

    distance = abs(x_blue - x_red)

    if distance < 100:
        r = requests.get(ip + port + "/?action=/hands_front&text=qwer")

    elif distance >= 100:
        r = requests.get(ip + port + "/?action=/hands_sides&text=open_right")

    print(distance)
    time.sleep(0.01)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam1.release()

cv2.destroyAllWindows()
