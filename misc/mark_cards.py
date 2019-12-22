import cv2
import time
import os
import math
import sys

sys.path.append("/Users/elijah/Dropbox/Programming/detectors/modules/")

from input_output import Source
import detectors
import tracker

X_WIND = 640
Y_WIND = 480

def main ():
    source = Source ("1")
    #templ  = Source ("1.jpg")
    #template = templ.get_frame ()

    cv2.namedWindow ("ff", cv2.WINDOW_NORMAL)
    cv2.resizeWindow ("ff", (X_WIND, Y_WIND))

    x0_cut = 0
    y0_cut = 0
    x1_cut = X_WIND
    y1_cut = Y_WIND

    crop_mode = False

    cut_num = 0

    frozen    = source.get_frame ()
    frozen    = source.get_frame ()
    out_frame = frozen

    while (True):
        frame = source.get_frame ()
        
        if (crop_mode == True):
            out_frame = frozen.copy ()

        else:
            out_frame = frame.copy ()

        cv2.rectangle (out_frame, (x0_cut, y0_cut), (x1_cut, y1_cut), (255, 0, 0), 5)

        #meth = 'cv2.TM_CCOEFF'
        #method = eval(meth)

        #res = cv2.matchTemplate (frame, template, method)
        #min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

        #top_left = max_loc
        #bottom_right = (top_left[0] + 300, top_left[1] + 430)

        #cv2.rectangle (out_frame, top_left, bottom_right, (0, 255, 0), 2)

        cv2.imshow ("ff", out_frame)

        cv2.waitKey (1)
        
        time.sleep (0.02)

        keyb = cv2.waitKey (1) & 0xFF

        if (keyb == ord('z')):
            print ("crop mode")
            crop_mode = True
            frozen = frame

        if (keyb == ord('x') and crop_mode == True):
            cut = frozen [y0_cut:y1_cut, x0_cut:x1_cut]
            cv2.imwrite (str (cut_num) + ".jpg", cut)
            time.sleep (1)
            cut_num += 1
            crop_mode = False

        step = 10

        if (keyb == ord('w') and y0_cut > 0):
            y0_cut -= step

        if (keyb == ord('a') and x0_cut > 0):
            x0_cut -= step

        if (keyb == ord('s') and y0_cut < y1_cut):
            y0_cut += step

        if (keyb == ord('d') and x0_cut < x1_cut):
            x0_cut += step

        if (keyb == ord('i') and y1_cut > y0_cut):
            y1_cut -= step

        if (keyb == ord('j') and x1_cut > x0_cut):
            x1_cut -= step

        if (keyb == ord('k') and y1_cut < Y_WIND):
            y1_cut += step

        if (keyb == ord('l') and x1_cut < X_WIND):
            x1_cut += step
        
        if (keyb == ord('q')):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main ()