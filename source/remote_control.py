import modalities
import fsm
import robots
from common import *

def main():
    AUTONOMOUS = True #without robot

    WIND_X = 500
    WIND_Y = 500
    cv2.namedWindow  ("remote_controller", cv2.WINDOW_AUTOSIZE)
    cv2.resizeWindow ("remote_controller", (WIND_Y, WIND_X))
    canvas = np.ones ((WIND_Y, WIND_X, 3), np.uint8) * 230
        
    #logfile = open ("log/" + str (curr_time) + ".txt", "w+")
    
    inputs = {"computer keyboard" : modalities.Computer_keyboard ()}

    robot = robots.Simulated_robot ()

    if (AUTONOMOUS == False):
        robot = robots.Real_robot ("10.0.0.105", "9569")

    fsm_processor = fsm.FSM_processor ()

    while (True):
        inputs ["computer keyboard"]._read_data ()

        keyboard_data = inputs ["computer keyboard"].get_read_data ()
        if (keyboard_data == ord ("q")):
            break

        #change task, go to next question

        for modality in inputs.keys ():
            skip_reading_data = False

            if (modality == "computer keyboard"):
                skip_reading_data = True

            command = inputs [modality].get_command (skip_reading_data)

            #print ("command: ", command)

            action = fsm_processor.handle_command (command)
            #print ("action", action)

            robot.add_action (action)

        robot.on_idle ()

        canvas = np.ones ((WIND_Y, WIND_X, 3), np.uint8) * 200
        
        robot.plot_state (canvas, 200, 200)

        cv2.imshow ("remote_controller", canvas)
        
        time.sleep  (0.3)        
            
    #logfile.close ()
    cv2.destroyAllWindows ()

if __name__ == '__main__':
    main()
