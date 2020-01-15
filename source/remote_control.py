import modalities
import fsm
import robots
from common import *

def main():
    AUTONOMOUS = False #without robot

    WIND_X = 800
    WIND_Y = 500
    cv2.namedWindow  ("remote_controller", cv2.WINDOW_AUTOSIZE)
    cv2.resizeWindow ("remote_controller", (WIND_Y, WIND_X))
    canvas = np.ones ((WIND_Y, WIND_X, 3), np.uint8) * 230
        
    #logfile = open ("log/" + str (curr_time) + ".txt", "w+")
    
    inputs = {"computer keyboard" : modalities.Computer_keyboard ()}

    robots_list = {}

    if (AUTONOMOUS == True):
        robots_list.update ({"simulated" : robots.Simulated_robot ()})
    
    if (AUTONOMOUS == False):
        robots_list.update ({"physical" : robots.Real_robot ("192.168.1.30", "9569")})

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

            for key in robots_list.keys ():
                robots_list [key].add_action (action)

        for key in robots_list.keys ():
            robots_list [key].on_idle ()

        canvas = np.ones ((WIND_Y, WIND_X, 3), np.uint8) * 200
        
        robots_list ["simulated"].plot_state (canvas, 150, 40, 2.5)

        cv2.imshow ("remote_controller", canvas)
        
        time.sleep  (0.3)        
            
    #logfile.close ()
    cv2.destroyAllWindows ()

if __name__ == '__main__':
    main()
