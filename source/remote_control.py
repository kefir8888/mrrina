import modalities
import fsm
import robots
from common import *

def main():
    AUTONOMOUS = True #without robot

    WIND_X = 800
    WIND_Y = 500
    cv2.namedWindow  ("remote_controller", cv2.WINDOW_AUTOSIZE)
    cv2.resizeWindow ("remote_controller", (WIND_Y, WIND_X))
    canvas = np.ones ((WIND_Y, WIND_X, 3), np.uint8) * 230
        
    #logfile = open ("log/" + str (curr_time) + ".txt", "w+")
    
    inputs = {"computer keyboard" : modalities.Computer_keyboard (),
              "video input" : modalities.Video()}
              # "archive skeleton"  : modalities.Skeleton ("/home/kompaso/Desktop/ISP/lightweight-human-pose-estimation_2/skel/skel_robot_ponomareva.txt")}

    robots_list = {}

    if (AUTONOMOUS == True):
        robots_list.update ({"simulated" : robots.Simulated_robot ()})
    
    if (AUTONOMOUS == False):
        #ip = "192.168.1.30"
        ip = "10.0.0.104"
        robots_list.update ({"physical" : robots.Real_robot (ip, "9569")})

    fsm_processor = fsm.FSM_processor ()

    while (True):
        inputs ["computer keyboard"]._read_data ()

        keyboard_data = inputs ["computer keyboard"].get_read_data ()
        if (keyboard_data == ord ("q")):
            break

        for modality in inputs.keys ():
            skip_reading_data = False

            if (modality == "computer keyboard"):
                skip_reading_data = True

            command = inputs [modality].get_command (skip_reading_data)

            #print ("modality: ", modality)

            action = fsm_processor.handle_command (command)
            #print ("action", action)

            for key in robots_list.keys ():
                robots_list [key].add_action (action)

        for key in robots_list.keys ():
            robots_list [key].on_idle ()

        canvas = np.ones ((WIND_Y, WIND_X, 3), np.uint8) * 200
        
        list (robots_list.items ()) [0] [1].plot_state (canvas, 150, 40, 2.5)

        cv2.imshow ("remote_controller", canvas)
        
        time.sleep  (0.2)        
            
    #logfile.close ()
    cv2.destroyAllWindows ()

if __name__ == '__main__':
    main()
