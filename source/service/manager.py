import service.fsm as fsm
from common import *
from time import time, sleep
from service.value_tracker import Value_tracker
import service.input_output as input_output

class Manager:
    def __init__ (self, config_ = "", silent_mode_ = True, time_to_not_silent_ = 0, color_ = 190, draw_tracker_ = True,
                  show_fps_ = True):
        self.inputs = {}
        self.robots_list = {}
        self.silent_mode = silent_mode_
        self.time_to_not_silent = time_to_not_silent_
        self.color = color_
        self.quit = False
        self.draw_tracker = draw_tracker_

        self.ticks_story = []
        self.averaging_window = 15
        self.show_fps = show_fps_

        self.init_time = time()

    def __del__ (self):
        self.logfile.close ()
        cv2.destroyAllWindows()

    def create_window (self, WIND_X, WIND_Y):
        self.WIND_X = WIND_X
        self.WIND_Y = WIND_Y

        cv2.namedWindow("remote_controller", cv2.WINDOW_AUTOSIZE)
        cv2.resizeWindow("remote_controller", (WIND_Y, WIND_X))
        self.canvas = np.ones ((WIND_Y, WIND_X, 3), np.uint8) * self.color

    def init (self):
        self.curr_time = time()
        self.logfile = open("log/" + str(self.curr_time) + ".txt", "w+")
        self.tracker = Value_tracker (self.draw_tracker)

        self.fsm_processor = fsm.FSM_processor ()
        self.start_time = self.curr_time

    def add_inputs (self, inputs):
        self.inputs.update (inputs)

    def add_robots (self, robots):
        self.robots_list.update (robots)

    def form_output_image (self, window_x_sz = -1, one_img_x_sz = -1):
        result = input_output.form_grid (self.output_images, window_x_sz, one_img_x_sz)

        return result

    def handle_modalities (self):
        self.output_images = []
        self.output_names  = []

        self.inputs["computer keyboard"][0]._read_data()
        keyboard_data = self.inputs["computer keyboard"][0].get_read_data()

        #print ("kb data:",keyboard_data)

        if (keyboard_data == ord("q")):
            self.quit = True

        if (keyboard_data == ord("-")):
            self.silent_mode = not self.silent_mode

        if (self.curr_time - self.start_time >= self.time_to_not_silent):
            self.silent_mode = False
            self.time_to_not_silent = 100000

        modalities_data = {}

        for modality in self.inputs.keys ():
            modalities_data.update ({modality : []})

            skip_reading_data = False

            if (modality == "computer keyboard"):
                skip_reading_data = True

            #commands_pack_len = 1

            #needs to be refactored by adding property of being read "completely" to the modalities
            #if (modality == "angles"):
            #if (modality):
            #    commands_pack_len = self.inputs [modality] [0].data_length ()

            commands_pack_len = self.inputs [modality] [0].get_available_data_len ()

            for i in range (commands_pack_len):
                command = self.inputs [modality] [0].get_command (skip_reading_data)

                # print ("command", command)

                self.logfile.write (str (self.curr_time) + str (command))

                action = self.fsm_processor.handle_command (command)
                modalities_data [modality].append (action)

            modality_frames = self.inputs [modality] [0].draw (self.canvas)

            #print ("shape", modality, modality_frames [0].shape [0])

            if (modality_frames [0].shape [0] > 1):
                self.output_images += modality_frames
                self.output_names.append (modality)

        if (self.silent_mode == False):
            for robot_key in self.robots_list.keys():
                commands_pack_num = 0

                #print ("robot", robot_key)

                while (True):
                    added = False

                    commands_pack = [[]]

                    for modality in self.inputs.keys ():
                        if (robot_key in self.inputs [modality] [1] and
                            commands_pack_num < len (modalities_data [modality])):
                            for c in modalities_data [modality] [commands_pack_num] [0]:
                                commands_pack [0].append (c)

                            added = True
                            #print ("added from modality", modality)

                    if (added == False):
                        break

                    print ("commands_pack", commands_pack)

                    self.robots_list[robot_key].add_action (commands_pack)
                    commands_pack_num += 1

            # for key in self.inputs[modality][1]:
            #     if (key in self.robots_list.keys()):
            #         # print ("adding action", key, action)
            #         self.robots_list[key].add_action(action)

    def handle_robots (self):
        self.canvas = np.ones ((self.WIND_Y, self.WIND_X, 3), np.uint8) * self.color
        canvas_ = self.canvas.copy ()

        if (self.draw_tracker == True):
            self.output_images += self.tracker.draw (self.canvas)

        if (self.silent_mode == False):
            for key in self.robots_list.keys ():
                # print(key)
                self.robots_list [key].on_idle ()

        for key in self.robots_list.keys ():
            #print (key)
            robot_canvas = self.robots_list [key].plot_state (canvas_, 150, 40, 2.5)
            self.output_images.append (robot_canvas)
            self.output_names.append  (key)

    def on_idle (self):
        new_time = time ()

        if (self.show_fps == True):
            tick_time = new_time - self.curr_time
            self.ticks_story.append (tick_time)
            tick_time_windowed = self.ticks_story [max (0, len (self.ticks_story) - self.averaging_window) :]
            avg_tick_time = np.mean (tick_time_windowed)
            self.tracker.update ("fps", 1 / avg_tick_time)

        self.tracker.update("uptime", new_time - self.init_time)

        self.curr_time = new_time

        self.handle_modalities ()
        self.handle_robots     ()

        if (self.silent_mode == True):
            self.canvas = cv2.putText (self.canvas, "silent mode", (30, 100), cv2.FONT_HERSHEY_SIMPLEX,
                   1, (0, 255, 0), 2, cv2.LINE_AA)

        sleep  (0.002)

        return {"quit" : self.quit}

# class Transfer_manager:
#     def __init__ (self, database_path_):
#         self.database_path = database_path_
