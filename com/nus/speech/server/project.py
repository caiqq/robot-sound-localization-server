import torch
import numpy as np
import os
from com.nus.speech.server.model.sl_inference_robot import sMLP, EncodeSpiking
from com.nus.speech.server.RespeakerTrans import Listener
from com.nus.speech.server.model.lib import angular_distance_compute, testing

import com.nus.speech.server.config as cfg


class Project(object):

    def __init__(self):
        print("init project")
        self.step = 30
        self.Tsim = 10
        self.device = 'cpu'
        model = sMLP(self.Tsim)
        self.model = model.to(self.device)
        self.listener = Listener()
        self.encode_spiking = EncodeSpiking()

        print("load model")
        abs_path = os.path.abspath(__file__).split('/')
        self.path = ""
        for i in range(len(abs_path) - 1):
            self.path = self.path + abs_path[i]
            self.path = self.path + "/"
        print(self.path)
        self.checkpoint = torch.load(self.path + "{0}.pt.tar".format("best"))
        self.model.load_state_dict(self.checkpoint['model_state_dict'])  # restore the model with best_loss
        self.step = 30

        self.spiking_01 = np.full((10, 1000), 0)
        self.spiking_02 = np.full((10, 1000), 0)
        self.spiking_03 = np.full((10, 1000), 0)
        self.y_pred = np.full((360), 0)
        self.desition = 0,
        self.hist_copy = np.full((12), 0)

    def get_calculate_results(self):
        print("get test results")
        change_flag = False
        audio_data = self.listener.get_audio_data()
        if len(audio_data) > 0:
            encode_spiking = self.encode_spiking.encode_robot_read(audio_data)
            x_spike1, x_spike2, x_spike3, out, y_pred = testing(self.model, encode_spiking, self.device)

            # Result Analysis
            angle_pred = torch.argmax(y_pred, 1)

            bin_range = np.arange((0 - self.step / 2), 360 + (self.step / 2) + 1, self.step)

            hist, bin = np.histogram(angle_pred, bin_range)
            hist_copy = hist[0:12]
            hist_copy[0] = hist_copy[0] + hist[12]

            decision = bin_range[np.argmax(hist_copy)] + self.step / 2

            change_flag = True

            return {cfg.grid_conv_1: x_spike1[-1].tolist(),
                    cfg.grid_conv_2: x_spike2[-1].tolist(),
                    cfg.grid_conv_3: x_spike3[-1].tolist(),
                    cfg.grid_conv_4: y_pred[-1].tolist(),
                    cfg.location: decision,
                    cfg.locationBins: hist_copy.tolist(),
                    cfg.change_flag: change_flag}
        else:
            return {cfg.grid_conv_1: self.spiking_01[-1].tolist(),
                    cfg.grid_conv_2: self.spiking_02[-1].tolist(),
                    cfg.grid_conv_3: self.spiking_03[-1].tolist(),
                    cfg.grid_conv_4: self.y_pred[-1].tolist(),
                    cfg.location: self.decision,
                    cfg.locationBins: self.hist_copy.tolist(),
                    cfg.change_flag: change_flag}