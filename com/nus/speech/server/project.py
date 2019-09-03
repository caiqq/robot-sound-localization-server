import scipy.io as spio
from sklearn.utils.extmath import softmax
from scipy.io.wavfile import write

from com.nus.speech.server.utils import state_dict_data_parallel
from com.nus.speech.server.parseModel_torch import *
from com.nus.speech.server.IFSim_visualization import *
from com.nus.speech.server.model.sl_cnn import Model

import com.nus.speech.server.config as cfg


class Project(object):

    def __init__(self):
        print("init project")
        # path = 'checkpoint/sl_pytorch.pt'
        path = './com/nus/speech/server/checkpoint/sl_pytorch.pt'

        self.model = Model()
        checkpoint = torch.load(path)
        model_state_dict_updated = state_dict_data_parallel(checkpoint['model_state_dict'])
        self.model.load_state_dict(model_state_dict_updated)

        self.model.eval()

        print("load model")

        # define the network structure
        ConvNet_layer_list = ['conv1', 'conv2', 'conv3', 'conv4', 'linear']

        # construct the SNN simulator and perform evaluation on the test data
        self.opts = {}
        self.opts['dt'] = 1e-3
        self.opts['duration'] = 10e-3
        self.opts['threshold'] = 1.0
        self.opts['report_every'] = 10  # results report period (# of time steps)

        # parse the pretrained ANN model
        self.Model = ParseModel(ConvNet_layer_list, self.model)

        self.simulator = IFSim(self.Model, self.opts)

        self.image_per_row_1 = 10
        self.n_feature_1 = 12
        self.size_h_1 = 19
        self.size_w_1 = 25
        self.margin_1 = 10

        self.image_per_row_2 = 12
        self.n_feature_2 = 24
        self.size_h_2 = 9
        self.size_w_2 = 12
        self.margin_2 = 2

        self.image_per_row_3 = 12
        self.n_feature_3 = 48
        self.size_h_3 = 4
        self.size_w_3 = 5
        self.margin_3 = 2

        self.n_feature_4 = 96

    def get_target_location(self, file_path):
        rate = 48000
        print("get target location")
        mat = spio.loadmat(file_path)
        wav, x, y = mat['wv'], mat['X'], mat['Y']  # Dim: (8192, 4) (51, 40, 6)
        result = np.where(y[0] == 1)
        wav1 = []
        wav2 = []
        wav3 = []
        wav4 = []
        for subWav in wav:
            wav1.append(subWav[0])
            wav2.append(subWav[1])
            wav3.append(subWav[2])
            wav4.append(subWav[3])
        audio1 = "audio1.wav"
        write(audio1, rate, np.array(wav1))
        audio2 = "audio2.wav"
        write(audio2, rate, np.array(wav2))
        audio3 = "audio3.wav"
        write(audio3, rate, np.array(wav3))
        audio4 = "audio4.wav"
        write(audio4, rate, np.array(wav4))
        return {cfg.location: str(result[0][0]),
                cfg.audio_files: wav.tolist()}

    def get_calculate_results(self, file_path, args):
        print("get test results")

        # load test data
        mat = spio.loadmat(file_path)
        wav, x, y = mat['wv'], mat['X'], mat['Y']  # Dim: (8192, 4) (51, 40, 6)
        model_layers_updated, test_y, out_mem = self.simulator.test((x, y))

        # layer_1
        iFeature = 0
        T = int(self.opts['duration'] / self.opts['dt'])

        iLayer = 0
        display_grid_conv1 = np.zeros(
            (T, self.size_h_1, self.size_w_1 * self.n_feature_1 + (self.n_feature_1 - 1) * self.margin_1))

        display_grid_conv1_1 = np.zeros((T, self.n_feature_1, self.size_h_1, self.size_w_1))
        for t in range(iLayer, T):
            for iFeature in range(self.n_feature_1):
                out = model_layers_updated[0].output_spikes[t].squeeze()
                out_image = out[iFeature, :, :].squeeze().cpu().detach().numpy()
                display_grid_conv1[t, 0: self.size_h_1, iFeature * self.size_w_1 + iFeature * self.margin_1: (iFeature + 1) * self.size_w_1 + iFeature * self.margin_1] = out_image

                display_grid_conv1_1[t, iFeature, :, :] = out_image

        # layer_2
        n_row = self.n_feature_2 // self.image_per_row_2
        T = int(self.opts['duration'] / self.opts['dt'])

        iLayer = 1
        # divide feature map into two rows
        display_grid_conv2 = np.zeros(
            (T, self.size_h_2 * n_row + self.margin_2,
             self.size_w_2 * self.image_per_row_2 + (self.image_per_row_2 - 1) * self.margin_2))

        display_grid_conv2_1 = np.zeros((T, self.n_feature_2, self.size_h_2, self.size_w_2))

        for t in range(iLayer, T):
            for iFeature in range(self.n_feature_2):
                out = model_layers_updated[iLayer].output_spikes[t].squeeze()
                out_image = out[iFeature, :, :].squeeze().cpu().detach().numpy()
                iRow = iFeature // self.image_per_row_2
                iCol = iFeature % self.image_per_row_2
                display_grid_conv2[t, iRow * (self.size_h_2) + iRow * self.margin_2:(iRow + 1) * self.size_h_2 + iRow * self.margin_2, \
                iCol * self.size_w_2 + iCol * self.margin_2: (iCol + 1) * self.size_w_2 + iCol * self.margin_2] = out_image

                display_grid_conv2_1[t, iFeature, :, :] = out_image

        # layer3
        n_row = self.n_feature_3 // self.image_per_row_3

        T = int(self.opts['duration'] / self.opts['dt'])
        display_grid_conv3 = np.zeros(
            (T, self.size_h_3 * n_row + self.margin_3 * (n_row - 1),
             self.size_w_3 * self.image_per_row_3 + self.margin_3 * (self.image_per_row_3 - 1)))

        display_grid_conv3_1 = np.zeros((T, self.n_feature_3, self.size_h_3, self.size_w_3))
        iLayer = 2
        for t in range(iLayer, T):
            for iFeature in range(self.n_feature_3):
                out = model_layers_updated[iLayer].output_spikes[t].squeeze()
                out_image = out[iFeature, :, :].squeeze().cpu().detach().numpy()
                iRow = iFeature // self.image_per_row_3
                iCol = iFeature % self.image_per_row_3
                display_grid_conv3[t,
                iRow * (self.size_h_3) + iRow * self.margin_3:(iRow + 1) * self.size_h_3 + iRow * self.margin_3, \
                iCol * self.size_w_3 + iCol * self.margin_3: (
                                                                     iCol + 1) * self.size_w_3 + iCol * self.margin_3] = out_image

                display_grid_conv3_1[t, iFeature, :, :] = out_image
        # layer4

        T = int(self.opts['duration'] / self.opts['dt'])
        display_grid_conv4 = np.zeros((T, 12, 16))

        iLayer = 3
        for t in range(iLayer, T):
            for iFeature in range(self.n_feature_4):
                out = model_layers_updated[iLayer].output_spikes[t].squeeze()
                out_image = out.view(-1).cpu().detach().numpy()
                display_grid_conv4[t, :] = out_image.reshape(12, 16)

        print("out layer")
        # output layer

        # T = int(self.opts['duration'] / self.opts['dt'])
        # layer_act = np.zeros((T, 360))
        # iLayer = 4
        #
        # for t in range(iLayer, T):
        #     out_prob = softmax(model_layers_updated[iLayer].output_spikes[t]).squeeze()
        #     layer_act[t, :] = out_prob

        T = int(self.opts['duration'] / self.opts['dt'])
        display_grid_out = np.zeros((T, 360))
        max_out_list = np.zeros((T))

        iLayer = 4
        for t in range(iLayer, T):
            out_prob = softmax(model_layers_updated[iLayer].output_spikes[t]).squeeze()
            out_prob_1 = softmax(model_layers_updated[iLayer].output_spikes[t]).squeeze()
            display_grid_out[t, :] = out_prob_1
            max_out_list[t] = np.argmax(out_prob_1, axis=0)

        print("conv1 shape: ", display_grid_conv1.shape)
        print("conv1_1 shape: ", display_grid_conv1_1.shape)

        return {cfg.display_grid_conv_1: display_grid_conv1.tolist(),
                cfg.display_grid_conv_1_1: display_grid_conv1_1.tolist(),
                cfg.display_grid_conv_2: display_grid_conv2.tolist(),
                cfg.display_grid_conv_2_1: display_grid_conv2_1.tolist(),
                cfg.display_grid_conv_3: display_grid_conv3.tolist(),
                cfg.display_grid_conv_3_1: display_grid_conv3_1.tolist(),
                cfg.display_grid_conv_4: display_grid_conv4.tolist(),
                cfg.display_grid_out: display_grid_out.tolist(),
                cfg.max_out_list: max_out_list.tolist(),
                cfg.out_prob: out_prob.tolist(),
                cfg.max_out: str(np.argmax(out_prob, axis=0))}

        # return {cfg.display_grid_conv_1: display_grid_conv1_1.tolist(),
        #         cfg.display_grid_conv_2: display_grid_conv2_1.tolist(),
        #         cfg.display_grid_conv_3: display_grid_conv3_1.tolist(),
        #         cfg.display_grid_conv_4: display_grid_conv4.tolist(),
        #         cfg.out_prob: display_grid_out.tolist(),
        #         cfg.max_out: max_out_list.tolist()}
