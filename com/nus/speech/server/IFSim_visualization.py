import numpy as np
import torch 
import copy

from com.nus.speech.server.utils import angular_distance_compute
from torch.nn.functional import conv2d

class IFSim(object):
	"""
		SNN simulator with IF neurons
		the pre-trained DNN model weights will be used.
	"""

	def __init__(self, Model, opts):
		self.Model = Model
		self.opts = opts

	def test(self, datapoint):
		mae = 0
		mae_history = []

		model_layers = self.Model.parseLayers() # initialize the model layers
		num_layers = len(model_layers)
		num_time_step = int(self.opts['duration']/self.opts['dt']) + num_layers

		test_x, test_y = datapoint

		batch_size = 1
		test_x = torch.from_numpy(test_x).permute(2,1,0).unsqueeze(0)
		test_y = torch.from_numpy(np.array(test_y))

		for t in range(num_time_step):
			for layer in model_layers:
				layer_index = model_layers.index(layer)

				if layer.type == 'conv':
					if layer_index == 0:
						inp_map = test_x/(num_time_step-num_layers)
					else:
						try:
							inp_map = model_layers[layer_index-1].output_spikes[t-1]
						except:
							continue

					# Update the neurons
					filters = layer.weight
					bias = layer.bias/(num_time_step-num_layers)

					# skip updates when no inputs arrive
					if t < layer_index: 
						continue

					# init the layer
					elif t == layer_index: 
						result = conv2d(inp_map, filters, bias, stride=2, padding=1)
						layer.membrane = result
						layer.mem_aggregate = copy.copy(result)
						# check for spiking
						bool_spike = torch.gt(layer.membrane, self.opts['threshold']).float()
						# reset by substraction
						layer.membrane -=  bool_spike*self.opts['threshold']
						# init the spike count for further analysis
						layer.spikes = bool_spike
						# update the layer output
						layer.output_spikes[t] = copy.deepcopy(bool_spike)

					# update the layer 
					elif t < (num_time_step-num_layers+layer_index):
						result = conv2d(inp_map, filters, bias, stride=2, padding=1)
						layer.membrane += result
						layer.mem_aggregate += copy.copy(result)

						# check for spiking
						bool_spike = torch.gt(layer.membrane, self.opts['threshold']).float()
						# reset by substraction
						layer.membrane -=  bool_spike*self.opts['threshold']
						# update the spike count for further analysis
						layer.spikes += bool_spike
						# update the layer output
						layer.output_spikes[t] = copy.deepcopy(bool_spike)

					# skip update once all information is delivered	
					else:
						continue

				elif layer.type == 'linear':
					try:
						inp_map = model_layers[layer_index-1].output_spikes[t-1]
						inp_map = inp_map.view(batch_size, -1)
					except:
						continue

					# skip updates when no inputs arrive
					if t < layer_index: 
						continue

					# init the layer
					elif t == layer_index:						
						layer.membrane = torch.matmul(inp_map, layer.weight) + \
											layer.bias/(num_time_step-num_layers)
						layer.mem_aggregate = copy.copy(layer.membrane)
						layer.output_spikes[t] = copy.deepcopy(layer.membrane)

					# update the layer 
					elif t < (num_time_step-num_layers+layer_index):
						result = torch.matmul(inp_map, layer.weight) + \
									layer.bias/(num_time_step-num_layers)
						layer.membrane += result
						layer.mem_aggregate += copy.copy(result)
						layer.output_spikes[t] = copy.deepcopy(layer.mem_aggregate)					
						# update the output
						output_mem = layer.mem_aggregate

			# return model_layers
			if t == num_time_step - 1:
				loc_pred = torch.argmax(output_mem, dim=1)
				test_y = torch.argmax(test_y, dim=1)
				print(loc_pred, test_y)
				mae_batch = angular_distance_compute(loc_pred, test_y)
				print(f'MAE: {mae_batch.data:5.2f}')		
				#break

		return model_layers, test_y, output_mem

		# mae_full = mae/num_test_batch
		# print(f'Test Accuracy on Full Test Set: {mae_full.data}')		
		
		# return model_layers, mae_history


