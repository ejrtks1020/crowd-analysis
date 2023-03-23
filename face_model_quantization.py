import os
import numpy as np
import cv2
import glob
from utils3 import letterbox
from openvino.tools.pot import DataLoader
from openvino.tools.pot import IEEngine
from openvino.tools.pot import load_model, save_model
from openvino.tools.pot import compress_model_weights
from openvino.tools.pot import create_pipeline

class ImageLoader(DataLoader):
	def __init__(self, dataset_path):
		self.files = glob.glob(dataset_path + "/*")
		self.input_mean = 127.5
		self.input_std = 128.0
	
	def __len__(self):
		return len(self.files)
		
	def __getitem__(self, index):
		
		if index >= len(self):
			raise IndexError("Index out of dataset size")
		
		img = cv2.imread(self.files[index])
		input_size = tuple(img.shape[0:2][::-1])
		im = cv2.dnn.blobFromImage(img, 1.0/self.input_std, input_size, (self.input_mean, self.input_mean, self.input_mean), swapRB=True)
		return im, None
		
data_loader = ImageLoader("./images")
print(len(data_loader))

q_params = [{
	"name" :"DefaultQuantization",
	"params" : {
		"target_device" : "CPU",
		"preset" : "performance",
		"stat_subset_size" : 300},
		}]

model_config = {
	"model_name" : "det_2.5g",
	"model" : "/home/xinapse/ailab-insightface-modified/det_2.5_openvino/det_2.5g.xml",
	"weights" : "/home/xinapse/ailab-insightface-modified/det_2.5_openvino/det_2.5g.bin"
	}
	
engine_config = {"device" : "CPU"}

model = load_model(model_config=model_config)

engine = IEEngine(config=engine_config, data_loader=data_loader)

pipeline = create_pipeline(q_params, engine)

compressed_model = pipeline.run(model=model)

# compress_model_weights(compress_model)

compressed_model_paths = save_model(
	model=compressed_model,
	save_path="quantized_face_det",
	model_name="quantized_face_det",)
	

