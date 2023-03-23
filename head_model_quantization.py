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
	def __init__(self, dataset_path, shape=(640, 640)):
		self.files = glob.glob(dataset_path + "/*")
		self.shape = shape
	
	def __len__(self):
		return len(self.files)
		
	def __getitem__(self, index):
		
		if index >= len(self):
			raise IndexError("Index out of dataset size")
		
		img = cv2.imread(self.files[index])
		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		img_head = img.copy()
		img_head, ratio, dwdh = letterbox(img_head, auto=False)
		img_head = img_head.transpose((2, 0, 1))
		img_head = np.expand_dims(img_head, 0)
		img_head = np.ascontiguousarray(img_head)
		im = img_head.astype(np.float32)
		im /= 255
		return im, None
		
data_loader = ImageLoader("./images")

q_params = [{
	"name" :"DefaultQuantization",
	"params" : {
		"target_device" : "CPU",
		"preset" : "performance",
		"stat_subset_size" : 300},
		}]

model_config = {
	"model_name" : "yolo-tiny-head",
	"model" : "/home/xinapse/ailab-insightface-modified/yolo-head-openvino/yolo-tiny-head.xml",
	"weights" : "/home/xinapse/ailab-insightface-modified/yolo-head-openvino/yolo-tiny-head.bin"
	}
	
engine_config = {"device" : "CPU"}

model = load_model(model_config=model_config)

engine = IEEngine(config=engine_config, data_loader=data_loader)

pipeline = create_pipeline(q_params, engine)

compressed_model = pipeline.run(model=model)

# compress_model_weights(compress_model)

compressed_model_paths = save_model(
	model=compressed_model,
	save_path="yolo_openvino_ptq",
	model_name="quantized_yolov7",)
	

