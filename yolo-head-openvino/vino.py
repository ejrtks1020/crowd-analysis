from openvino.runtime import Core
import numpy as np
model_path = "./yolo-tiny-head.xml"
ie = Core()

network = ie.read_model(model=model_path, weights="yolo-tiny-head.bin")
executable_network = ie.compile_model(model=network, device_name="CPU")

im = np.random.randn(1,3,640, 640)
output_layer = next(iter(executable_network.outputs))
y = executable_network([im])[output_layer]
print(executable_network.outputs)
print(next(iter(executable_network.outputs)))
print(y.shape)
for i,(batch_id,x0,y0,x1,y1,cls_id,score) in enumerate(y):
    print(cls_id, score, x0, y0, x1, y1)
