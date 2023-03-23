from openvino.runtime import Core
import numpy as np
model_path = "./det_2.5g.xml"
ie = Core()

network = ie.read_model(model=model_path, weights="det_2.5g.bin")
executable_network = ie.compile_model(model=network, device_name="CPU")

im = np.random.randn(1,3,640, 640)
print(len(executable_network([im]).values()))
#for i, output_layer in enumerate(executable_network.outputs):
#    print(i, output_layer)
#    y = executable_network([im])[output_layer]
#    print(y.shape)
#output_layer = executable_network.outputs[1]
#y = executable_network([im])[output_layer]
#print(next(iter(executable_network.outputs)))
#print(y.shape)
