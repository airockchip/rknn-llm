from rknn.api import RKNN
import numpy as np
import os

model_path = "./onnx/Janus_pro_vision.onnx"
target_platform = "rk3588"

rknn = RKNN(verbose=False)
rknn.config(
    target_platform=target_platform,
    mean_values=[[0.5 * 255, 0.5 * 255, 0.5 * 255]],
    std_values=[[0.5 * 255, 0.5 * 255, 0.5 * 255]],
    )
rknn.load_onnx(model_path,  inputs=['pixel_values'], input_size_list=[[1,3,384,384]])
rknn.build(do_quantization=False, dataset=None)
os.makedirs("rknn", exist_ok=True)
rknn.export_rknn("./rknn/" + model_path.split("/")[-1].split(".")[0] + "_{}.rknn".format(target_platform))
