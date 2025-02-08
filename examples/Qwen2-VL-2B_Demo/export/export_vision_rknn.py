from rknn.api import RKNN
import numpy as np
import os

model_path = "./onnx/qwen2_vl_2b_vision.onnx"
target_platform = "rk3588"

rknn = RKNN(verbose=False)
rknn.config(target_platform=target_platform, mean_values=[[0.48145466 * 255, 0.4578275 * 255, 0.40821073 * 255]], std_values=[[0.26862954 * 255, 0.26130258 * 255, 0.27577711 * 255]])
rknn.load_onnx(model_path)
rknn.build(do_quantization=False, dataset=None)
os.makedirs("rknn", exist_ok=True)
rknn.export_rknn("./rknn/" + model_path.split("/")[-1].split(".")[0] + "_{}.rknn".format(target_platform))
