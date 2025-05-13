from rknn.api import RKNN
import numpy as np
import os
import argparse

argparse = argparse.ArgumentParser()
argparse.add_argument('--path', type=str, default='qwen2-vl-2b/qwen2_vl_2b_vision.onnx', help='model path', required=False)
argparse.add_argument('--target-platform', type=str, default='rk3588', help='target platform', required=False)
args = argparse.parse_args()

model_path = args.path
target_platform = args.target_platform

rknn = RKNN(verbose=False)
rknn.config(target_platform=target_platform, mean_values=[[0.48145466 * 255, 0.4578275 * 255, 0.40821073 * 255]], std_values=[[0.26862954 * 255, 0.26130258 * 255, 0.27577711 * 255]])
rknn.load_onnx(model_path)
rknn.build(do_quantization=False, dataset=None)
os.makedirs("rknn", exist_ok=True)
rknn.export_rknn("./rknn/" + os.path.splitext(os.path.basename(model_path))[0] + "_{}.rknn".format(target_platform))
