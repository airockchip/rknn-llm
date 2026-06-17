from rknn.api import RKNN
import numpy as np
import os
import argparse

argparse = argparse.ArgumentParser()
argparse.add_argument('--path', type=str, default='./onnx/qwen2_5-vl-3b_vision.onnx', help='model path', required=False)
argparse.add_argument('--model_name', type=str, default='qwen2_5-vl-3b',
                    choices=['minicpm-v-2_6', 'qwen2_5-vl-3b', 'qwen3-vl', 'qwen3.5', 'smolvlm', 'internvl3-1b', 'deepseekocr'],
                    help='model name', required=True)
argparse.add_argument('--target-platform', type=str, default='rk3588', help='target platform', required=False)
argparse.add_argument('--batch_size', type=int, default=1, help='batch size', required=False)
argparse.add_argument('--height', type=int, default=448, help='image height', required=False)
argparse.add_argument('--width', type=int, default=448, help='image width', required=False)

args = argparse.parse_args()

model_path = args.path
target_platform = args.target_platform
modelname = args.model_name

if 'qwen2' in model_path.lower():
    mean_value = [[0.48145466 * 255, 0.4578275 * 255, 0.40821073 * 255]]
    std_value = [[0.26862954 * 255, 0.26130258 * 255, 0.27577711 * 255]]
elif 'internvl3' in model_path.lower():
    mean_value = [[0.485 * 255, 0.456 * 255, 0.406 * 255]]
    std_value = [[0.229 * 255, 0.224 * 255, 0.225 * 255]]
else:
    mean_value = [[0.5 * 255, 0.5 * 255, 0.5 * 255]]
    std_value = [[0.5 * 255, 0.5 * 255, 0.5 * 255]]

if modelname == 'qwen2_5-vl-3b':
    inputs = ['pixel', 'grid_thw']
    input_size_list = [[args.batch_size, 3, args.height, args.width], [1,3]]
    grid_t = args.batch_size//2 if args.batch_size % 2 == 0 else (args.batch_size + 1)//2
    input_initial_val = [None, np.array([[grid_t, args.height//14, args.width//14]], dtype=np.int64)]
    op_target = {"/vpm/patch_embed/proj/Conv_output_0_conv_tp_sw": 'cpu'}
elif modelname == 'qwen3-vl' or modelname == 'qwen3.5':
    inputs = ['pixel', 'grid_thw']
    input_size_list = [[args.batch_size, 3, args.height, args.width], [1,3]]
    grid_t = args.batch_size//2 if args.batch_size % 2 == 0 else (args.batch_size + 1)//2
    input_initial_val = [None, np.array([[grid_t, args.height//16, args.width//16]], dtype=np.int64)]
    op_target = None
else:
    inputs = ['pixel']
    input_size_list = [[args.batch_size, 3, args.height, args.width]]
    input_initial_val = None
    op_target = None

if modelname == 'deepseekocr':
    disable_rules=['convert_rs_add_rs_to_rs_gather_elements']
else:
    disable_rules=[]

rknn = RKNN(verbose=False)
rknn.config(disable_rules=disable_rules, target_platform=target_platform, mean_values=mean_value, std_values=std_value, op_target=op_target)
rknn.load_onnx(model_path, inputs=inputs, input_size_list=input_size_list, input_initial_val=input_initial_val)
rknn.build(do_quantization=False, dataset=None)
os.makedirs("rknn", exist_ok=True)
rknn.export_rknn("./rknn/" + os.path.splitext(os.path.basename(model_path))[0] + "_{}.rknn".format(target_platform))
