# Qwen2-VL Demo
1. This demo demonstrates how to deploy the Qwen2-VL-2B/7B model. The Vision + Projector component is exported as an RKNN model using the `rknn-toolkit2`, while the LLM component is exported as an RKLLM model using the `rkllm-toolkit`.
2. The open-source model used in this demo is available at: [Qwen2-VL-2B](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct), [Qwen2-VL-7B](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct)

## 1. Requirements
```
rkllm-toolkit==1.2.x
rknn-toolkit2>=2.3.2
python==3.8
```

rknn-toolkit2 installation guide：

pip install rknn-toolkit2 -i https://mirrors.aliyun.com/pypi/simple

## 2. HuggingFace Demo

```bash
# modify the modelpath in infer.py
cd examples/Qwen2-VL_Demo
python infer.py
# expect results:
["The image depicts an astronaut in a white spacesuit, reclining on a green chair with his feet up. He is holding a green beer bottle in his right hand. The astronaut is on a moon-like surface, with the Earth visible in the background. The scene is set against a backdrop of stars and the moon's surface, creating a surreal and whimsical atmosphere."]
```

## 3. Model Conversion
- ### convert to onnx

1. Export the Vision + Projector component of the Qwen2-VL model to an ONNX model using the `export/export_vision.py` script.

2. Since RKNN currently supports only `float32`, if the data type is restricted when loading weights, you need to set the `"use_flash_attn"` parameter in `config.json` to `false`.

```bash
# First time to generate cu_seqlens and rotary_pos_emb, need to set 'step' to 1
python export/export_vision.py --step 1 --path /path/to/Qwen2-VL-model --batch 1 --height 392 --width 392
# Second time to export onnx model, need to set 'step' to any number except 1
python export/export_vision.py --step 0 --path /path/to/Qwen2-VL-model --savepath /path/to/save/qwen2-vl-vision.onnx --batch 1 --height 392 --width 392
```

Note: if change the batch, height and width, the cu_seqlens and rotary_pos_emb must be re-generated.

- ### convert to rknn

1. After successfully exporting the ONNX model, you can use the `export/export_vision_rknn.py` script along with the `rknn-toolkit2` tool to convert the ONNX model to an RKNN model.

```bash
python export/export_vision_rknn.py --path /path/to/save/qwen2-vl-vision.onnx --target-platform rk3588
```

- ### convert to rkllm

1. We collected 20 image-text examples from the MMBench_DEV_EN dataset, stored in `data/datasets.json` and `data/datasets`. To use these data, you first need to create `input_embeds` for quantizing the RKLLM model. Run the following code to generate `data/inputs.json`.

```bash
python data/make_input_embeds_for_quantize.py --path /path/to/Qwen2-VL-model
```

2. Use the following code to export the RKLLM model.

```bash
python export/export_rkllm.py --path /path/to/Qwen2-VL-model --target-platform rk3588 --num_npu_core 3 --quantized_dtype w8a8 --device cpu --savepath /path/to/save/qwen2-vl-llm_rk3588.rkllm
```

## 4. C++ Demo
In the `deploy` directory, we provide example code for board-side inference. This code demonstrates the process of "image input to image features," where an input image is processed to output its corresponding image features. These features are then used by the RKLLM model for multimodal content inference.

### 1. Compile and Build
Users can directly compile the example code by running the `deploy/build-linux.sh` or `deploy/build-android.sh` script (replacing the cross-compiler path with the actual path). This will generate an `install/demo_Linux_aarch64` folder in the `deploy` directory, containing the executables `imgenc`, `llm`, `demo`, and the `lib` folder.

'IMAGE_HEIGHT', 'IMAGE_WIDTH' and 'EMBED_SIZE' param in src/main.cpp and src/img_encoder.cpp should be set specially. 

For Qwen2-VL-2B, EMBED_SIZE=1536; For Qwen2-VL-7B, EMBED_SIZE=3584.

```bash
cd deploy
# for linux
./build-linux.sh
# for android
./build-android.sh
# push install dir to device
adb push ./install/demo_Linux_aarch64 /data
# push model file to device
adb push qwen2-vl-vision_rk3588.rknn /data/models
adb push qwen2-vl-llm_rk3588.rkllm /data/models
```

### 2. Run Demo
Enter the `/data/demo_Linux_aarch64` directory on the board and run the example using the following code

```bash
adb shell
cd /data/demo_Linux_aarch64
# export lib path
export LD_LIBRARY_PATH=./lib
# soft link models dir
ln -s /data/models .
# run imgenc
./imgenc models/qwen2-vl-vision_rk3588.rknn demo.jpg 3
# run llm(Pure Text Example)
./llm models/qwen2-vl-llm_rk3588.rkllm 128 512
# run demo(Multimodal Example)
./demo demo.jpg models/qwen2-vl-vision_rk3588.rknn models/qwen2-vl-llm_rk3588.rkllm 128 512 3
```

Note: max_context_len must be larger than text-token-num+image-token-num+max_new_tokens

The user can view the relevant runtime logs in the terminal and obtain the `img_vec.bin` file in the current directory, which contains the image features corresponding to the input image.

Multimodal Example

```
user: <image>What is in the image?
robot: The image depicts an astronaut on the moon, enjoying a beer. The background shows the Earth and stars, creating a surreal and futuristic scene.
```

Pure Text Example
```
user: 把这句话翻译成英文: RK3588是新一代高端处理器，具有高算力、低功耗、超强多媒体、丰富数据接口等特点
robot: The RK3588 is a new generation of high-end processors with high computational power, low power consumption, strong multimedia capabilities, and rich data interfaces.
```