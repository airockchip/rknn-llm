# Janus-Pro-1B Demo
1. This demo demonstrates how to deploy the Janus-Pro-1B model. The Vision + Projector component is exported as an RKNN model using the `rknn-toolkit2`, while the LLM component is exported as an RKLLM model using the `rkllm-toolkit`.
2. The open-source model used in this demo is available at: [Janus-Pro-1B](https://huggingface.co/deepseek-ai/Janus-Pro-1B)

## 1. Requirements
```
rkllm-toolkit==1.1.4
rknn-toolkit2==2.2.1
python==3.8
```

rknn-toolkit2 installation guide：

pip install rknn-toolkit2==2.2.1 -i https://mirrors.aliyun.com/pypi/simple

install janus env: 

```shell
git clone https://github.com/deepseek-ai/Janus.git`
 
cd Janus

pip install -e .
```

install onnx:

```shell
pip install onnxruntime onnx
```

## 2. HuggingFace Demo

```
1、modify the modelpath in infer.py
2、python infer.py
3、expect results:
[" 这张图片展示了一个宇航员在月球表面休息的场景。宇航员身穿白色太空服，头戴头盔，头盔上有反光镜片。他正躺在月球表面，双腿伸直，脚上穿着宇航服靴子，手里拿着一瓶绿色的啤酒，似乎在享受片刻的休息。

背景中可以看到地球，它被月球表面的坑洞所环绕，显得格外壮观。月球表面布满了陨石坑和坑洞，看起来非常荒凉。宇航员旁边有一个绿色的箱子，上面有“Binty's”字样，可能是宇航员用来存放物品的。

整个场景给人一种幽默和奇幻的感觉，仿佛宇航员在月球上悠闲地享受着一瓶啤酒，而不是在执行任务。这种构图和场景设计可能意在创造一种轻松愉快的氛围，同时又带有一些科幻的元素。"]
```

## 3. Model Conversion
- ### convert to onnx

1. Export the Vision + Projector component of the Janus-Pro-1B model to an ONNX model using the `export/export_vision.py` script.

2. Since RKNN currently supports only `float32`, if the data type is restricted when loading weights, you need to set the `"use_flash_attn"` parameter in `config.json` to `false`.

```bash
python export/export_vision.py
```

- ### convert to rknn

1. After successfully exporting the ONNX model, you can use the `export/export_vision_rknn.py` script along with the `rknn-toolkit2` tool to convert the ONNX model to an RKNN model.

```bash
python export/export_vision_rknn.py
```

- ### Export the LLM component

Janus-Pro-1B model LLM component to an Huggingface model using the `export/export_llm.py` script.

```shell
python export/export_llm.py
```

- ### convert to rkllm component

1. We collected 20 image-text examples from the MMBench_DEV_EN dataset, stored in `data/datasets.json` and `data/datasets`. To use these data, you first need to create `input_embeds` for quantizing the RKLLM model. Run the following code to generate `data/inputs.json`.

```bash
#Modify the Juans-Pro-1B ModelPath in data/make_input_embeds_for_quantize.py, and then
python data/make_input_embeds_for_quantize.py
```

2. Use the following code to export the RKLLM model.

```bash
python export/export_rkllm.py
```

## 4. C++ Demo
In the `deploy` directory, we provide example code for board-side inference. This code demonstrates the process of "image input to image features," where an input image is processed to output its corresponding image features. These features are then used by the RKLLM model for multimodal content inference.

### 1. Compile and Build
Users can directly compile the example code by running the `deploy/build-linux.sh` or `deploy/build-android.sh` script (replacing the cross-compiler path with the actual path). This will generate an `install/demo_Linux_aarch64` folder in the `deploy` directory, containing the executables `imgenc`, `llm`, `demo`, and the `lib` folder.

```bash
cd deploy
# for linux
./build-linux.sh
# for android
./build-android.sh
# push install dir to device
adb push ./install/demo_Linux_aarch64 /data
# push model file to device
adb push Janus_pro_vision_rk3588.rknn /data/models
adb push Janus-Pro-1B-rk3588.rkllm /data/models
# push demo image to device
adb push ../data/demo.jpg /data/demo_Linux_aarch64
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
./imgenc models/Janus_pro_vision_rk3588.rknn demo.jpg
# run llm(Pure Text Example)
./llm models/Janus-Pro-1B-rk3588.rkllm 128 768
# run demo(Multimodal Example)
./demo demo.jpg models/Janus_pro_vision_rk3588.rknn models/Janus-Pro-1B-rk3588.rkllm 128 768
```

The user can view the relevant runtime logs in the terminal and obtain the `img_vec.bin` file in the current directory, which contains the image features corresponding to the input image.

Multimodal Example

```
user: <image>What is in the image?
robot: The image shows an astronaut sitting on the moon's surface while holding a green bottle labeled "Greenpeace." There are also two objects nearby—a cooler box marked as well-known and another object
 that looks like part of some equipment or structure.
```
