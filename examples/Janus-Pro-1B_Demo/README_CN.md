Janus-Pro-1B

本文档展示了如何部署 Janus-Pro-1B 模型。视觉 + 投影器组件使用 rknn-toolkit2 导出为 RKNN 模型，而大语言模型（LLM）组件使用 rkllm-toolkit 导出为 RKLLM 模型。

本文档中使用的开源模型可在以下链接获取：[Janus-Pro-1B](https://huggingface.co/deepseek-ai/Janus-Pro-1B)

## 1. 环境要求

```shell
rkllm-toolkit==1.1.4
rknn-toolkit2==2.2.1
python==3.8
```

rknn-toolkit2 安装指南：

```shell
pip install rknn-toolkit2==2.2.1 -i https://mirrors.aliyun.com/pypi/simple
```

Janus环境安装指南：

```shell
# 克隆仓库
git clone https://github.com/deepseek-ai/Janus.git
# 进入仓库目录
cd Janus
# 安装环境
pip install -e .
```

onnx安装:

```shell
pip install onnxruntime onnx
```

## 2. 运行HuggingFace模型

1、修改 infer.py 中的模型路径
2、运行 python infer.py
3、预期结果:
[" 这张图片展示了一个宇航员在月球表面休息的场景。宇航员身穿白色太空服，头戴头盔，头盔上有反光镜片。他正躺在月球表面，双腿伸直，脚上穿着宇航服靴子，手里拿着一瓶绿色的啤酒，似乎在享受片刻的休息。

背景中可以看到地球，它被月球表面的坑洞所环绕，显得格外壮观。月球表面布满了陨石坑和坑洞，看起来非常荒凉。宇航员旁边有一个绿色的箱子，上面有“Binty's”字样，可能是宇航员用来存放物品的。

整个场景给人一种幽默和奇幻的感觉，仿佛宇航员在月球上悠闲地享受着一瓶啤酒，而不是在执行任务。这种构图和场景设计可能意在创造一种轻松愉快的氛围，同时又带有一些科幻的元素。"]

## 3. 模型转换

### 导出vision和Projectors组件，转成onnx格式

使用 export/export_vision.py 脚本将 Janus-Pro-1B 模型的视觉 + 投影器组件导出为 ONNX 模型。

由于 RKNN 当前仅支持 float32 数据类型，如果在加载权重时数据类型受限，需要将 config.json 中的 "use_flash_attn" 参数设置为 false。

```shell
# 修改 export/export_vision.py 中的模型路径
python export/export_vision.py
```

### 转换为RKNN格式

成功导出 ONNX 模型后，可以使用 export/export_vision_rknn.py 脚本结合 rknn-toolkit2 工具将 ONNX 模型转换为 RKNN 模型。

```shell
# 修改 export/export_vision_rknn.py 中的模型路径
python export/export_vision_rknn.py
```

### 3.导出LLM组件，转成huggingface格式。

把Janus-Pro-1B模型的LLM组件导出为huggingface格式。

```shell
# 修改 export/export_llm.py 中的模型路径
python export/export_llm.py
```

### 4. LLM模型转成RKLLM格式

我们从 MMBench_DEV_EN 数据集中收集了 20 个图像 - 文本示例，存储在 data/datasets.json 和 data/datasets 中。要使用这些数据，首先需要创建 input_embeds 来对 RKLLM 模型进行量化。运行以下代码生成 data/inputs.json。

```shell
# 修改 data/make_input_embeds_for_quantize.py 中的 Juans-Pro-1B 模型路径
python data/make_input_embeds_for_quantize.py
```

使用以下代码导出 RKLLM 模型

```shell
# 修改 data/export_rkllm.py 中的 Juans-Pro-1B 模型路径
python export/export_rkllm.py
```

### 5. C++ 演示

在 deploy 目录中，我们提供了板端推理的示例代码。此代码展示了“图像输入到图像特征”的过程，即输入图像经过处理后输出其对应的图像特征。这些特征随后被 RKLLM 模型用于多模态内容推理。

1. 编译和构建

用户可以直接运行 deploy/build-linux.sh 或 deploy/build-android.sh 脚本（将交叉编译器路径替换为实际路径）来编译示例代码。这将在 deploy 目录中生成一个 install/demo_Linux_aarch64 文件夹，其中包含可执行文件 imgenc、llm、demo 以及 lib 文件夹。

```shell
cd deploy
# 针对 Linux
./build-linux.sh
# 针对 Android
./build-android.sh
# 将安装目录推送到设备
adb push ./install/demo_Linux_aarch64 /data
# 将模型文件推送到设备
adb push Janus_pro_vision_rk3588.rknn /data/models
adb push Janus-Pro-1B-rk3588.rkllm /data/models
# 将演示图像推送到设备
adb push ../data/demo.jpg /data/demo_Linux_aarch64
```

2. 运行演示

进入板端的 /data/demo_Linux_aarch64 目录，并使用以下代码运行示例

```shell
adb shell
cd /data/demo_Linux_aarch64
# 导出库路径
export LD_LIBRARY_PATH=./lib
# 创建模型目录的软链接
ln -s /data/models .
# 运行 imgenc
./imgenc models/Janus_pro_vision_rk3588.rknn demo.jpg
# 运行 llm（纯文本示例）
./llm models/Janus-Pro-1B-rk3588.rkllm 128 768
# 运行 demo（多模态示例）
./demo demo.jpg models/Janus_pro_vision_rk3588.rknn models/Janus-Pro-1B-rk3588.rkllm 128 768
```

用户可以在终端中查看相关的运行时日志，并在当前目录中获取 img_vec.bin 文件，该文件包含输入图像对应的图像特征。

多模态示例

```
user: <image>What is in the image?
robot: The image shows an astronaut sitting on the moon's surface while holding a green bottle labeled "Greenpeace." There are also two objects nearby—a cooler box marked as well-known and another object
 that looks like part of some equipment or structure.
```
