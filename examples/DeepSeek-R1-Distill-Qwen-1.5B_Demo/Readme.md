# DeepSeek-R1-Distill-Qwen-1.5B Demo
1. This demo demonstrates how to deploy the DeepSeek-R1-Distill-Qwen-1.5B model.
2. The open-source model used in this demo is available at: [DeepSeek-R1-Distill-Qwen-1.5B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B)

## 1. Requirements

```
rkllm-toolkit==1.2.0
rkllm-runtime==1.2.0
python >=3.8
```

## 2. Model Conversion

1. Firstly, you need to create `data_quant.json` for quantizing the rkllm model, we use fp16 model generation results as the quantization calibration data.
2. Secondly, you run the following code to generate `data_quant.json`  and export the rkllm model.
3. You can also download the **converted rkllm model**  from [rkllm_model_zoo](https://console.box.lenovo.com/l/l0tXb8), fetch code: rkllm

```bash
cd export
python generate_data_quant.py -m /path/to/DeepSeek-R1-Distill-Qwen-1.5B
python export_rkllm.py
```

## 3. C++ Demo

In the `deploy` directory, we provide example code for board-side inference. 

### 1. Compile and Build

Users can directly compile the example code by running the `deploy/build-linux.sh` or `deploy/build-android.sh` script (replacing the cross-compiler path with the actual path). This will generate an `install/demo_Linux_aarch64` folder in the `deploy` directory, containing the executables `llm_demo`, and the `lib` folder.

```bash
cd deploy
# for linux
./build-linux.sh
# for android
./build-android.sh
# push install dir to device
adb push install/demo_Linux_aarch64 /data
# push model file to device
adb push DeepSeek-R1-Distill-Qwen-1.5B.rkllm /data/demo_Linux_aarch64
# push the appropriate fixed-frequency script to the device
adb push ../../../scripts/fix_freq_rk3588.sh /data/demo_Linux_aarch64
```

### 2. Run Demo

Enter the `/data/demo_Linux_aarch64` directory on the board and run the example using the following code

```bash
adb shell
cd /data/demo_Linux_aarch64
# export lib path
export LD_LIBRARY_PATH=./lib
# Execute the fixed-frequency script
sh fix_freq_rk3588.sh
# Set the logging level for performance analysis
export RKLLM_LOG_LEVEL=1
./llm_demo /path/to/your/rkllm/model 2048 4096

# Running result                                                          
rkllm init start
rkllm init success

**********************可输入以下问题对应序号获取回答/或自定义输入********************

[0] 现有一笼子，里面有鸡和兔子若干只，数一数，共有头14个，腿38条，求鸡和兔子各有多少只？
[1] 有28位小朋友排成一行,从左边开始数第10位是学豆,从右边开始数他是第几位?

*************************************************************************


user:
```

example 1 （DeepSeek-R1-Distill-Qwen-1.5B_W8A8_RK3588.rkllm）

```
user: 0
现有一笼子，里面有鸡和兔子若干只，数一数，共有头14个，腿38条，求鸡和兔子各有多少只？
robot: <think>
首先，设鸡的数量为x，兔子的数量为y。

根据题目中的条件，我们知道：

1. 鸡和兔子的总数是14，因此有方程：
   x + y = 14

2. 鸡有两条腿，兔子有四条腿，总腿数是38，所以有另一个方程：
   2x + 4y = 38

接下来，通过代入法或消元法来解这两个方程。假设我们用代入法：

从第一个方程中，可以得到：
x = 14 - y

将这个表达式代入第二个方程：
2(14 - y) + 4y = 38
展开计算后得到：
28 - 2y + 4y = 38
合并同类项：
2y = 10
解得：
y = 5

然后，将y的值代入x = 14 - y中：
x = 14 - 5 = 9

因此，鸡有9只，兔子有5只。
</think>

要解决这个问题，我们可以设鸡的数量为 \( x \)，兔子的数量为 \( y \)。根据题目给出的条件：

1. **头的总数**：每只鸡和兔子都有一个头，所以：
   \[
   x + y = 14
   \]

2. **腿的总数**：鸡有两条腿，兔子有四条腿，总腿数为38条，因此：
   \[
   2x + 4y = 38
   \]

接下来，我们可以通过解这两个方程来找到 \( x \) 和 \( y \) 的值。

**步骤一：简化第二个方程**

将第二个方程两边同时除以2：
\[
x + 2y = 19
\]

现在，我们有两个方程：
\[
\begin{cases}
x + y = 14 \\
x + 2y = 19
\end{cases}
\]

**步骤二：消元法**

用第二个方程减去第一个方程：
\[
(x + 2y) - (x + y) = 19 - 14 \\
y = 5
\]

**步骤三：代入求 \( x \)**

将 \( y = 5 \) 代入第一个方程：
\[
x + 5 = 14 \\
x = 14 - 5 \\
x = 9
\]

因此，鸡的数量是 **9只**，兔子的数量是 **5只**。

**最终答案：**
鸡有 \(\boxed{9}\) 只，兔子有 \(\boxed{5}\) 只。
```

example 2 （DeepSeek-R1-Distill-Qwen-1.5B_W8A8_RK3588.rkllm）

```

user: 1
有28位小朋友排成一行,从左边开始数第10位是学豆,从右边开始数他是第几位?
robot: <think>
首先，总共有28位小朋友。

从左边开始数，第10位是学豆的位置。

因此，从右边开始数，学豆的位置是从右边数的第(28 - 10 + 1) = 第19位。
</think>

**解答：**

我们有28位小朋友排成一行。题目要求确定从右边开始数时，第10位是学豆的位置。

**步骤如下：**

1. **总人数**：共有28位小朋友。
2. **左边数的顺序**：从左边开始数，第10位是学豆。
3. **右边数的计算**：
   - 从右边数时，第1位对应左边数的第28位。
   - 因此，第n位在左边对应的是第(28 - n + 1)位在右边。

4. **具体计算**：
   \[
   第10位在左边 = 第(28 - 10 + 1) = 第19位在右边
   \]

**最终答案：**

\boxed{19}
```

