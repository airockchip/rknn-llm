# **Multimodal Interactive Dialogue Demo**

This demo uses the **Qwen2-VL-2B** model as an example to demonstrate interactive dialogue. For model conversion details, refer to **Qwen2-VL-2B_Demo**.

------

## **1. Download the Model**

You can download the **converted RKLLM model** from [rkllm_model_zoo](https://console.box.lenovo.com/l/l0tXb8) (fetch code: `rkllm`). The following models are required for this demo:

```
1. rkllm_model_zoo/1.1.4/RK3588/Qwen2-VL-2B_Demo/Qwen2-VL-2B_llm_w8a8_rk3588.rkllm
2. rkllm_model_zoo/1.1.4/RK3588/Qwen2-VL-2B_Demo/Qwen2-VL-2B_vision_rk3588.rknn
```

------

## **2. Parameter Configuration**

1. **Enable Multi-turn Dialogue Mode**

   - Set the `keep_history` parameter to `1` to retain conversation history. This prevents the cache from being cleared after each round.
   - To manually clear the cache, call the `rkllm_clear_kv_cache` function.

   ```cpp
   rkllm_infer_params.keep_history = 1;
   rkllm_clear_kv_cache(llmHandle, 1);
   ```

2. **Customizing Chat Templates**

   - The new model version includes a built-in chat template that formats prompts.
   - You can modify the system prompt, prefix, and postfix using the following function:

   ```cpp
   rkllm_set_chat_template(llmHandle, 
      "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n", 
      "<|im_start|>user\n", 
      "<|im_end|>\n<|im_start|>assistant\n"
   );
   ```

------

## **3. C++ Demo Setup**

We provide a sample C++ code for on-device inference. Users can compile the example by running `build-linux.sh` or `build-android.sh` (ensure the cross-compiler path is correctly set). This generates an `install/demo_Linux_aarch64` directory containing the executable `demo` and the required `lib` folder.

```bash
# Compile for Linux
./build-linux.sh

# Compile for Android
./build-android.sh

# Push the compiled demo to the device
adb push ./install/demo_Linux_aarch64 /data

# Push model files to the device
adb push Qwen2-VL-2B_llm_w8a8_rk3588.rkllm /data/demo_Linux_aarch64
adb push Qwen2-VL-2B_vision_rk3588.rknn /data/demo_Linux_aarch64

# Push a demo image to the device
adb push ./demo.jpg /data/demo_Linux_aarch64

# Push the frequency-setting script to the device
adb push rknn-llm/scripts/fix_freq_rk3588.sh /data
```

------

## **4. Running the Demo**

1. Access the demo directory on the target board:

   ```bash
   adb shell
   cd /data/demo_Linux_aarch64
   ```

2. Set up the environment and run the demo:

   ```bash
   # Export library path
   export LD_LIBRARY_PATH=./lib
   
   # Set CPU and NPU frequency
   sh fix_freq_rk3588.sh
   
   # Run the demo
   ./demo demo.jpg ./Qwen2-VL-2B_vision_rk3588.rknn ./Qwen2-VL-2B_llm_w8a8_rk3588.rkllm 2048 4096
   ```

------

## **5. Example Interaction**

Upon running the demo, you will see the following options:

Note: `<image>` indicates inserting image content.

```
**********************可输入以下问题对应序号获取回答/或自定义输入********************

[0] <image>这张图片中有什么？
[1] 图片中地球是什么颜色的？
[2] 图片中月球表面是什么颜色的？
[3] 图片中星空是什么颜色的？
[4] 结合图片，说明下月球和地球表面颜色相同吗，为什么？

*************************************************************************
```

### **Sample Dialogue**

```
user: 0
<image>这张图片中有什么？
robot: 这张图片展示了一位宇航员坐在月球上，手里拿着一瓶绿色的啤酒。背景是地球和星空，给人一种在太空中享受饮料的感觉。

user: 1
图片中地球是什么颜色的？
robot: 图片中的地球是绿色和黄色相间的。

user: 2
图片中月球表面是什么颜色的？
robot: 图片中的月球表面是灰色和黑色相间的。

user: 3
图片中星空是什么颜色的？
robot: 图片中的星空是蓝色和白色的相间。

user: 4
结合图片，说明下月球和地球表面颜色相同吗，为什么？
robot: 不是的。月球表面是灰色和黑色相间的，而地球表面则是绿色和黄色相间的。这是因为月球没有大气层，所以看起来像是灰色；而地球有大气层，所以看起来像是绿色和黄色相间。
```

