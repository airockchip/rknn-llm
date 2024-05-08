# Aarch64 Linux Demo
## Build
Ensure the `GCC_COMPILER_PATH` option in the `build-linux.sh` script is correctly configured:
```sh
GCC_COMPILER_PATH=/path/to/your/gcc-arm-10.2-2020.11-x86_64-aarch64-none-linux-gnu/bin/aarch64-none-linux-gnu
```
To execute, run:
```bash
bash build-linux.sh
```

## Install
Push the compiled `llm_demo` file and `librkllmrt.so` file to the device:
```bash
adb push build/build_linux_aarch64_Release/llm_demo /userdata/llm
adb push ../../runtime/Linux/librkllm_api/aarch64/librkllmrt.so /userdata/llm/lib
```

## Run
```bash
adb shell
cd /userdata/llm
export LD_LIBRARY_PATH=./lib
taskset f0 ./llm_demo /path/to/your/rkllm/model
```

# Android Demo
## Build
Ensure the `ANDROID_NDK_PATH` option in the `build-android.sh` script is correctly configured:
```sh
ANDROID_NDK_PATH=/path/to/your/android-ndk-r18b
```
To execute, run:
```bash
bash build-android.sh
```

## Install
Push the compiled `llm_demo` file and `librkllmrt.so` file to the device:
```bash
adb push build/build_android_arm64-v8a_Release/llm_demo /userdata/llm
adb push ../../runtime/Android/librkllm_api/arm64-v8a/librkllmrt.so /userdata/llm/lib
```

## Run
```bash
adb shell
cd /userdata/llm
export LD_LIBRARY_PATH=./lib
taskset f0 ./llm_demo /path/to/your/rkllm/model
```