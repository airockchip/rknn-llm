set -e
rm -rf build
mkdir build && cd build

ANDROID_NDK_PATH=~/opts/android-ndk-r21e
cmake .. -DCMAKE_ANDROID_NDK=${ANDROID_NDK_PATH} \
        -DCMAKE_SYSTEM_NAME=Android \
        -DCMAKE_SYSTEM_VERSION=23 \
        -DCMAKE_ANDROID_ARCH_ABI=arm64-v8a \
        -DCMAKE_ANDROID_STL_TYPE=c++_static \
        -DCMAKE_BUILD_TYPE=Release \

make -j8
make install