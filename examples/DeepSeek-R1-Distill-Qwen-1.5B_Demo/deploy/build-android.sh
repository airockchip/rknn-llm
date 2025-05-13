#!/bin/bash
# Debug / Release / RelWithDebInfo
set -e
if [[ -z ${BUILD_TYPE} ]];then
    BUILD_TYPE=Release
fi

ANDROID_NDK_PATH=~/opts/android-ndk-r21e
TARGET_ARCH=arm64-v8a

TARGET_PLATFORM=android
if [[ -n ${TARGET_ARCH} ]];then
TARGET_PLATFORM=${TARGET_PLATFORM}_${TARGET_ARCH}
fi

ROOT_PWD=$( cd "$( dirname $0 )" && cd -P "$( dirname "$SOURCE" )" && pwd )
BUILD_DIR=${ROOT_PWD}/build/build_${TARGET_PLATFORM}_${BUILD_TYPE}

if [[ ! -d "${BUILD_DIR}" ]]; then
  mkdir -p ${BUILD_DIR}
fi

cd ${BUILD_DIR}
cmake ../.. \
    -DCMAKE_SYSTEM_NAME=Android \
    -DCMAKE_SYSTEM_VERSION=23 \
    -DCMAKE_ANDROID_ARCH_ABI=${TARGET_ARCH} \
    -DCMAKE_ANDROID_STL_TYPE=c++_static \
    -DCMAKE_ANDROID_NDK=${ANDROID_NDK_PATH} \
    -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
    -DCMAKE_INSTALL_PREFIX=${INSTALL_DIR} \
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON \

make -j4
make install