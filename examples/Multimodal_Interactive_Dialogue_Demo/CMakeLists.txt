cmake_minimum_required(VERSION 3.15.1)

project(demo)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wl,--allow-shlib-undefined")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wl,--allow-shlib-undefined")

if(${CMAKE_VERSION} VERSION_GREATER "3.15.0" AND CMAKE_SYSTEM_NAME STREQUAL "Linux")
  add_link_options("-Wl,-Bsymbolic")
endif()

if (CMAKE_SYSTEM_NAME STREQUAL "Android")
    set (TARGET_LIB_ARCH ${CMAKE_ANDROID_ARCH_ABI})
else()
    if(CMAKE_SIZEOF_VOID_P EQUAL 8)
        set (TARGET_LIB_ARCH aarch64)
    else()
        set (TARGET_LIB_ARCH armhf)
    endif()
    if (CMAKE_C_COMPILER MATCHES "uclibc")
        set (TARGET_LIB_ARCH ${TARGET_LIB_ARCH}_uclibc)
    endif()
endif()

# opencv
if (CMAKE_SYSTEM_NAME MATCHES "Linux")
  set(OpenCV_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../Qwen2-VL_Demo/deploy/3rdparty/opencv-linux-aarch64/share/OpenCV)
elseif(CMAKE_SYSTEM_NAME MATCHES "Android")
  set(OpenCV_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../Qwen2-VL_Demo/deploy/3rdparty/opencv-mobile-3.4.20-android/sdk/native/jni/abi-${CMAKE_ANDROID_ARCH_ABI})
endif()
find_package(OpenCV REQUIRED)

# rknn runtime
set(RKNN_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../Qwen2-VL_Demo/deploy/3rdparty/librknnrt)
set(LIBRKNNRT ${RKNN_PATH}/${CMAKE_SYSTEM_NAME}/librknn_api/${TARGET_LIB_ARCH}/librknnrt.so)
set(LIBRKNNRT_INCLUDES ${RKNN_PATH}/${CMAKE_SYSTEM_NAME}/librknn_api/include)

# rkllm runtime
set(RKLLM_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../../rkllm-runtime)
set(LIBRKLLMRT ${RKLLM_PATH}/${CMAKE_SYSTEM_NAME}/librkllm_api/${TARGET_LIB_ARCH}/librkllmrt.so)
set(LIBRKLLMRT_INCLUDES ${RKLLM_PATH}/${CMAKE_SYSTEM_NAME}/librkllm_api/include)

# demo
include_directories(${LIBRKNNRT_INCLUDES} ${LIBRKLLMRT_INCLUDES})
add_executable(${PROJECT_NAME} src/image_enc.cc src/main.cpp)
target_link_libraries(${PROJECT_NAME} PRIVATE ${OpenCV_LIBS} ${LIBRKNNRT} ${LIBRKLLMRT})

# Hide symbols to avoid possible symbol conflicts with RKLLM
set_target_properties(${PROJECT_NAME} PROPERTIES LINK_FLAGS  -Wl,--no-undefined,--version-script,${CMAKE_SOURCE_DIR}/c_export.map)

# Install the executable file to the specified directory
set(CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR}/install/demo_${CMAKE_SYSTEM_NAME}_${TARGET_LIB_ARCH})
install(TARGETS ${PROJECT_NAME} DESTINATION ./)
install(PROGRAMS ${LIBRKNNRT} ${LIBRKLLMRT} DESTINATION lib)
install(PROGRAMS demo.jpg DESTINATION ./)