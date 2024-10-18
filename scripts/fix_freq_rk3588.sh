#!/system/bin/sh

echo 1 > /sys/devices/system/cpu/cpu0/cpuidle/state1/disable
echo 1 > /sys/devices/system/cpu/cpu1/cpuidle/state1/disable
echo 1 > /sys/devices/system/cpu/cpu2/cpuidle/state1/disable
echo 1 > /sys/devices/system/cpu/cpu3/cpuidle/state1/disable
echo 1 > /sys/devices/system/cpu/cpu4/cpuidle/state1/disable
echo 1 > /sys/devices/system/cpu/cpu5/cpuidle/state1/disable
echo 1 > /sys/devices/system/cpu/cpu6/cpuidle/state1/disable
echo 1 > /sys/devices/system/cpu/cpu7/cpuidle/state1/disable

echo "NPU available frequencies:"
cat /sys/class/devfreq/fdab0000.npu/available_frequencies
echo "Fix NPU max frequency:"
echo userspace > /sys/class/devfreq/fdab0000.npu/governor
echo 1000000000 > /sys/class/devfreq/fdab0000.npu/userspace/set_freq
cat /sys/class/devfreq/fdab0000.npu/cur_freq

echo "CPU available frequencies:"
cat /sys/devices/system/cpu/cpufreq/policy0/scaling_available_frequencies
cat /sys/devices/system/cpu/cpufreq/policy4/scaling_available_frequencies
cat /sys/devices/system/cpu/cpufreq/policy6/scaling_available_frequencies
echo "Fix CPU max frequency:"
echo userspace > /sys/devices/system/cpu/cpufreq/policy0/scaling_governor
echo 1800000 > /sys/devices/system/cpu/cpufreq/policy0/scaling_setspeed
cat /sys/devices/system/cpu/cpufreq/policy0/scaling_cur_freq
echo userspace > /sys/devices/system/cpu/cpufreq/policy4/scaling_governor
echo 2352000 > /sys/devices/system/cpu/cpufreq/policy4/scaling_setspeed
cat /sys/devices/system/cpu/cpufreq/policy4/scaling_cur_freq
echo userspace > /sys/devices/system/cpu/cpufreq/policy6/scaling_governor
echo 2352000 > /sys/devices/system/cpu/cpufreq/policy6/scaling_setspeed
cat /sys/devices/system/cpu/cpufreq/policy6/scaling_cur_freq

echo "GPU available frequencies:"
cat /sys/class/devfreq/fb000000.gpu/available_frequencies
echo "Fix GPU max frequency:"
echo userspace > /sys/class/devfreq/fb000000.gpu/governor
echo 1000000000 > /sys/class/devfreq/fb000000.gpu/userspace/set_freq
cat /sys/class/devfreq/fb000000.gpu/cur_freq

echo "DDR available frequencies:"
cat /sys/class/devfreq/dmc/available_frequencies
echo "Fix DDR max frequency:"
echo userspace > /sys/class/devfreq/dmc/governor
echo 2112000000 > /sys/class/devfreq/dmc/userspace/set_freq
cat /sys/class/devfreq/dmc/cur_freq
