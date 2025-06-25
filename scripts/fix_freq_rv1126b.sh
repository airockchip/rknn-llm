#!/system/bin/sh
echo 1 > /sys/devices/system/cpu/cpu0/cpuidle/state1/disable
echo 1 > /sys/devices/system/cpu/cpu1/cpuidle/state1/disable
echo 1 > /sys/devices/system/cpu/cpu2/cpuidle/state1/disable
echo 1 > /sys/devices/system/cpu/cpu3/cpuidle/state1/disable

echo "NPU available frequencies:"
cat /sys/class/devfreq/22000000.npu/available_frequencies
echo "Fix NPU max frequency:"
echo userspace > /sys/class/devfreq/22000000.npu/governor
echo 950000000 > /sys/class/devfreq/22000000.npu/userspace/set_freq
cat /sys/class/devfreq/22000000.npu/cur_freq

echo "CPU available frequencies:"
cat /sys/devices/system/cpu/cpufreq/policy0/scaling_available_frequencies
echo "Fix CPU max frequency:"
echo userspace > /sys/devices/system/cpu/cpufreq/policy0/scaling_governor
echo 1608000 > /sys/devices/system/cpu/cpufreq/policy0/scaling_setspeed
cat /sys/devices/system/cpu/cpufreq/policy0/scaling_cur_freq

echo "DDR available frequencies:"
cat /sys/class/devfreq/dmc/available_frequencies
echo "Fix DDR max frequency:"
echo userspace > /sys/class/devfreq/dmc/governor
echo 1332000000 > /sys/class/devfreq/dmc/userspace/set_freq
cat /sys/class/devfreq/dmc/cur_freq