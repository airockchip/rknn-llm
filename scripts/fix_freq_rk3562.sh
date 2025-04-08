echo 1 > /sys/devices/system/cpu/cpu0/cpuidle/state1/disable
echo 1 > /sys/devices/system/cpu/cpu1/cpuidle/state1/disable
echo 1 > /sys/devices/system/cpu/cpu2/cpuidle/state1/disable
echo 1 > /sys/devices/system/cpu/cpu3/cpuidle/state1/disable

echo "NPU available frequencies:"
cat /sys/class/devfreq/ff300000.npu/available_frequencies
echo "Fix NPU max frequency:"
echo 1000000000 > /sys/kernel/debug/rknpu/freq
cat /sys/kernel/debug/rknpu/freq

echo "CPU available frequencies:"
cat /sys/devices/system/cpu/cpufreq/policy0/scaling_available_frequencies
echo "Fix CPU max frequency:"
echo userspace > /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor
echo 2016000 > /sys/devices/system/cpu/cpu0/cpufreq/scaling_setspeed
cat /sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_cur_freq

echo "DDR available frequencies:"
cat /sys/class/devfreq/dmc/available_frequencies
echo "Fix DDR max frequency:"
echo userspace > /sys/class/devfreq/dmc/governor
echo 2112000000 > /sys/class/devfreq/dmc/userspace/set_freq
cat /sys/class/devfreq/dmc/cur_freq

export LD_LIBRARY_PATH=./lib
export RKLLM_LOG_LEVEL=1
sync; echo 3 > /proc/sys/vm/drop_caches
