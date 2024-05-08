#!/system/bin/sh

echo userspace > /sys/class/devfreq/27700000.npu/governor
echo 1000000000 > /sys/class/devfreq/27700000.npu/userspace/set_freq

echo userspace > /sys/devices/system/cpu/cpufreq/policy0/scaling_governor
echo 2208000 > /sys/devices/system/cpu/cpufreq/policy0/scaling_setspeed
echo userspace > /sys/devices/system/cpu/cpufreq/policy4/scaling_governor
echo 2304000 > /sys/devices/system/cpu/cpufreq/policy4/scaling_setspeed
