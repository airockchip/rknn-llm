echo userspace > /sys/class/devfreq/fdab0000.npu/governor
echo 1000000000 > /sys/class/devfreq/fdab0000.npu/userspace/set_freq

echo userspace > /sys/devices/system/cpu/cpufreq/policy0/scaling_governor
echo 1800000 > /sys/devices/system/cpu/cpufreq/policy0/scaling_setspeed
echo userspace > /sys/devices/system/cpu/cpufreq/policy4/scaling_governor
echo 2400000 > /sys/devices/system/cpu/cpufreq/policy4/scaling_setspeed
echo userspace > /sys/devices/system/cpu/cpufreq/policy6/scaling_governor
echo 2400000 > /sys/devices/system/cpu/cpufreq/policy6/scaling_setspeed

echo userspace > /sys/class/devfreq/dmc/governor
echo 2112000000 > /sys/class/devfreq/dmc/userspace/set_freq

echo userspace > /sys/class/devfreq/fb000000.gpu/governor
echo 1000000000 > /sys/class/devfreq/fb000000.gpu/userspace/set_freq
