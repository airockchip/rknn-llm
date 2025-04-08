#!/bin/bash  
# 定义需要检查的包
packages=("bc" "sysstat")

# 遍历每个包，检查是否安装
for pkg in "${packages[@]}"; do
    dpkg -l | grep -qw "$pkg"
    if [ $? -ne 0 ]; then
        echo "Package $pkg is not installed. Installing..."
        sudo apt install -y "$pkg"
    fi
done

if [ $# -lt 1 ]; then
    echo "Usage: bash $0 program_name"
    echo "Note: for example, bash eval_perf_watch_cpu.sh llm_demo"
    echo "Error: not enough arguments. At least 1 arguments are required."
    exit 1
fi

# 获取目标进程的 PID
PID=`ps -ef | grep $1 | head -n 1 | awk '{print $2}'`
# 获取逻辑核心数
total_cores=$(nproc)
echo "Total CPU cores (logical): $total_cores"
# 初始化峰值变量
peak_value=0
current_value=0

# 检查PID是否有效
if [[ -z "$PID" ]]; then
    echo "No process found for time_eval."
    exit 1
fi

# 无限循环监控CPU使用率
while true; do
    # 获取当前的CPU使用率
    current_value=$(pidstat -p $PID 1 1 | awk 'NR==4 {print $8}')
    current_value=$(echo "$current_value" | bc) # 去除小数点，取整

    # 将当前值和峰值除以800
    current_divided=$(echo "scale=5; $current_value / $total_cores / 100" | bc)
    peak_divided=$(echo "scale=5; $peak_value / $total_cores / 100" | bc)

    # 检测并更新峰值
    if [[ $(echo "$current_value > $peak_value" | bc) -eq 1 ]]; then
        peak_value=$current_value
        peak_divided=$(echo "scale=3; $peak_value / $total_cores" | bc)
	echo "New peak CPU usage: $peak_value% (Divided by $total_cores : $peak_divided %)"
    fi

    # 等待一秒后再次检查
    sleep 1
done

