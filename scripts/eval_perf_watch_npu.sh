#!/bin/bash  
  
# 初始化峰值变量  
peak_value=0  
current_value=0  
last_value=0  
  
# 无限循环读取文件  
while true; do  
    # 读取文件的第一行（假设文件只包含一行数值）  
    core0=`awk -F'[:% ]+' '/NPU load:/ {print $4}' /sys/kernel/debug/rknpu/load`
    current_value=$core0

    # echo "current: $current_value"
  
    # 检查是否找到新的峰值  
    if [[ $current_value -gt $peak_value ]]; then  
        peak_value=$current_value  
        echo "Peak value: $peak_value"  
        # peak_value=0  # 重置峰值  
    fi  
  
    # 保存当前值作为下一次循环的last_value  
    last_value=$current_value  
  
    # 休眠一段时间再读取文件  
    sleep 0.5  # 例如，每秒读取一次  
done
