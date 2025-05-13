#!/bin/sh                                                                                                                                                     
 
killall start_rknn.sh > /dev/null 2>&1
killall rknn_server > /dev/null 2>&1
start_rknn.sh &