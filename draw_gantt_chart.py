#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Xinquan Chen
@contact:XinquanChen0117@163.com
@file: draw_gantt_chart.py
@time: 2022-03-08 14:37
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import cm

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

def draw_gantt(result):
    machine = list(set(map(lambda v: v["machine"], result)))
    workpiece = list(set(map(lambda v: v['workpiece'], result)))
    machine_index = {}
    for i in range(len(machine)):
        machine_index[machine[i]] = i
    workpiece_color = {}
    for i in range(len(workpiece)):
        workpiece_color[workpiece[i]] = plt.cm.tab20(i)
    for i in range(len(result)):
        order_name = result[i]['order']
        machine_id = machine_index[result[i]['machine']]
        start_time = result[i]['startTime']
        end_time = result[i]['endTime']
        width = end_time - start_time
        workpiece_name = result[i]['workpiece']
        color = workpiece_color[workpiece_name]
        # process_name = result[i]['process']
        # process_time = process_name + ' 时间 ' + str(width)
        plt.barh(machine_id, width, left=start_time, facecolor=color, edgecolor='black', label=workpiece_name)
        text = order_name + '-' + workpiece_name + ':' + str(end_time)
        plt.text(start_time, machine_id, text, fontsize='large', color='black')
    plt.title('调度甘特图', fontdict={'weight':'normal','size': 20})
    plt.yticks(range(len(machine)), machine)
    # plt.xticks(range(50))
    handles, labels = plt.gca().get_legend_handles_labels()
    handle_list, label_list = [], []
    for handle, label in zip(handles, labels):
        if label not in label_list:
            handle_list.append(handle)
            label_list.append(label)
    plt.legend(handle_list, label_list)
    plt.ylabel('机器编号', fontdict={'weight':'normal','size': 13})
    plt.xlabel('时间', fontdict={'weight':'normal','size': 13})
    #plt.show()