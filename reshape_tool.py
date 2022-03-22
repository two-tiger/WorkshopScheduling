#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Xinquan Chen
@contact:XinquanChen0117@163.com
@file: reshape_tool.py
@time: 2022-03-08 14:38
@brief: 用于将表格数据转换为标准数据
"""
from typing import (List, Dict)
from collections import namedtuple

test = [{'order':'#o-1','workpiece':'#w-1','number':'100','process':'#p-111','machine':['#m-1','#m-2'],'time':[65,70],'sequences':0},
        {'order':'#o-1','workpiece':'#w-1','number':'100','process':'#p-112','machine':['#m-7'],'time':[40],'sequences':1},
        {'order':'#o-1','workpiece':'#w-2','number':'10','process':'#p-121','machine':['#m-1','#m-2','#m-3'],'time':[30,40,35],'sequences':0},
        {'order':'#o-1','workpiece':'#w-2','number':'10','process':'#p-122','machine':['#m-7'],'time':[30],'sequences':1},
        {'order':'#o-2','workpiece':'#w-3','number':'100','process':'#p-231','machine':['#m-3','#m-4'],'time':[69,70],'sequences':0},
        {'order':'#o-2','workpiece':'#w-3','number':'100','process':'#p-232','machine':['#m-8'],'time':[25],'sequences':1},
        {'order':'#o-3','workpiece':'#w-4','number':'10','process':'#p-341','machine':['#m-4','#m-5'],'time':[145,140],'sequences':0},
        {'order':'#o-3','workpiece':'#w-4','number':'10','process':'#p-342','machine':['#m-6'],'time':[10],'sequences':1},
        {'order':'#o-3','workpiece':'#w-4','number':'10','process':'#p-343 ','machine':['#m-8'],'time':[17],'sequences':2}]

class WorkPiece():
    def __init__(self, order, workpieceIndex, number, process, machine, time):
        self.order = order
        self.workpieceIndex = workpieceIndex
        self.number = number
        self.process = process
        self.machine = machine
        self.time = time

def get_machine_name(data: Dict):
    for element in data["machine"]:
        return element

def get_workpiece_info(data: List[Dict], workpieceName):
    tempList = []
    for element in data:
        if element["workpiece"] == workpieceName:
            tempList.append(element)
    if tempList is None:
        print('The workpiece is not in the data!')
        return
    order = tempList[0]["order"]
    number = tempList[0]["number"]
    process = list(map(lambda v: v["process"], tempList))
    machine = list(map(lambda v: v["machine"], tempList))
    time = list(map(lambda v: v["time"], tempList))
    return order, number, process, machine, time

def reshape_data(data: List[Dict]):
    orderList = list(set(map(lambda v: v["order"], data)))
    workpieceList = list(set(map(lambda v: v["workpiece"], data)))
    processList = list(set(map(lambda v: v["process"], data)))
    machineList = list(map(lambda v: v["machine"], data))
    machineList = sum(machineList, [])
    orderWorkpiece = [[] for _ in range(len(orderList))]
    for workpieceIndex, workpiece in enumerate(workpieceList):
        order, number, process, machineName, time = get_workpiece_info(data, workpiece)
        order = orderList.index(order)
        process = list(map(lambda v: processList.index(v), process))
        machineIndex = []
        for machine in machineName:
            machine = list(map(lambda v: machineList.index(v), machine))
            machineIndex.append(machine)
        W = WorkPiece(order, workpieceIndex, number, process, machineIndex, time)
        orderWorkpiece[order].append(W)
    return orderWorkpiece, orderList, workpieceList, processList, machineList

if __name__ == '__main__':
    orderWorkpiece, orderList, workpieceList, processList, machineList = reshape_data(test)
    print(orderWorkpiece[1][1].process)