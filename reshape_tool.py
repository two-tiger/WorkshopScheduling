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

test = [{'order':'#o-1','workpiece':'#w-1','number':10000,'process':'#p-111','machine':['#m-1','#m-2','#m-3','#m-4','#m-5'],'time':[300,300,300,280,280]},
        {'order':'#o-1','workpiece':'#w-1','number':10000,'process':'#p-112','machine':['#m-6','#m-7','#m-8','#m-9'],'time':[40,40,40,40]},
        {'order':'#o-1','workpiece':'#w-2','number':10000,'process':'#p-121','machine':['#m-1','#m-2','#m-3','#m-4'],'time':[180,180,180,180]},
        {'order':'#o-1','workpiece':'#w-2','number':10000,'process':'#p-122','machine':['#m-6','#m-7','#m-8','#m-9'],'time':[40,40,40,40]},
        {'order':'#o-1','workpiece':'#w-3','number':10000,'process':'#p-131','machine':['#m-1','#m-2','#m-3','#m-5'],'time':[340,340,350,350]},
        {'order':'#o-1','workpiece':'#w-3','number':10000,'process':'#p-132','machine':['#m-6','#m-7','#m-8','#m-9'],'time':[40,38,40,38]},
        {'order':'#o-1','workpiece':'#w-3','number':10000,'process':'#p-133','machine':['#m-10'],'time':[20]},
        {'order':'#o-2','workpiece':'#w-4','number':1000,'process':'#p-241','machine':['#m-1','#m-2','#m-3','#m-4','#m-5'],'time':[290,290,285,285,290]},
        {'order':'#o-2','workpiece':'#w-4','number':1000,'process':'#p-242','machine':['#m-6','#m-7','#m-9'],'time':[40,40,40]},
        {'order':'#o-2','workpiece':'#w-5','number':10000,'process':'#p-251','machine':['#m-1','#m-2','#m-3','#m-4'],'time':[184,184,180,184]},
        {'order':'#o-2','workpiece':'#w-5','number':10000,'process':'#p-252','machine':['#m-6','#m-8','#m-9'],'time':[40,40,40]},
        {'order':'#o-2','workpiece':'#w-6','number':100000,'process':'#p-261','machine':['#m-4','#m-5'],'time':[140,140]},
        {'order':'#o-2','workpiece':'#w-6','number':100000,'process':'#p-262','machine':['#m-7','#m-8','#m-9'],'time':[20,20,20]},
        {'order':'#o-3','workpiece':'#w-1','number':8000,'process':'#p-311','machine':['#m-1','#m-2','#m-3','#m-4','#m-5'],'time':[300,300,300,280,280]},
        {'order':'#o-3','workpiece':'#w-1','number':8000,'process':'#p-312','machine':['#m-6','#m-7','#m-8','#m-9'],'time':[40,40,40,40]},
        {'order':'#o-3','workpiece':'#w-3','number':10000,'process':'#p-331','machine':['#m-1','#m-2','#m-3','#m-5'],'time':[340,340,350,350]},
        {'order':'#o-3','workpiece':'#w-3','number':10000,'process':'#p-332','machine':['#m-6','#m-7','#m-8','#m-9'],'time':[40,38,40,38]},
        {'order':'#o-3','workpiece':'#w-3','number':10000,'process':'#p-333','machine':['#m-10'],'time':[20]},
        {'order':'#o-3','workpiece':'#w-7','number':12000,'process':'#p-371 ','machine':['#m-3','#m-4','#m-5'],'time':[660,660,660]},
        {'order':'#o-3','workpiece':'#w-7','number':12000,'process':'#p-372 ','machine':['#m-7','#m-8','#m-9','#m-10'],'time':[40,40,40,40]}]

class OrderWorkpieceNonExistent(Exception):
    pass

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

def get_workpiece_info(data: List[Dict], orderName, workpieceName):
    tempList = []
    for element in data:
        if element["order"] == orderName and element["workpiece"] == workpieceName:
            tempList.append(element)
    if len(tempList) == 0:
        raise OrderWorkpieceNonExistent("This order workpiece is non-existent")
    number = tempList[0]["number"]
    process = list(map(lambda v: v["process"], tempList))
    machine = list(map(lambda v: v["machine"], tempList))
    time = list(map(lambda v: v["time"], tempList))
    return number, process, machine, time

def reshape_data(data: List[Dict]):
    orderList = list(set(map(lambda v: v["order"], data)))
    workpieceList = list(set(map(lambda v: v["workpiece"], data)))
    processList = list(set(map(lambda v: v["process"], data)))
    machineList = list(map(lambda v: v["machine"], data))
    machineList = list(set(sum(machineList, [])))
    orderWorkpiece = [[] for _ in range(len(orderList))]
    for orderIndex, order in enumerate(orderList):
        for workpieceIndex, workpiece in enumerate(workpieceList):
            try:
                number, process, machineName, time = get_workpiece_info(data, order, workpiece)
            except OrderWorkpieceNonExistent as e:
                # print(e)
                continue
            process = list(map(lambda v: processList.index(v), process))
            machineIndex = []
            for machine in machineName:
                machine = list(map(lambda v: machineList.index(v), machine))
                machineIndex.append(machine)
            W = WorkPiece(orderIndex, workpieceIndex, number, process, machineIndex, time)
            orderWorkpiece[orderIndex].append(W)
    return orderWorkpiece, orderList, workpieceList, processList, machineList

if __name__ == '__main__':
    orderWorkpiece, orderList, workpieceList, processList, machineList = reshape_data(test)
    print(processList)
    print(orderWorkpiece[0][0].process)
    print(machineList)
    print(orderWorkpiece[0][0].machine)