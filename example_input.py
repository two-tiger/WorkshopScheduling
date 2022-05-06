#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Xinquan Chen
@contact:XinquanChen0117@163.com
@file: example_input.py
@time: 2022-05-05 20:47
"""
import numpy as np

def experimental_data_reshape(filename):
    f = open(filename, 'r', encoding='utf-8')
    workpieceCount = 1
    processCount = 1
    result = []
    workpieceNameSet = []
    for line in f.readlines()[1:]:
        if line == '\n':
            workpieceCount += 1
            processCount = 1
            continue
        processInfo = {}
        line_list = []
        line = line.split()
        for element in line:
            line_list.append(int(element))
        timeSet = line_list[1::2]
        machineSet = list(map(lambda x: '#m-' + str(x), line_list[0::2]))
        workpieceName = '#w-' + str(workpieceCount)
        processName = '#p-1' + str(workpieceCount) + str(processCount)
        processCount += 1
        processInfo['order'] = '#o-1'
        processInfo['workpiece'] = workpieceName
        processInfo['number'] = 1
        processInfo['process'] = processName
        processInfo['machine'] = machineSet
        processInfo['time'] = timeSet
        result.append(processInfo)
        workpieceNameSet.append(workpieceName)
    workpieceNameSet = list(set(workpieceNameSet))
    wpstMatrix = np.full((len(workpieceNameSet),len(workpieceNameSet)), 0)
    return result, workpieceNameSet, wpstMatrix

if __name__ == '__main__':
    result, workpieceNameSet, wpstMatrix = experimental_data_reshape('./expData/Kacem1.txt')
    print(result)