#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Xinquan Chen
@contact:XinquanChen0117@163.com
@file: experiment.py
@time: 2022-05-05 22:29
"""
import matplotlib.pyplot as plt
import numpy as np
import time
from example_input import experimental_data_reshape
from reshape_tool import reshape_data
from workshop_scheduling import GeneticAlgorithm
from draw_gantt_chart import draw_gantt

files = ['./expData/Kacem1.txt', './expData/Kacem2.txt','./expData/Kacem3.txt', './expData/Kacem4.txt']
saveName = ['45', '88', '1010', '1510']
experimentName = ['4*5', '8*8', '10*10', '15*10']
parameterP = [50, 100, 120, 150]
parameterT = [200, 1000, 1500, 2500]
parameterPc = [0.8, 0.88, 0.85, 0.9]
parameterPm = [0.05, 0.05, 0.05, 0.1]
route = './resultFig/'
for num in range(1):
    experimentNum = str(5) + str(num + 1)
    for index, file in enumerate(files):
        name = str('_') + saveName[index] + str('_')
        expName = experimentName[index]
        data, workpieceIndex, wpstMatrix = experimental_data_reshape(file)
        orderWorkpiece, orderList, workpieceList, processList, machineList = reshape_data(data)
        idx = np.array([workpieceIndex.index(x) for x in workpieceList],dtype=int)
        wpstMatrix = wpstMatrix[idx, :][:, idx]
        begin = time.time()
        ga = GeneticAlgorithm(orderList, workpieceList, processList, machineList, wpstMatrix, populationNumber=parameterP[index],
                              times=parameterT[index], crossProbability=parameterPc[index], mutationProbability=parameterPm[index])
        rowData, bestGene, fitnessList, averageFitness = ga.exec(orderWorkpiece)
        end = time.time()
        x = [i for i in range(len(fitnessList))]
        plt.cla()
        plt.figure(figsize=(10, 8))
        plt.title(expName + '实验迭代曲线图', fontdict={'weight':'normal','size': 20})
        plt.plot(x, fitnessList)
        plt.plot(x, averageFitness)
        plt.ylabel('进化代数', fontdict={'weight':'normal','size': 13})
        plt.xlabel('适应度值', fontdict={'weight':'normal','size': 13})
        plt.savefig(route + experimentNum + name + 'curve.png')
        plt.close()
        plt.cla()
        print('experiment ' + experimentNum + expName + ' best makespan is '+ str(bestGene.fulfillTime))
        print('run time is {:.3f}'.format(end - begin))
        plt.figure(figsize=(24, 15))
        draw_gantt(rowData)
        plt.savefig(route + experimentNum + name + 'gantt.png')
        plt.close()
