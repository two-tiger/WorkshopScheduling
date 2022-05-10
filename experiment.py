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
from multiprocessing.pool import Pool
import json
from tqdm import tqdm

files = ['./expData/Kacem1.txt', './expData/Kacem2.txt', './expData/Kacem3.txt', './expData/Kacem4.txt']
saveName = ['45', '88', '1010', '1510']
experimentName = ['4*5', '8*8', '10*10', '15*10']
# parameterP = [[50, 120, 120, 200], [50, 120, 120, 200], [50, 120, 120, 230]]
# parameterT = [[200, 1500, 1500, 5000], [200, 1500, 1500, 3000], [200, 1200, 1500, 3000]]
# parameterPc = [[0.8, 0.88, 0.85, 0.9], [0.8, 0.85, 0.85, 0.9], [0.8, 0.85, 0.85, 0.9]]
# parameterPm = [[0.2, 0.28, 0.25, 0.3], [0.2, 0.25, 0.25, 0.32], [0.2, 0.25, 0.25, 0.33]]
parameterP = [[50, 120, 120, 200]]
parameterT = [[200, 1500, 1500, 5000]]
parameterPc = [[0.8, 0.88, 0.85, 0.9]]
parameterPm = [[0.2, 0.28, 0.25, 0.3]]
route = './resultFig/'


class Experiment():
    def __init__(self, index, file, pp, pt, pc, pm, experimentNum):
        self.index = index
        self.file = file
        self.pp = pp
        self.pt = pt
        self.pc = pc
        self.pm = pm
        self.experimentNum = experimentNum

    def __call__(self):
        index = self.index
        file = self.file
        experimentNum = self.experimentNum
        name = str('_') + saveName[index] + str('_')
        expName = experimentName[index]
        data, workpieceIndex, wpstMatrix = experimental_data_reshape(file)
        orderWorkpiece, orderList, workpieceList, processList, machineList = reshape_data(data)
        idx = np.array([workpieceIndex.index(x) for x in workpieceList], dtype=int)
        wpstMatrix = wpstMatrix[idx, :][:, idx]
        begin = time.time()
        ga = GeneticAlgorithm(orderList, workpieceList, processList, machineList, wpstMatrix,
                              populationNumber=self.pp[index],
                              times=self.pt[index],
                              crossProbability=self.pc[index],
                              mutationProbability=self.pm[index])
        rowData, bestGene, fitnessList, averageFitness = ga.exec(orderWorkpiece)
        end = time.time()
        x = [i for i in range(len(fitnessList))]
        plt.cla()
        plt.figure(figsize=(12, 10))
        plt.title(expName + '实验迭代曲线图', fontdict={'weight': 'normal', 'size': 16})
        plt.plot(x, fitnessList)
        plt.plot(x, averageFitness)
        plt.ylabel('进化代数', fontdict={'weight': 'normal', 'size': 16})
        plt.xlabel('适应度值', fontdict={'weight': 'normal', 'size': 16})
        plt.savefig(route + experimentNum + name + 'curve.svg')
        plt.close()
        plt.cla()
        #print('experiment ' + experimentNum + expName + ' best makespan is ' + str(bestGene.fulfillTime))
        #print('run time is {:.3f}'.format(end - begin))
        plt.figure(figsize=(18, 10))
        draw_gantt(rowData)
        plt.savefig(route + experimentNum + name + 'gantt.svg')
        plt.close()

        return {experimentNum + expName: str(bestGene.fulfillTime), "time": str(round((end - begin), 3))}


class wrapper():
    def __call__(self, e: Experiment):
        return e()


if __name__ == '__main__':
    experiments = []
    for experimentIndex in range(9,10):
        pp = parameterP[experimentIndex - 9]
        pt = parameterT[experimentIndex - 9]
        ppc = parameterPc[experimentIndex - 9]
        ppm = parameterPm[experimentIndex - 9]
        for num in range(10):
            experimentNum = str(experimentIndex) + str(num + 1)
            for index, file in enumerate(files):
                experiments.append(Experiment(index, file, pp, pt, ppc, ppm, experimentNum))
    with Pool() as pool:
        returns = list(tqdm(pool.imap(wrapper(), experiments), total=len(experiments)))
    # print(returns)
    js = open("experimentResult.json", "w")
    json.dump(returns, js)
    js.close()

    # experiments = []
    # pp = parameterP[0]
    # pt = parameterT[0]
    # ppc = parameterPc[0]
    # ppm = parameterPm[0]
    # for num in range(20, 30):
    #     experimentNum = str(num + 1)
    #     experiments.append(Experiment(3, files[3], pp, pt, ppc, ppm, experimentNum))
    # with Pool() as pool:
    #     returns = list(tqdm(pool.imap(wrapper(), experiments), total=len(experiments)))
    # # print(returns)
    # js = open("experimentResult.json", "w")
    # json.dump(returns, js)
    # js.close()
