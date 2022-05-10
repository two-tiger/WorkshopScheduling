#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Xinquan Chen
@contact:XinquanChen0117@163.com
@file: workshop_scheduling.py
@time: 2022-03-08 13:59
"""
from random import randint, choice, random
from typing import List, Tuple, Set, Dict, Any
import matplotlib.pyplot as plt
import numpy as np
from reshape_tool import reshape_data, WorkPiece
from draw_gantt_chart import draw_gantt


# 三层编码基因
class Gene(object):
    def __init__(self, fitness: float = 0, length=0, first_layer=None, second_layer=None, third_layer=None):
        self.fitness = fitness # 适应度
        self.length = length # 一层染色体长度
        self.first_layer: list = first_layer # 第一层染色体
        self.second_layer: list = second_layer # 第二层染色体
        self.third_layer: list = third_layer # 第三层染色体

    # 将基因进行打印
    def print_gene(self):
        print("chromosome: ", self.first_layer, self.second_layer, self.third_layer)


# 存储解码结果
class GeneEvaluation():
    def __init__(self, processNumber, machineNumber, orderNumber, workpieceNumber):
        self.fulfillTime = 0 # 存储总时间
        self.processMachine = [0 for _ in range(processNumber)] # 存储每一步使用的机器
        self.machineWorkTime = [0 for _ in range(machineNumber)] # 存储机器工作时间
        self.machineWorkPiece = [-1 for _ in range(machineNumber)] # 记录机器正在加工的工件序号
        self.machineEmptyTime = [[] for _ in range(machineNumber)] # 记录机器上的空窗时间
        # 存储每一步开始时间
        self.startTime = [[[0 for _ in range(processNumber)] for _ in range(workpieceNumber)] for _ in range(orderNumber)]
        # 存储每一步结束时间
        self.endTime = [[[0 for _ in range(processNumber)] for _ in range(workpieceNumber)] for _ in range(orderNumber)]


# 遗传算法实现
class GeneticAlgorithm():
    def __init__(self, orderList, workpieceList, processList, machineList, wpstMatrix, orderPriority, populationNumber=150, times=400,
                 crossProbability=0.8, mutationProbability=0.1):
        self.populationNumber = populationNumber  # 种群数量
        self.times = times  # 遗传代数
        self.crossProbability = crossProbability  # 交叉概率
        self.mutationProbability = mutationProbability  # 变异概率

        self.orderList = orderList # 订单列表
        self.workpieceList = workpieceList # 工件列表
        self.processList = processList # 步骤列表
        self.machineList = machineList # 机器列表
        self.wpstMatrix = wpstMatrix # 换机时间矩阵
        self.orderPriority = orderPriority # 订单优先级

        self.orderWorkpiece = None # 订单工件信息存储

        self.genes: List[Gene] = [] # 当前代数基因列表

    # 评估基因长度，计算每步开始时间和结束时间
    def evaluate_gene(self, g: Gene) -> GeneEvaluation:
        # 根据订单顺序调整染色体
        priorityList = np.array([self.orderPriority[i] for i in g.first_layer])
        idxPriority = np.argsort(priorityList, kind='stable')
        firstLayer = np.array(g.first_layer)
        secondLayer = np.array(g.second_layer)
        thirdLayer = np.array(g.third_layer)
        firstLayer = firstLayer[idxPriority]
        secondLayer = secondLayer[idxPriority]
        thirdLayer = thirdLayer[idxPriority]
        # 解码存储初始化
        processNumber = len(self.processList)
        machineNumber = len(self.machineList)
        orderNumber = len(self.orderList)
        workpieceNumber = len(self.workpieceList)
        evaluation = GeneEvaluation(processNumber, machineNumber, orderNumber, workpieceNumber)
        # 步骤计数
        processCount = [[0 for _ in range(workpieceNumber)] for _ in range(orderNumber)]
        for i in range(g.length):
            # 当前步骤信息记录
            presentWorkpiece = self.orderWorkpiece[firstLayer[i]][secondLayer[i]].workpieceIndex
            processOrder = processCount[firstLayer[i]][secondLayer[i]]
            processId = self.orderWorkpiece[firstLayer[i]][secondLayer[i]].process[processOrder]
            machineId = self.orderWorkpiece[firstLayer[i]][secondLayer[i]].machine[processOrder][thirdLayer[i]]
            evaluation.processMachine[processId] = machineId
            time = self.orderWorkpiece[firstLayer[i]][secondLayer[i]].time[processOrder][thirdLayer[i]]
            time *= self.orderWorkpiece[firstLayer[i]][secondLayer[i]].number
            processCount[firstLayer[i]][secondLayer[i]] += 1
            # 步骤在机器上开始和结束时间解码
            # 该工序为工件的第一道加工工序，且是机器的第一道加工工序，直接从机器的零时刻进行加工
            if processOrder == 0 and evaluation.machineWorkTime[machineId] == 0:
                evaluation.startTime[firstLayer[i]][secondLayer[i]][processOrder] = 0

            # 工序是机器的第一道工序，不是工件的第一道工序，从上道工序完工时间处开始加工
            elif evaluation.machineWorkTime[machineId] == 0 and processOrder > 0:
                evaluation.startTime[firstLayer[i]][secondLayer[i]][processOrder] = \
                    evaluation.endTime[firstLayer[i]][secondLayer[i]][processOrder - 1]
                emptyStart = 0
                emptyEnd = evaluation.startTime[firstLayer[i]][secondLayer[i]][processOrder]
                preWorkpiece = -1
                aftWorkpiece = presentWorkpiece
                evaluation.machineEmptyTime[machineId].append(np.array([emptyStart, emptyEnd, preWorkpiece, aftWorkpiece]))

            # 工序不是机器的第一道工序，是工件的第一道工序
            elif processOrder == 0 and evaluation.machineWorkTime[machineId] != 0:
                for empty_time in evaluation.machineEmptyTime[machineId]:
                    needTime = time
                    if empty_time[2] != -1:
                        needTime += self.wpstMatrix[empty_time[2]][presentWorkpiece]
                        needTime += self.wpstMatrix[preWorkpiece][empty_time[3]]
                    else:
                        needTime += self.wpstMatrix[preWorkpiece][empty_time[3]]
                    if needTime <= (empty_time[1] - empty_time[0]):
                        evaluation.startTime[firstLayer[i]][secondLayer[i]][processOrder] = \
                            empty_time[0] + self.wpstMatrix[empty_time[2]][presentWorkpiece]
                        empty_time[0] = evaluation.startTime[firstLayer[i]][secondLayer[i]][processOrder] + time
                        empty_time[2] = presentWorkpiece
                        break
                if evaluation.startTime[firstLayer[i]][secondLayer[i]][processOrder] == 0:
                    evaluation.startTime[firstLayer[i]][secondLayer[i]][processOrder] = \
                        evaluation.machineWorkTime[machineId] + self.wpstMatrix[evaluation.machineWorkPiece[machineId]][presentWorkpiece]

            # 工序既不是机器的第一道工序，也不是工件的第一道工序
            else:
                for empty_time in evaluation.machineEmptyTime[machineId]:
                    if empty_time[2] == -1:
                        continue
                    start_time = max(empty_time[0] + self.wpstMatrix[empty_time[2]][presentWorkpiece],
                                     evaluation.endTime[firstLayer[i]][secondLayer[i]][processOrder - 1])
                    end_time = start_time + time + self.wpstMatrix[preWorkpiece][empty_time[3]]
                    if end_time <= empty_time[1]:
                        evaluation.startTime[firstLayer[i]][secondLayer[i]][processOrder] = start_time
                        empty_time[0] = start_time + time
                        empty_time[2] = presentWorkpiece
                        break
                if evaluation.startTime[firstLayer[i]][secondLayer[i]][processOrder] == 0:
                    machineTime = evaluation.machineWorkTime[machineId] + \
                                  self.wpstMatrix[evaluation.machineWorkPiece[machineId]][presentWorkpiece]
                    preProcessTime = evaluation.endTime[firstLayer[i]][secondLayer[i]][processOrder - 1]
                    if machineTime >= preProcessTime:
                        evaluation.startTime[firstLayer[i]][secondLayer[i]][processOrder] = machineTime
                    else:
                        evaluation.startTime[firstLayer[i]][secondLayer[i]][processOrder] = preProcessTime
                        emptyStart = evaluation.machineWorkTime[machineId]
                        emptyEnd = preProcessTime
                        preWorkpiece = evaluation.machineWorkPiece[machineId]
                        aftWorkpiece = presentWorkpiece
                        evaluation.machineEmptyTime[machineId].append(np.array([emptyStart, emptyEnd, preWorkpiece, aftWorkpiece]))
            # 解码
            evaluation.endTime[firstLayer[i]][secondLayer[i]][processOrder] = evaluation.startTime[firstLayer[i]][secondLayer[i]][processOrder] + time
            evaluation.machineWorkTime[machineId] = max(evaluation.machineWorkTime[machineId], evaluation.endTime[firstLayer[i]][secondLayer[i]][processOrder])
            evaluation.machineWorkPiece[machineId] = presentWorkpiece
            evaluation.fulfillTime = max(evaluation.fulfillTime, evaluation.machineWorkTime[machineId])
        return evaluation


    # 计算适应度
    def calculate_fitness(self, g: Gene) -> float:
        return 1 / self.evaluate_gene(g).fulfillTime

    # 两个基因交叉 双点交叉
    def gene_cross(self, g1: Gene, g2: Gene) -> Gene:
        chromosomeSize = len(self.processList)
        orderNumber = len(self.orderList)
        workpieceNumber = len(self.workpieceList)
        pos1 = randint(0, chromosomeSize - 1)
        pos2 = randint(0, chromosomeSize - 1)
        start = min(pos1, pos2)
        end = max(pos1, pos2)
        # 存储第二层信息
        secondLayerRecord = [[] for _ in range(orderNumber)]
        for i in range(g1.length):
            secondLayerRecord[g1.first_layer[i]].append(g1.second_layer[i])
        # 存储第三层信息
        thirdLayerRecord = {}
        processCount = [[0 for _ in range(workpieceNumber)] for _ in range(orderNumber)]
        for i in range(g1.length):
            processOrder = processCount[g1.first_layer[i]][g1.second_layer[i]]
            key = (g1.first_layer[i],g1.second_layer[i],processOrder)
            thirdLayerRecord[key] = g1.third_layer[i]
            processCount[g1.first_layer[i]][g1.second_layer[i]] += 1
        # 双点交叉
        prototype = g1.first_layer[start: end + 1]
        t = g2.first_layer[0:]
        for v in prototype:
            for i in range(len(t)):
                if v == t[i]:
                    t.pop(i)
                    break
        firstLayer = t[0:start] + prototype + t[start:]
        # 第二层第三层染色体调整
        secondLayer = [-1] * chromosomeSize
        thirdLayer = [-1] * chromosomeSize
        orderCount = [0] * orderNumber
        for i in range(chromosomeSize):
            secondLayer[i] = secondLayerRecord[firstLayer[i]][orderCount[firstLayer[i]]]
            orderCount[firstLayer[i]] += 1
        processCountNew = [[0 for _ in range(workpieceNumber)] for _ in range(orderNumber)]
        for i in range(chromosomeSize):
            processOrderNew = processCountNew[firstLayer[i]][secondLayer[i]]
            indexTuple = (firstLayer[i], secondLayer[i], processOrderNew)
            thirdLayer[i] = thirdLayerRecord[indexTuple]
            processCountNew[firstLayer[i]][secondLayer[i]] += 1
        # 交叉结束，计算child基因
        child = Gene()
        child.length = g1.length
        child.first_layer = firstLayer
        child.second_layer = secondLayer
        child.third_layer = thirdLayer
        child.fitness = self.calculate_fitness(child)
        # child.print_gene()
        # child.fitness = self.calculate_fitness(child)
        # child.print_gene()
        return child

    # 基因变异 前两层：交换变异 第三层：基本位变异
    def gene_mutation(self, g: Gene) -> None:
        chromosomeSize = g.length
        orderNumber = len(self.orderList)
        workpieceNumber = len(self.workpieceList)
        # 存储第三层信息
        thirdLayerRecord = {}
        processCount = [[0 for _ in range(workpieceNumber)] for _ in range(orderNumber)]
        for i in range(chromosomeSize):
            processOrder = processCount[g.first_layer[i]][g.second_layer[i]]
            key = (g.first_layer[i],g.second_layer[i],processOrder)
            thirdLayerRecord[key] = g.third_layer[i]
            processCount[g.first_layer[i]][g.second_layer[i]] += 1
        # 开始变异
        # 交换变异
        pos1 = randint(0, chromosomeSize - 1)
        pos2 = randint(0, chromosomeSize - 1) # 确定两个变异点
        g.first_layer[pos1], g.first_layer[pos2] = g.first_layer[pos2], g.first_layer[pos1]
        g.second_layer[pos1], g.second_layer[pos2] = g.second_layer[pos2], g.second_layer[pos1]
        # g.third_layer[pos1], g.third_layer[pos2] = g.third_layer[pos2], g.third_layer[pos1]
        # 基本位变异
        pos3 = randint(0, chromosomeSize - 1) # 确定基本位变异点
        processCountNew = [[0 for _ in range(workpieceNumber)] for _ in range(orderNumber)]
        for i in range(chromosomeSize):
            processOrderNew = processCountNew[g.first_layer[i]][g.second_layer[i]]
            if i == pos3:
                # processMachine = self.orderWorkpiece[g.first_layer[i]][g.second_layer[i]].machine[processOrderNew]
                processTime = self.orderWorkpiece[g.first_layer[i]][g.second_layer[i]].time[processOrderNew]
                machineProbability = [1/i for i in processTime]
                sumProbability = sum(machineProbability)
                indexDict = {}
                for j, probability in enumerate(machineProbability):
                    indexDict[j] = probability
                u = random() * sumProbability
                sum_ = 0
                for k in range(len(machineProbability)):
                    sum_ += indexDict[k]
                    if sum_ >= u:
                        g.third_layer[i] = k
                        break
            else:
                indexTuple = (g.first_layer[i], g.second_layer[i], processOrderNew)
                g.third_layer[i] = thirdLayerRecord[indexTuple]
            processCountNew[g.first_layer[i]][g.second_layer[i]] += 1
        g.fitness = self.calculate_fitness(g)

    # 随机初始化种群
    def init_population(self) -> None:
        for _ in range(self.populationNumber):
            g = Gene()
            processNumber = len(self.processList)
            index_list = list(range(processNumber))
            firstLayer = [-1] * processNumber
            secondLayer = [-1] * processNumber
            thirdLayer = [-1] * processNumber
            for i in range(len(self.orderList)):
                for j in range(len(self.orderWorkpiece[i])):
                    for k in range(len(self.orderWorkpiece[i][j].process)):
                        val = choice(index_list)
                        index_list.remove(val)
                        firstLayer[val] = i
                        secondLayer[val] = j
            processCount = [[0 for _ in range(len(self.workpieceList))] for _ in range(len(self.orderList))]
            for i in range(processNumber):
                order = firstLayer[i]
                workpieceOrder = secondLayer[i]
                processOrder = processCount[order][workpieceOrder]
                machineNumber = len(self.orderWorkpiece[order][workpieceOrder].machine[processOrder])
                thirdLayer[i] = randint(0, machineNumber - 1)
                processCount[order][workpieceOrder] += 1
            g.length = processNumber
            g.first_layer = firstLayer
            g.second_layer = secondLayer
            g.third_layer = thirdLayer
            g.fitness = self.calculate_fitness(g)
            # g.print_gene()
            self.genes.append(g)

    # 从当前种群中选择k个个体， 轮盘赌法
    def select_gene(self, k) -> List[Gene]:
        genes = sorted(self.genes, key=lambda x: x.fitness)
        sumFitness = sum(g.fitness for g in genes)
        chosen = []
        for i in range(k):
            u = random() * sumFitness
            sum_ = 0
            for gene in genes:
                sum_ += gene.fitness
                if sum_ >= u:
                    chosen.append(gene)
                    break
        return chosen

    # 选择当前种群中最佳个体
    def select_best(self) -> Gene:
        genes = sorted(self.genes, key=lambda x: x.fitness, reverse=True)
        return genes[0]

    # GeneticAlgorithm_main
    def exec(self, parameter: List[List[WorkPiece]]):
        # 初始化
        self.orderWorkpiece = parameter
        self.init_population()

        fitnessList = [self.select_best().fitness]
        averageFitness = [sum(map(lambda v: v.fitness, self.genes)) / len(self.genes)]
        bestGene = Gene()
        # 开始进化
        #print("------ Start of evolution ------")
        for generate in range(self.times):
            #print("################### Generation {} ###################".format(generate))
            genes = sorted(self.genes, key=lambda x: x.fitness, reverse=True)
            retainNumber = int(self.populationNumber * 0.7)
            optNumber = self.populationNumber - retainNumber
            newGenes : list = genes[0:retainNumber]
            count = 0
            while count <= optNumber: # 通过交叉补全种群
                chosen = self.select_gene(2)
                if random() < self.crossProbability:
                    crossGene1 = self.gene_cross(chosen[0],chosen[1])
                    crossGene2 = self.gene_cross(chosen[1],chosen[0])
                    newGenes.append(crossGene1)
                    newGenes.append(crossGene2)
                    count += 2
            for gene in newGenes[1:]: # 变异
                if random() < self.mutationProbability:
                    self.gene_mutation(gene)
            self.genes = newGenes

            generateBest = self.select_best()
            fitnessList.append(generateBest.fitness)
            averageFitness.append(sum(map(lambda v: v.fitness, self.genes)) / len(self.genes))
            if generateBest.fitness > bestGene.fitness:
                bestGene = generateBest
            #print("Max fitness of current generate: {}".format(generateBest.fitness))

        #print("------ End of evolution ------")

        # bestGene.print_gene()
        result = self.evaluate_gene(bestGene)

        rowData = []
        processCount = [[0 for _ in range(len(self.workpieceList))] for _ in range(len(self.orderList))]
        for i in range(bestGene.length):
            orderId = bestGene.first_layer[i]
            workpieceId = self.orderWorkpiece[orderId][bestGene.second_layer[i]].workpieceIndex
            processOrder = processCount[orderId][bestGene.second_layer[i]]
            processId = self.orderWorkpiece[orderId][bestGene.second_layer[i]].process[processOrder]
            machineId = result.processMachine[processId]
            processCount[orderId][bestGene.second_layer[i]] += 1
            temp = {
                "order": self.orderList[orderId],
                "workpiece": self.workpieceList[workpieceId],
                "process": self.processList[processId],
                "machine": self.machineList[machineId],
                "startTime": result.startTime[orderId][bestGene.second_layer[i]][processOrder],
                "endTime": result.endTime[orderId][bestGene.second_layer[i]][processOrder]
            }
            rowData.append(temp)
        return rowData, result, fitnessList, averageFitness


if __name__ == "__main__":
    # 测试数据
    test = [{'order':'#o-1','workpiece':'#w-1','number':10000,'process':'#p-111','machine':['#m-1','#m-2','#m-3','#m-4','#m-5'],'time':[300,300,300,280,280]},
               {'order':'#o-1','workpiece':'#w-1','number':10000,'process':'#p-112','machine':['#m-6','#m-7','#m-8','#m-9'],'time':[40,40,40,40]},
               {'order':'#o-1','workpiece':'#w-2','number':10000,'process':'#p-121','machine':['#m-1','#m-2','#m-3','#m-4'],'time':[180,180,180,180]},
               {'order':'#o-1','workpiece':'#w-2','number':10000,'process':'#p-122','machine':['#m-6','#m-7','#m-8','#m-9'],'time':[40,40,40,40]},
               {'order':'#o-1','workpiece':'#w-3','number':8000,'process':'#p-131','machine':['#m-1','#m-2','#m-3','#m-5'],'time':[340,340,350,350]},
               {'order':'#o-1','workpiece':'#w-3','number':8000,'process':'#p-132','machine':['#m-6','#m-7','#m-8','#m-9'],'time':[40,38,40,38]},
               {'order':'#o-1','workpiece':'#w-3','number':8000,'process':'#p-133','machine':['#m-10'],'time':[20]},
               {'order':'#o-2','workpiece':'#w-4','number':1000,'process':'#p-241','machine':['#m-1','#m-2','#m-3','#m-4','#m-5'],'time':[290,290,285,285,290]},
               {'order':'#o-2','workpiece':'#w-4','number':1000,'process':'#p-242','machine':['#m-6','#m-7','#m-9'],'time':[40,40,40]},
               {'order':'#o-2','workpiece':'#w-5','number':10000,'process':'#p-251','machine':['#m-1','#m-2','#m-3','#m-4'],'time':[184,184,180,184]},
               {'order':'#o-2','workpiece':'#w-5','number':10000,'process':'#p-252','machine':['#m-6','#m-8','#m-9'],'time':[40,40,40]},
               {'order':'#o-2','workpiece':'#w-6','number':10000,'process':'#p-261','machine':['#m-4','#m-5'],'time':[140,140]},
               {'order':'#o-2','workpiece':'#w-6','number':10000,'process':'#p-262','machine':['#m-7','#m-8','#m-9'],'time':[20,20,20]},
               {'order':'#o-3','workpiece':'#w-1','number':8000,'process':'#p-311','machine':['#m-1','#m-2','#m-3','#m-4','#m-5'],'time':[300,300,300,280,280]},
               {'order':'#o-3','workpiece':'#w-1','number':8000,'process':'#p-312','machine':['#m-6','#m-7','#m-8','#m-9'],'time':[40,40,40,40]},
               {'order':'#o-3','workpiece':'#w-3','number':8000,'process':'#p-331','machine':['#m-1','#m-2','#m-3','#m-5'],'time':[340,340,350,350]},
               {'order':'#o-3','workpiece':'#w-3','number':8000,'process':'#p-332','machine':['#m-6','#m-7','#m-8','#m-9'],'time':[40,38,40,38]},
               {'order':'#o-3','workpiece':'#w-3','number':8000,'process':'#p-333','machine':['#m-10'],'time':[20]},
               {'order':'#o-3','workpiece':'#w-7','number':1200,'process':'#p-371','machine':['#m-3','#m-4','#m-5'],'time':[660,660,660]},
               {'order':'#o-3','workpiece':'#w-7','number':1200,'process':'#p-372','machine':['#m-7','#m-8','#m-9','#m-10'],'time':[40,40,40,40]}]
    wpstMatrix = np.array([[0, 10, 10, 10, 10, 10, 10],
                           [10, 0, 20, 20, 20, 20, 20],
                           [10, 20, 0, 15, 15, 15, 15],
                           [10, 20, 15, 0, 15, 15, 15],
                           [15, 20, 10, 15, 0, 10, 10],
                           [10, 20, 15, 15, 10, 0, 10],
                           [10, 20, 15, 15, 10, 10, 0]])
    workpieceIndex = ['#w-1','#w-2','#w-3','#w-4','#w-5','#w-6','#w-7']
    # 注 ： 数字越小代表优先级越高
    orderPriority = np.array([2, 1, 2])
    orderIndex = ['#o-1', '#o-2', '#o-3']
    orderWorkpiece, orderList, workpieceList, processList, machineList = reshape_data(test)
    idxWorkpiece = np.array([workpieceIndex.index(x) for x in workpieceList], dtype=int)
    wpstMatrix = wpstMatrix[idxWorkpiece, :][:, idxWorkpiece]
    idxOrder = np.array([orderIndex.index(x) for x in orderList], dtype=int)
    orderPriority = orderPriority[idxOrder]
    ga = GeneticAlgorithm(orderList, workpieceList, processList, machineList, wpstMatrix, orderPriority)
    rowData, bestGene, fitnessList, averageFitness = ga.exec(orderWorkpiece)
    x = [i for i in range(len(fitnessList))]
    plt.cla()
    plt.figure(figsize=(12, 10))
    plt.title('实验迭代曲线图', fontdict={'weight': 'normal', 'size': 16})
    plt.plot(x, fitnessList)
    plt.plot(x, averageFitness)
    plt.ylabel('进化代数', fontdict={'weight': 'normal', 'size': 16})
    plt.xlabel('适应度值', fontdict={'weight': 'normal', 'size': 16})
    #plt.savefig('./img/curve.svg')
    plt.show()
    #plt.close()
    plt.cla()
    #print(rowData)
    print(bestGene.fulfillTime)
    plt.figure(figsize=(18, 10))
    draw_gantt(rowData)
    #plt.savefig('./img/gantt.svg')
    plt.show()
    #plt.close()