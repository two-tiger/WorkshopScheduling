#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Xinquan Chen
@contact:XinquanChen0117@163.com
@file: workshop_scheduling.py
@time: 2022-03-08 13:59
"""
from random import randint, choice
from typing import List, Tuple, Set, Dict, Any
from reshape_tool import reshape_data, WorkPiece
from draw_gantt_chart import draw_gantt

SIZE = 10


# 三层编码基因
class Gene(object):
    def __init__(self, fitness: float = 0, length=0, first_layer=None, second_layer=None, third_layer=None):
        self.fitness = fitness
        self.length = length
        self.first_layer: list = first_layer
        self.second_layer: list = second_layer
        self.third_layer: list = third_layer

    # def __eq__(self, other):
    #     if isinstance(other, Gene):
    #         return other.fitness == self.fitness and other.chromosome == self.chromosome
    #     return False
    #
    # def __hash__(self):
    #     return hash("".join(map(lambda x: str(x), self.chromosome)))
    #
    # def __str__(self):
    #     return "{} => {}".format(self.chromosome, self.fitness)

    def print_gene(self):
        print("chromosome: ", self.first_layer, self.second_layer, self.third_layer)


# 存储解码结果
class GeneEvaluation():
    def __init__(self, processNumber, machineNumber, orderNumber, workpieceNumber):
        self.fulfillTime = 0
        self.processMachine = [0 for _ in range(processNumber)]
        self.machineWorkTime = [0 for _ in range(machineNumber)]
        self.processIds = [[0 for _ in range(SIZE)] for _ in range(orderNumber)]
        self.endTime = [[0 for _ in range(SIZE)] for _ in range(workpieceNumber)]
        self.startTime = [[0 for _ in range(SIZE)] for _ in range(workpieceNumber)]


# # 生产计划树对象
# class ProductionPlanningTree():
#     def __init__(self, rootObj):
#         self.key = rootObj
#         self.leftChild = None
#         self.rightChild = None
#
#     def generate_order_matrix(self):
#         pass
#
#     def generate_gene(self, orderMatrix):
#         pass


class GeneticAlgorithm():
    def __init__(self, orderList, workpieceList, processList, machineList, populationNumber=50, times=20, crossProbability=0.95,
                 mutationProbability=0.05):
        self.populationNumber = populationNumber  # 种群数量
        self.times = times  # 遗传代数
        self.crossProbability = crossProbability  # 交叉概率
        self.mutationProbability = mutationProbability  # 变异概率

        self.orderList = orderList
        self.workpieceList = workpieceList
        self.processList = processList
        self.machineList = machineList

        self.orderWorkpiece = None

        self.genes: Set[Gene] = set()

    # 评估基因长度
    def evaluate_gene(self, g: Gene) -> GeneEvaluation:
        processNumber = len(self.processList)
        machineNumber = len(self.machineList)
        orderNumber = len(self.orderList)
        workpieceNumber = len(self.workpieceList)
        evaluation = GeneEvaluation(processNumber,machineNumber,orderNumber,workpieceNumber)
        for index, orderId in enumerate(g.first_layer):
            workpieceOrder = g.second_layer[index]
            workpieceId = self.orderWorkpiece[orderId][workpieceOrder].workpieceIndex
            processOrder = evaluation.processIds[orderId][workpieceOrder]
            processId = self.orderWorkpiece[orderId][workpieceOrder].process[processOrder]
            machineOrder = g.third_layer[index]
            machineId = self.orderWorkpiece[orderId][workpieceOrder].machine[processOrder][machineOrder]
            evaluation.processMachine[processId] = machineId
            time = self.orderWorkpiece[orderId][workpieceOrder].time[processOrder][machineOrder]
            time *= self.orderWorkpiece[orderId][workpieceOrder].number
            evaluation.processIds[orderId][workpieceOrder] += 1
            if processOrder == 0:
                evaluation.startTime[workpieceId][processOrder] = evaluation.machineWorkTime[machineId]
            else:
                evaluation.startTime[workpieceId][processOrder] = max(evaluation.endTime[workpieceId][processOrder - 1],
                                                                      evaluation.machineWorkTime[machineId])
            evaluation.machineWorkTime[machineId] = evaluation.startTime[workpieceId][processOrder] + time
            evaluation.endTime[workpieceId][processOrder] = evaluation.machineWorkTime[machineId]
            evaluation.fulfillTime = max(evaluation.fulfillTime, evaluation.machineWorkTime[machineId])
        return evaluation

    # 计算适应度
    def calculate_fitness(self, g: Gene) -> float:
        return 1 / self.evaluate_gene(g).fulfillTime

    # 两个基因交叉
    def gene_cross(self, g1: Gene, g2: Gene) -> Gene:
        pos0 = randint(0, len(self.processList) - 1)


    # 基因变异
    def gene_mutation(self, g: Gene) -> None:
        pass

    # 初始化种群
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
            processCount = [[0 for _ in range(SIZE)] for _ in range(len(self.orderList))]
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
            self.genes.add(g)


    def select_gene(self) -> Gene:
        pass

    def exec(self, parameter: List[List[WorkPiece]]):
        self.orderWorkpiece = parameter
        self.init_population()

        # 交叉
        # 变异

        bestGene = Gene()
        for gene in self.genes:
            if bestGene.fitness < gene.fitness:
                bestGene = gene

        result = self.evaluate_gene(bestGene)

        rowData = []
        for i in range(len(self.orderList)):
            for j in range(len(self.orderWorkpiece[i])):
                for k in range(len(self.orderWorkpiece[i][j].process)):
                    workpieceId = self.orderWorkpiece[i][j].workpieceIndex
                    processId = self.orderWorkpiece[i][j].process[k]
                    machineId = result.processMachine[processId]
                    temp = {
                        "order": self.orderList[i],
                        "workpiece": self.workpieceList[workpieceId],
                        "process": self.processList[processId],
                        "machine": self.machineList[machineId],
                        "startTime": result.startTime[workpieceId][k],
                        "endTime": result.endTime[workpieceId][k]
                    }
                    rowData.append(temp)
        return rowData, result

if __name__ == "__main__":
    # 测试数据
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
    orderWorkpiece, orderList, workpieceList, processList, machineList = reshape_data(test)
    ga = GeneticAlgorithm(orderList, workpieceList, processList, machineList)
    rowData, bestGene = ga.exec(orderWorkpiece)
    #print(rowData)
    print(bestGene.fulfillTime)
    draw_gantt(rowData)
