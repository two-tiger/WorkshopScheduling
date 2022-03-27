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
    def __init__(self, orderList, workpieceList, processList, machineList, populationNumber=50, times=100, crossProbability=0.95,
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

        self.genes: List[Gene] = []

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
        chromosomeSize = len(self.processList)
        orderNumber = len(self.orderList)
        workpieceNumber = len(self.workpieceList)
        pos1 = randint(0, chromosomeSize - 1)
        pos2 = randint(0, chromosomeSize - 1)
        start = min(pos1, pos2)
        end = max(pos1, pos2)
        secondLayerRecord = [[] for _ in range(orderNumber)]
        for i in range(g1.length):
            secondLayerRecord[g1.first_layer[i]].append(g1.second_layer[i])
        thirdLayerRecord = {}
        processCount = [[0 for _ in range(workpieceNumber)] for _ in range(orderNumber)]
        for i in range(g1.length):
            processOrder = processCount[g1.first_layer[i]][g1.second_layer[i]]
            key = (g1.first_layer[i],g1.second_layer[i],processOrder)
            thirdLayerRecord[key] = g1.third_layer[i]
            processCount[g1.first_layer[i]][g1.second_layer[i]] += 1
        prototype = g1.first_layer[start: end + 1]
        t = g2.first_layer[0:]
        for v in prototype:
            for i in range(len(t)):
                if v == t[i]:
                    t.pop(i)
                    break
        firstLayer = t[0:start] + prototype + t[start:]
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
        child = Gene()
        child.length = g1.length
        child.first_layer = firstLayer
        child.second_layer = secondLayer
        child.third_layer = thirdLayer
        # child.print_gene()
        # child.fitness = self.calculate_fitness(child)
        # child.print_gene()
        return child


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
            self.genes.append(g)

    # 选择个体， 轮盘赌法
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

    def select_best(self) -> Gene:
        genes = sorted(self.genes, key=lambda x: x.fitness, reverse=True)
        return genes[0]

    def exec(self, parameter: List[List[WorkPiece]]):
        self.orderWorkpiece = parameter
        self.init_population()

        fitnessList = [self.select_best().fitness]
        bestGene = Gene()
        print("------ Start of evolution ------")
        for generate in range(self.times):
            print("############### Generation {} ###############".format(generate))
            genes = sorted(self.genes, key=lambda x: x.fitness, reverse=True)
            retainNumber = int(self.populationNumber * 0.8)
            optNumber = self.populationNumber - retainNumber
            newGenes : list = self.genes[0:retainNumber]
            count = 0
            while count <= optNumber: # 通过交叉补全种群
                chosen = self.select_gene(2)
                if random() < self.crossProbability:
                    crossGene = self.gene_cross(chosen[0],chosen[1])
                    newGenes.append(crossGene)
                    count += 1
            for gene in newGenes: # 变异
                if random() < self.mutationProbability:
                    self.gene_mutation(gene)
            self.genes = newGenes

            generateBest = self.select_best()
            fitnessList.append(generateBest.fitness)
            if generateBest.fitness > bestGene.fitness:
                bestGene = generateBest
            print("Max fitness of current generate: {}".format(generateBest.fitness))

        print("------ End of evolution ------")

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
        return rowData, result, fitnessList

    def test(self, parameter: List[List[WorkPiece]]):
        self.orderWorkpiece = parameter
        self.init_population()
        gene1 = Gene()
        gene2 = Gene()
        for gene in self.genes:
            if gene1.fitness < gene.fitness:
                gene2 = gene1
                gene1 = gene
        gene1.print_gene()
        gene2.print_gene()
        child1 = Gene()
        child2 = Gene()
        child1 = self.gene_cross(gene1, gene2)
        child2 = self.gene_cross(gene2, gene1)

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
            {'order':'#o-2','workpiece':'#w-6','number':10000,'process':'#p-261','machine':['#m-4','#m-5'],'time':[140,140]},
            {'order':'#o-2','workpiece':'#w-6','number':10000,'process':'#p-262','machine':['#m-7','#m-8','#m-9'],'time':[20,20,20]},
            {'order':'#o-3','workpiece':'#w-1','number':8000,'process':'#p-311','machine':['#m-1','#m-2','#m-3','#m-4','#m-5'],'time':[300,300,300,280,280]},
            {'order':'#o-3','workpiece':'#w-1','number':8000,'process':'#p-312','machine':['#m-6','#m-7','#m-8','#m-9'],'time':[40,40,40,40]},
            {'order':'#o-3','workpiece':'#w-3','number':10000,'process':'#p-331','machine':['#m-1','#m-2','#m-3','#m-5'],'time':[340,340,350,350]},
            {'order':'#o-3','workpiece':'#w-3','number':10000,'process':'#p-332','machine':['#m-6','#m-7','#m-8','#m-9'],'time':[40,38,40,38]},
            {'order':'#o-3','workpiece':'#w-3','number':10000,'process':'#p-333','machine':['#m-10'],'time':[20]},
            {'order':'#o-3','workpiece':'#w-7','number':1200,'process':'#p-371 ','machine':['#m-3','#m-4','#m-5'],'time':[660,660,660]},
            {'order':'#o-3','workpiece':'#w-7','number':1200,'process':'#p-372 ','machine':['#m-7','#m-8','#m-9','#m-10'],'time':[40,40,40,40]}]
    orderWorkpiece, orderList, workpieceList, processList, machineList = reshape_data(test)
    ga = GeneticAlgorithm(orderList, workpieceList, processList, machineList)
    rowData, bestGene, fitnessList = ga.exec(orderWorkpiece)
    x = [i for i in range(len(fitnessList))]
    plt.plot(x, fitnessList)
    plt.show()
    print(bestGene.fulfillTime)
    draw_gantt(rowData)
