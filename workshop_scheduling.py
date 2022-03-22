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

MATRIX_SIZE = 500


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
    def __init__(self):
        self.fulfillTime = 0
        self.machineWorkTime = [0 for _ in range(MATRIX_SIZE)]
        self.processIds = [0 for _ in range(MATRIX_SIZE)]
        self.endTime = [[0 for _ in range(MATRIX_SIZE)] for _ in range(MATRIX_SIZE)]
        self.startTime = [[0 for _ in range(MATRIX_SIZE)] for _ in range(MATRIX_SIZE)]


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
        self.orderProcess = None
        self.orderMachine = None

        self.genes: Set[Gene] = set()

    # 评估基因
    # 固定优化基因
    def evaluate_gene(self, g: Gene) -> GeneEvaluation:
        pass

    # 计算适应度
    def calculate_fitness(self, g: Gene) -> float:
        pass

    # 两个基因交叉
    def gene_cross(self, g1: Gene, g2: Gene) -> Gene:
        pass

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
                    for _ in range(len(self.orderWorkpiece[i][j].process)):
                        val = choice(index_list)
                        index_list.remove(val)
                        firstLayer[val] = i
                        secondLayer[val] = j
                        machineNumber = len(self.orderWorkpiece[i][j].machine)
                        thirdLayer[val] = randint(0, machineNumber-1)
            g.length = processNumber
            g.first_layer = firstLayer
            g.second_layer = secondLayer
            g.third_layer = thirdLayer
            # g.fitness =
            g.print_gene()
            self.genes.add(g)


    def select_gene(self) -> Gene:
        pass

    def exec(self, parameter: List[List[WorkPiece]]):
        self.orderWorkpiece = parameter
        self.init_population()


if __name__ == "__main__":
    # 测试数据
    test = [{'order':'#o-1','workpiece':'#w-1','number':'100','process':'#p-111','machine':['#m-1','#m-2'],'time':[65,70],'sequences':0},
            {'order':'#o-1','workpiece':'#w-1','number':'100','process':'#p-112','machine':['#m-7'],'time':[40],'sequences':1},
            {'order':'#o-1','workpiece':'#w-2','number':'10','process':'#p-121','machine':['#m-1','#m-2','#m-3'],'time':[30,40,35],'sequences':0},
            {'order':'#o-1','workpiece':'#w-2','number':'10','process':'#p-122','machine':['#m-7'],'time':[30],'sequences':1},
            {'order':'#o-2','workpiece':'#w-3','number':'100','process':'#p-231','machine':['#m-3','#m-4'],'time':[69,70],'sequences':0},
            {'order':'#o-2','workpiece':'#w-3','number':'100','process':'#p-232','machine':['#m-8'],'time':[25],'sequences':1},
            {'order':'#o-3','workpiece':'#w-4','number':'10','process':'#p-341','machine':['#m-4','#m-5'],'time':[145,140],'sequences':0},
            {'order':'#o-3','workpiece':'#w-4','number':'10','process':'#p-342','machine':['#m-6'],'time':[10],'sequences':1},
            {'order':'#o-3','workpiece':'#w-4','number':'10','process':'#p-343 ','machine':['#m-8'],'time':[17],'sequences':2}]
    orderWorkpiece, orderList, workpieceList, processList, machineList = reshape_data(test)
    ga = GeneticAlgorithm(orderList, workpieceList, processList, machineList)
    ga.exec(orderWorkpiece)
