# WorkshopScheduling

This is a project to solve the production scheduling problem of job shop based on three-layer coding genetic algorithm.
Take Dalian R enterprise as an example.

## File description

1. workshop_scheduling.py

Genetic algorithm core code.

2. reshape_tool.py

Convert input data to standard format.

3. draw_gantt_chart.py

Draw a Gantt chart.

## Dependencies

* Python(3.8)
* some necessary libraries

## How to Use

The input data is shown as:

```python
test = [{'order':'#o-1','workpiece':'#w-1','number':100,'process':'#p-111','machine':['#m-1','#m-2'],'time':[65,70]},
        {'order':'#o-1','workpiece':'#w-1','number':100,'process':'#p-112','machine':['#m-7'],'time':[40]},
        {'order':'#o-1','workpiece':'#w-2','number':10,'process':'#p-121','machine':['#m-1','#m-2','#m-3'],'time':[30,40,35]},
        {'order':'#o-1','workpiece':'#w-2','number':10,'process':'#p-122','machine':['#m-7'],'time':[30]},
        {'order':'#o-2','workpiece':'#w-3','number':100,'process':'#p-231','machine':['#m-3','#m-4'],'time':[69,70]},
        {'order':'#o-2','workpiece':'#w-3','number':100,'process':'#p-232','machine':['#m-8'],'time':[25]},
        {'order':'#o-3','workpiece':'#w-4','number':10,'process':'#p-341','machine':['#m-4','#m-5'],'time':[145,140]},
        {'order':'#o-3','workpiece':'#w-4','number':10,'process':'#p-342','machine':['#m-6'],'time':[10]},
        {'order':'#o-3','workpiece':'#w-4','number':10,'process':'#p-343 ','machine':['#m-8'],'time':[17]}]
```

**Note:** The 'process' name must be separate and unique. 'Orders' and 'workpieces' should preferably be written in order.

You can run "workshop_scheduling.py" directly or change the relevant parameters of genetic algorithm.

## Project framework

- [X]  reshaping data
- [X]  Implementation of Genetic Algorithm
  - [X]  Three-layer coding
  - [X]  Initializing the population
  - [X]  Evaluating genes
  - [X]  Calculating fitness
  - [X]  Cross
  - [X]  Mutation
  - [X]  Selecting gene
- [ ]  Add transfer batch
- [X]  Add workpiece switching time
- [X]  Draw gantt chart

## Actual situation of Dalian R enterprise

The input data is as follows:

```python
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
```

Input workpiece switching time matrix as follows:

```python
wpstMatrix = np.array([[0, 10, 10, 10, 10, 10, 10],
                       [10, 0, 20, 20, 20, 20, 20],
                       [10, 20, 0, 15, 15, 15, 15],
                       [10, 20, 15, 0, 15, 15, 15],
                       [15, 20, 10, 15, 0, 10, 10],
                       [10, 20, 15, 15, 10, 0, 10],
                       [10, 20, 15, 15, 10, 10, 0]])
```

At the same time, you must specify the matrix index order as input like this：

```python
workpieceIndex = ['#w-1','#w-2','#w-3','#w-4','#w-5','#w-6','#w-7']
```

## Result

#### Specified parameters:

populationNumber=50
times=50
crossProbability=0.95
mutationProbability=0.05

#### Fitness iteration curve:

![Fitness iteration curve](/img/Fitness_iteration_curve.png)

#### Gantt chart:

![Gantt chart](/img/gantt_chart.png)

#### Minimum time:

4521000
