# WorkshopScheduling

这是一个基于遗传算法解决作业车间生产排产问题，以大连R企业为例

## 作者信息

**姓名**：陈鑫泉
**学校**：哈尔滨工程大学
**学院**：数学科学学院
**专业**：数学与应用数学
**目的**：本科毕设

## 使用

输入数据形如：

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


## 项目框架

- [X]  reshaping data
- [ ]  Implementation of Genetic Algorithm
  - [X]  Three-layer coding
  - [X]  Initializing the population
  - [X]  Evaluating genes
  - [X]  Calculating fitness
  - [ ]  Cross
  - [ ]  Mutation
  - [ ]  Selecting gene
- [ ]  Add transfer batch
- [ ]  Add workpiece switching time
- [X]  Draw gantt chart
