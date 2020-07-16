import utils
import pyDOE
import numpy as np
from pyDOE import lhs
import random
import fitness_function
import pandas as pd
import multiprocessing as mp
from matplotlib import animation
from matplotlib import pyplot as plt
import time as tm

# Optimisation Parameters ##################
num_var = 2
num_init_designs = 100
variable_bounds = np.array([[5.12,5.12],[-5.12,-5.12]])
listOfPrecisionDigits = [3,3]
crossoverProbability = 0.7
mutationProbability = 0.00000001
beta = 1 # fitness function parameter
maxNumIterations = 100
numberOfRuns = 10
listOfVariableHeadings = ['GenerationNumber','FunctionValue','FitnessValue','ConstraintViolation']
metaDfVariableHeadings = ['GenerationNumber','MinFunctionValue','AverageFunctionValue','BestFitnessValue','AverageFitness','BestIndexNumber']
#________________________________________________________________________________#

# make dataframe

ResultsDf = pd.DataFrame()

#print("init look of df",df)
for i in range(numberOfRuns):
    print("Run no",i)
    #Generating and storing initial real population of solutions
    # latin hypercube sampling
    init_designs = lhs(num_var, num_init_designs)
    # print(init_designs)
    # generate population from lhs
    population = utils.pop_generator(init_designs,listOfPrecisionDigits)

    # equalise variable string lengths
    population = utils.equaliseVariableStringLength(population,num_var)
    # print("pop",population)
    df = pd.DataFrame()
    metaDf = pd.DataFrame(columns=metaDfVariableHeadings)
    interations = 0
    while interations < maxNumIterations:
        
        print("iteration number",interations)
        listOfVarStringLengths = []
        for var in population[0]:
            listOfVarStringLengths.append(len(var[1]))
        # print(listOfVarStringLengths)

        # print("popuation",population)
        # fitness evaluation score dictionary generation
        fitness_scores,df = utils.fitnessEvaluation(population,variable_bounds,
        beta,fitness_function.deJongsFcn,df,interations,listOfVariableHeadings,num_var)
        # print(fitness_scores,"fitness scores")

        MinFunctionValue = df.loc[df['GenerationNumber'] == interations,'FunctionValue'].min()
        AverageFunctionValue = df.loc[df['GenerationNumber'] == interations,'FunctionValue'].mean()
        MaxFitness = df.loc[df['GenerationNumber'] == interations,'FitnessValue'].max()
        variableValues = pd.DataFrame(df[(df['GenerationNumber']==interations) & (df['FitnessValue']==MaxFitness)].iloc[:,-num_var:])
        AverageFitness = df.loc[df['GenerationNumber'] == interations,'FitnessValue'].mean()
        CurrentDf = df[df['GenerationNumber']==interations]
        BestIndex = CurrentDf['FitnessValue'].idxmax()
        temp=pd.DataFrame([[interations,MinFunctionValue,AverageFunctionValue,MaxFitness,AverageFitness,BestIndex]], columns=metaDfVariableHeadings)    
        metaDf = pd.concat([metaDf,temp])

        mating_pool = utils.suss(fitness_scores,population,listOfVarStringLengths)
        # selection - mating pool generation 

        # crossover of mating pool
        utils.generalCrossover(crossoverProbability,mating_pool,utils.twoPointCrossover,listOfVarStringLengths,listOfPrecisionDigits)

        # mutation of new mating pool
        utils.mutation(mating_pool,mutationProbability,listOfVarStringLengths,listOfPrecisionDigits)
        population = mating_pool

        interations = interations + 1

    # print("minim",df.iloc[df["FunctionValue"].idxmin(),:])
    
    p = df.loc[df["FunctionValue"].idxmin()]
    ResultsDf = pd.concat([ResultsDf,pd.DataFrame(p).transpose()])
    #ResultsDf.append(p)
    print("Results",ResultsDf)

# best mean and std
best = ResultsDf["FunctionValue"].min()
print("Best",best)
mean = ResultsDf["FunctionValue"].mean()
print("mean",mean)
std = ResultsDf["FunctionValue"].std()
print("std",std)
f4 = plt.figure(4)
boxPlot = ResultsDf.boxplot(column=["FunctionValue"])

# plotting
f1 = plt.figure(1)
ax = plt.gca()
metaDf.plot(kind='line',x='GenerationNumber',y='AverageFunctionValue',color='blue',ax=ax)
metaDf.plot(kind='line',x='GenerationNumber',y='MinFunctionValue',color='red',ax=ax)

f2 = plt.figure(2)
ax = plt.gca()
metaDf.plot(kind='line',x='GenerationNumber',y='AverageFitness',color='blue',ax=ax)
metaDf.plot(kind='line',x='GenerationNumber',y='BestFitnessValue',color='red',ax=ax)

f3 = plt.figure(3)
ax = plt.gca()
df.plot(kind='line',x='GenerationNumber',y='ConstraintViolation',color='blue',ax=ax)
plt.show()