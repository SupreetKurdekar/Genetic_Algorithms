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

# print("Number of processors: ", mp.cpu_count())
# print(2/3)
# listed = [1,4,6,20]
# print(max(listed))

# Optimisation Parameters ##################
num_var = 2
num_init_designs = 100
variable_bounds = np.array([[5.12,5.12],[-5.12,-5.12]])
listOfPrecisionDigits = [3,3]
# listOfVarStringLengths = [9,14,17,24]
crossoverProbability = 0.8
mutationProbability = 0.01
beta = 1 # fitness function parameter
maxNumIterations = 4000
numOfExperimentRepetitions = 10
listOfVariableHeadings = ['GenerationNumber','FunctionValue','FitnessValue','ConstraintViolation']
metaDfVariableHeadings = ['GenerationNumber','MinFunctionValue','AverageFunctionValue','BestFitnessValue','AverageFitness','BestIndexNumber']
#________________________________________________________________________________#

#Generating and storing initial real population of solutions
# latin hypercube sampling
init_designs = lhs(num_var, num_init_designs)
# print(init_designs)
# generate population from lhs
population = utils.pop_generator(init_designs,listOfPrecisionDigits)

# equalise variable string lengths
population = utils.equaliseVariableStringLength(population,num_var)
# print("pop",population)

# make dataframe
ResultsDf = pd.Dataframe()
df = pd.DataFrame()
metaDf = pd.DataFrame(columns=metaDfVariableHeadings)
interations = 0
print("init look of df",df)

for i = in range(len(numOfExperimentRepetitions)):
    
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
        # print(metaDf,"meta")
        #'BestFitnessValue','AverageFitness','BestIndexNumber']
        # plotting
        # metaDf.plot(, ax = ax) # Pass the axes to plot. 

        # print("max fitness value",df['FitnessValue'].idxmax())
        mating_pool = utils.suss(fitness_scores,population,listOfVarStringLengths)
        # selection - mating pool generation 

        # crossover of mating pool
        utils.generalCrossover(crossoverProbability,mating_pool,utils.singlePointCrossover,listOfVarStringLengths,listOfPrecisionDigits)

        # mutation of new mating pool
        utils.mutation(mating_pool,mutationProbability,listOfVarStringLengths,listOfPrecisionDigits)

        population = mating_pool
        
        interations = interations + 1

f1 = plt.figure(1)
ax = plt.gca()
metaDf.plot(kind='line',x='GenerationNumber',y='AverageFunctionValue',color='blue',ax=ax)
metaDf.plot(kind='line',x='GenerationNumber',y='MinFunctionValue',color='red',ax=ax)

f2 = plt.figure(2)
ax = plt.gca()
metaDf.plot(kind='line',x='GenerationNumber',y='AverageFitness',color='blue',ax=ax)
metaDf.plot(kind='line',x='GenerationNumber',y='BestFitnessValue',color='red',ax=ax)
plt.show()