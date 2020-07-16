from itertools import islice
import math
# needs to account for constraints too
def my_fit_fcn(design):
    sum = 0
    for var in design:
        sum = sum + var[0]

    return sum

def constraint_fcn(design):

    return sum

def deJongsFcn(design):
    sum = 0
    for var in design:
        sum = sum + var[0]**2
    return sum

def RosenBorocksFcn(design):
#     f2(x)=sum(100·(x(i+1)-x(i)^2)^2+(1-x(i))^2)
#    i=1:n-1; -2.048<=x(i)<=2.048.
    # print("design",design)
    sum = 0
    id = 0
    # for id in range(len(design))-1: Look this up
    # for item in islice(design, len(design) - 1):
    #     sum = sum + 100*(item[0]-item[0]**2)**2 + (1-item)
    #     sum = sum + 100*(design[id+1][0]-(design[id][0])**2)**2+(1-design[id][0])**2
    #     print(sum)
    while id < len(design)-1:
        sum = sum + 100*(design[id+1][0]-(design[id][0])**2)**2+(1-design[id][0])**2
        id = id + 1
    return sum

def RastriginFcn(design):
    #f6(x)=10·n+sum(x(i)^2-10·cos(2·pi·x(i))), i=1:n; -5.12<=x(i)<=5.12.
    # print("design",design)
    sum = 0
    id = 0
    while id < len(design):
        # sum = sum + 100*(design[id+1][0]-(design[id][0])**2)**2+(1-design[id][0])**2
        sum = sum + design[id][0]**2-10*(math.cos(1*math.pi*design[id][0]))
        id = id + 1
    return sum + 10*len(design)

def schwefelFcn(design):
    # f7(x)=sum(-x(i)·sin(sqrt(abs(x(i))))), i=1:n; -500<=x(i)<=500.
    sum = 0
    id = 0
    while id < len(design):
        # sum = sum + 100*(design[id+1][0]-(design[id][0])**2)**2+(1-design[id][0])**2
        sum = sum + (-design[id][0]*math.sin(math.sqrt(math.fabs(design[id][0]))))
        id = id + 1
    return sum

def griewangkFcn(design):
#     f8(x)=sum(x(i)^2/4000)-prod(cos(x(i)/sqrt(i)))+1, i=1:n
#    -600<=x(i)<= 600
    sum = 0
    prod = 1
    id = 0
    while id < len(design):
        sum = sum + ((design[id][0]**2)/4000)
        prod = prod*math.cos(design[id][0])/math.sqrt(id+1)
        id = id + 1

    return sum - prod + 1

def ackleysFcn(design):
    a = 20
    b = 0.2
    c = 2*math.pi

    sum1 = 0
    sum2 = 0
    id = 0
    while id < len(design):
        sum1 = sum1 + design[id][0]**2
        sum2 = sum2 + math.cos(c*design[id][0])
        id = id + 1

    return -a*math.exp(-b*math.sqrt(sum1/len(design)))-math.exp(sum2/len(design))+a+math.exp(1)

def LiAndSunFcn(design):
    sum1 = 0
    sum2 = 0
    id = 0
    while id < len(design):
        sum1 = sum1 + (design[id][0])**4
        id = id + 1
    
    id = 0
    while id < len(design):
        sum2 = sum2 + design[id][0]
        id = id + 1

    return sum1 + sum2**2


#   f10(x)=-a·exp(-b·sqrt(1/n·sum(x(i)^2)))-exp(1/n·sum(cos(c·x(i))))+a+exp(1)
#    a=20; b=0.2; c=2·pi; i=1:n; -32.768<=x(i)<=32.768.
# global minimum:

#   f(x)=0; x(i)=0, i=1:n.

def variableBoundsConstraint(design,variable_bounds):
    gx = 0
    for id,var in enumerate(design):
        if var[0] < variable_bounds[1][id]:
            gx = gx + variable_bounds[1][id] - var[0]
        elif var[0] > variable_bounds[0][id]:
            gx = gx - variable_bounds[0][id] + var[0]

    return gx