# file to compute the maximum k subset of d^2 bell states
# @oscars47

# using sympy to handle symbolic manipulation
from sympy import sqrt
from sympy.solvers import solve
from sympy import *
from sympy.physics.quantum.dagger import Dagger
import numpy as np

## Initialize d value ##

# d = int(input('Enter a value for d: '))
d=2

## Helper Functions for Bell State Generation ##

# generate bell states in number basis
# e.g. |01> for d = 2: [1, 0, 0, 1]
# key: cp (correlation class, phae class). e.g. 12 => c=1, p=2
bs_dict ={}
def generate_bs(d): 
    # store the phase in separate array to make measurement 
    # adapting equation 2 in the 2019
    for c in range(d): # set correlation class
        for p in range(d): # set phase class
            phase = []# stores phase for each join-particle ket
            # ignoring * 1/np.sqrt(d) factor
            numv_ls = [] # number (state) vector
            for j in range(d): # generate qubit
                phase.append(exp((I*2*np.pi*p*j)/d))
                numv = np.zeros(2*d)
                numv[j] = 1 #left qudit: from index 0 -> d-1
                numv[(j+c)%d+d] = 1 # right qudit: from index d -> 2d-1
                numv_ls.append(numv) # add the numv to the overall list
            # we've generated a BS at this point
            bs_dict[str(c)+str(p)]=[numv_ls, phase] # store numv and phase in the dict

# function to make BS more readable
def read_BS(bs):
    print_str = '' # initialize total output string
    for i in range(len(bs[0])):
        numv1 = bs[0][i]
        phase1=bs[1][i]
        i1, = np.where(numv1 == 1)
        print_str+=str(phase1)+'|%i%i> + '%(i1[0], i1[1]-d)
    
    print_str = print_str[:-2] # remove extra + at end
    print(print_str)

## Generate Bell States ##
generate_bs(d)
print('---------')
print('initializing with d =', d)
print('num BS:', len(bs_dict))
for key in bs_dict:
    read_BS(bs_dict[key])
print('---------')

## Helper functions for measurement ##

# initialize symbols
var_dict={}
for i in range(2*d):
    var_dict['v'+str(i)] = Symbol('v'+str(i))

# for measurement
def measure_BS(bs):
    measured_ls = [] # list to hold sympy matrices which we will take the inner product of with other BS
    # go through each joint particle state
    for i, numv in enumerate(bs[0]):
        for j in range(2*d): # check over each index to annihilate
            if numv[j]==1: # found a particle, lower it and append lowered numv to measured ls
                lowered = numv.copy() # need to create copy so we don't annihilate on the original, which we will need as we continue iterating
                lowered[j]-=1
                phase = bs[1][i] # get phase factor
                vj = var_dict['v'+str(j)] # get sympy variable coefficient for this annihilation operator
                result = Matrix(lowered)*phase*vj
                measured_ls.append(result)
            
    return measured_ls

print('---------')
print('measuring:')
bs = bs_dict['01'] # pick sample qudit to measure
read_BS(bs)
print(measure_BS(bs))
print('---------')

# measure all qudits and store
measured_all = []
for key in bs_dict:
    measured_all.append(measure_BS(bs_dict[key]))

## Choose k ##
# k = int(input('Enter a value for k: '))
k = 3

## Find all unique combinations of the d**2 BS choose k ##
# Use Ben's work to find equivalence classes #
# for now, take combinations of k indices from the measured

## Take inner product within all unique k groups ##

## Solve resultant system ##
# need to implement check for time?