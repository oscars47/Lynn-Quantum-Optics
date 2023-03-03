# file to compute the maximum k subset of d^2 bell states
# @oscars47

# using sympy to handle symbolic manipulation
from sympy.solvers import solve
from sympy import *
from sympy.physics.quantum.dagger import Dagger
import numpy as np
from itertools import combinations

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
print('*---------*')
print('initializing with d =', d)
print('num BS:', len(bs_dict))
for key in bs_dict:
    read_BS(bs_dict[key])
print('*---------*')

## Helper functions for measurement ##

# initialize symbols
var_dict={}
alphabet = [] # alphabet of synbols
for i in range(2*d):
    a_i, b_i = Symbol('a'+str(i), real=True), Symbol('b'+str(i), real=True)
    # alphabet['a'+str(i)] = a_i
    # alphabet['b'+str(i)] = b_i
    alphabet.append(a_i)
    alphabet.append(b_i)
    var_dict['v'+str(i)] = a_i+b_i*I
    # var_dict['v'+str(i)] = Symbol('v'+str(i))

# get normalization term: sum of modulus squared for each variable
norm_ls = [Dagger(var_dict['v'+str(i)])*var_dict['v'+str(i)] for i in range(len(var_dict))]
norm_ls.append(-1) # append a -1, i.e. sum_i v_iv_i^\dagger -1 = 0
norm_cond = sum(norm_ls)

# for measurement
def measure_BS(bs):
    # measured_ls = [] # list to hold sympy matrices which we will take the inner product of with other BS
    measured = Matrix(np.zeros(2*d))
    # go through each joint particle state
    for i, numv in enumerate(bs[0]):
        for j in range(2*d): # check over each index to annihilate
            if numv[j]==1: # found a particle, lower it and append lowered numv to measured ls
                lowered = numv.copy() # need to create copy so we don't annihilate on the original, which we will need as we continue iterating
                lowered[j]-=1
                phase = bs[1][i] # get phase factor
                vj = var_dict['v'+str(j)] # get sympy variable coefficient for this annihilation operator
                # break up phase into re and im; need to deal with really small and should be 0 terms
                phase_re = re(phase)
                phase_im = im(phase)
                if phase_re < 10**(-4):
                    phase_re=0
                if phase_im < 10**(-4):
                    phase_im=0
                phase = phase_re + phase_im*I
                coef= phase*vj
                # print('coef', coef)
                # if N(coef) < 10**(-4):
                #     coef=0
                result = Matrix(lowered)*coef
                measured+=result
            
    return measured

# print('-----*----')
# print('measuring:')
# bs = bs_dict['01'] # pick sample qudit to measure
# read_BS(bs)
# print(measure_BS(bs))
# print('-----*----')

# measure all qudits and store
measured_all = []
for key in bs_dict:
    measured_all.append(measure_BS(bs_dict[key]))

## Choose k ##
# k = int(input('Enter a value for k: '))
k = 3
print('k = ', k)

## Find all unique combinations of the d**2 BS choose k ##
# Use Ben's work to find equivalence classes #
# for now, take combinations of k indices from the measured
k_groups = list(combinations(np.arange(0, len(measured_all)), k))

## Take inner product within all unique k groups ##
# takes in k_group and exisiting eqn for the entire k group
def solve_k_eqn(k_group):
    print('-*----*---*-')
    print('using k group:', k_group)
    pairs_comb = list(combinations(k_group, 2))
    # we want to find 2d unique solutions to cover all the detector modes
    eqn_re = [] # list to hold equations which we generate from the inner products
    eqn_im = []
    for pair in pairs_comb:
        inner_prod = Dagger(measured_all[pair[0]])*measured_all[pair[1]]
        # split result into real and imaginary components
        # print(inner_prod)
        eqn_re.append(re(inner_prod))
        eqn_im.append(im(inner_prod))

    eqn_re.append(re(norm_cond))
    # eqn_im.append(im(norm_cond)) # no imaginary component of length
    # eqn.append(im(norm_cond))
    # eqs,S('(r5,r6,r9)'),manual=1,check=0,simplify=0
    # print('len re:', len(eqn_re), 'len im:', len(eqn_im))
    print('eqn re:', eqn_re)
    print('eqn im:', eqn_im) 
    # soln = nsolve(eqn, [alphabet['a0'], alphabet['a1'], alphabet['a2'], alphabet['a3'], alphabet['b0'], alphabet['b1'], alphabet['b2'], alphabet['b3']], 0, dict=True)
    # print(tuple(alphabet[:-1]))
    # if len(eqn)<len(alphabet):
    # #     soln = nsolve(tuple(eqn), tuple(alphabet[:-(len(alphabet) - len(eqn))]), tuple(np.zeros(len(alphabet)-1)), dict=True)
    #     soln =solve(eqn, tuple(alphabet[:len(eqn)-len(alphabet)]), dict=True)
    #     # print('here')
    # else:
    # #      soln = nsolve(tuple(eqn), tuple(alphabet), tuple(np.zeros(len(alphabet)-1)), dict=True)
    #     soln =solve(eqn, tuple(alphabet), dict=True)
    # sometimes im part or real part is 0
    # if eqn_re!=[Matrix([[0]])]:
    #     soln_re =solve(eqn_re,dict=True)
    # else:
    #     soln_re=[]
    # if eqn_im!=[Matrix([[0]])]:
    #     soln_im =solve(eqn_im,dict=True)
    # else:
    #     soln_im=[]

    soln_re =solve(eqn_re,dict=True)
    soln_im =solve(eqn_im,dict=True) 

    if len(soln_re)>=1 and len(soln_im)>=1:
        soln_re =solve(eqn_re,dict=True)[0]
        soln_im =solve(eqn_im,dict=True)[0]
        # combine eqns to solve for intersection
        eqn_intersect = []
        soln_re_dep = soln_re.keys() # dependent vars for real
        for re_dep in soln_re_dep:
            eqn_intersect.append(re_dep - soln_re[re_dep])
        soln_im_dep = soln_im.keys() # dependent vars for real
        for im_dep in soln_im_dep:
            eqn_intersect.append(im_dep - soln_im[im_dep])

        soln_intersect = solve(eqn_intersect)
        print('soln of intersect:', soln_intersect)
        print('len of soln intersect:', len(soln_intersect))
    
    elif len(soln_re)>=1 and len(soln_im)<1:
        soln_re =solve(eqn_re,dict=True)[0]
        print('soln re (only):', soln_re)
        print('len of re only:', len(soln_re))

    elif len(soln_im)>=1 and len(soln_re)<1:
        soln_im =solve(eqn_im,dict=True)[0]
        print('soln im (only):', soln_im)
        print('len of im only:', len(soln_im))
    

    
    
    

    ## ISSUE: not finding symbolic solution ##
    # https://stackoverflow.com/questions/61548790/sympy-solve-returns-an-empty-set
    
for k_group in k_groups:
    solve_k_eqn(k_group)
## Solve resultant system ##
# need to implement check for time?