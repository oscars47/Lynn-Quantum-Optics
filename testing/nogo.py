# file to compute the maximum k subset of d^2 bell states
# @oscars47

# using sympy to handle symbolic manipulation
from sympy import *
# from sympy.solvers import solve, solveset
from sympy.physics.quantum.dagger import Dagger
# from sympy.printing.mathematica import MCodePrinter
import numpy as np
from itertools import combinations
import pandas as pd
import time
import signal # limit time for solvers

## Initialize d value ##

# d = int(input('Enter a value for d: '))
d=4
## Choose k ##
# k = int(input('Enter a value for k: '))
k = 3


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
print('k = ', k)
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
# alphabet=tuple(alphabet)
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

## Find all unique combinations of the d**2 BS choose k ##
# Use Ben's work to find equivalence classes #
# for now, take combinations of k indices from the measured
k_groups = list(combinations(np.arange(0, len(measured_all)), k))

## Take inner product within all unique k groups ##
# takes in k_group and exisiting eqn for the entire k group

# define dataframe to log info about what possible pairs in k group generate solns
# k is the k number, L is left qubit, R is right qubit (in the inner product), and num soln is the number of sets in the sol
# global results
results= pd.DataFrame({'d': [], 'k_group':[], 'num_soln':[]})
def solve_k_eqn(k_group):
    print('-*----*---*-')
    print('using k group:', k_group)
    pairs_comb = list(combinations(k_group, 2))
    # we want to find 2d unique solutions to cover all the detector modes
    eqn_re = [] # list to hold equations which we generate from the inner products
    eqn_im = []
    eqn_total = []
    for pair in pairs_comb:
        inner_prod = Dagger(measured_all[pair[0]])*measured_all[pair[1]]
        # split result into real and imaginary components
        # print(inner_prod)
        re_part= re(inner_prod)[0]
        im_part = im(inner_prod)[0]
        # re_part= nsimplify(re(inner_prod)[0]) # convert to rationals
        # im_part = nsimplify(im(inner_prod)[0])
        if re_part != 0: # want only nonzero terms
            eqn_re.append(re_part)
            eqn_total.append(re_part)
        if im_part != 0:
            eqn_im.append(im_part)
            eqn_total.append(im_part)

    eqn_re.append(re(norm_cond))
    eqn_total.append(re(norm_cond))
    
    print('eqn re:', eqn_re)
    print('eqn im:', eqn_im) 
    print('eqn total:', eqn_total)

    
    ## for timing out ##
    class TimeoutException(Exception):   # Custom exception class
        pass


    def break_after(seconds=2):
        def timeout_handler(signum, frame):   # Custom signal handler
            raise TimeoutException
        def function(function):
            def wrapper(*args, **kwargs):
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(seconds)
                try:
                    res = function(*args, **kwargs)
                    signal.alarm(0)      # Clear alarm
                    return res
                except TimeoutException:
                    print('Solve time exceeded :(')
                return
            return wrapper
        return function

    # soln = solve(eqn_total, alphabet, force=True, dict=True)
    global findsoln
    global results_new
    results_new=0
    findsoln=False
    
    @break_after(5) # 30 second limit
    def try_soln():  
        global findsoln
        global results_new
        # try:
        #     soln = nonlinsolve(eqn_total, alphabet)
        #     print('soln:', soln)
        #     print(len(soln))
        #     if len(soln)>0:
        #         results_new = {'d': d, 'k_group':k_group, 'num_soln':len(soln)}
        #         findsoln = True # found a solution!
        # except:
        #     print('there was a problem with the solve')
            # ValueError: Absolute values cannot be inverted in the complex domain.

        # soln = nonlinsolve(eqn_total, alphabet)
        soln = solve(eqn_total, alphabet, manual=True, set=True)
        # soln = solve(eqn_total, alphabet, check=False, rational=True, manual=True, set=True, implicit=True)

        # soln = nsolve(eqn_total, alphabet, np.zeros(len(alphabet))) # need as many equations as variables for nsolve
        print('soln:', soln)
        if len(soln) > 0 and len(soln[1])>2:
            # print(len(soln[1]))
            results_new = {'d': d, 'k_group':k_group, 'num_soln':len(soln[1])}
            findsoln = True # found a solution!
    

    
    try_soln()
    if findsoln== True: # did find >= 1 solution:
        print('found a solution for k=%i, exiting search.'%k)
    return findsoln, results_new
    
    

    ## ISSUE: not finding symbolic solution ##
    # https://stackoverflow.com/questions/61548790/sympy-solve-returns-an-empty-set

# log computation time
start_time = time.time()
for k_group in k_groups:
    findsoln, results_new = solve_k_eqn(k_group)
    if not(isinstance(results_new, int)):
        results = results.append(results_new, ignore_index=True)
    if findsoln==True:
        break
end_time = time.time()
print(results)
results.to_csv('d=%i,k=%i_results.csv'%(d, k))
print('computation took %s seconds'%(end_time-start_time))