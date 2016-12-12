# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from __future__ import print_function
import pyopencl as cl
import numpy as np
import matplotlib
matplotlib.use('Qt4Agg')

con = cl.create_some_context()
print('Context %s' % con)
print('Device %s' % con.devices)

nCandidates = 64
nCandDim = 2
onlooker_acceptable_change = 0.01
scout_iterations = 2
max_iteration = 100

employee_g, employee_l = (nCandidates * nCandDim, nCandidates, 1), (nCandDim, nCandidates, 1)
nWorkItems = employee_l[0]*employee_l[1]*employee_l[2]

hashDefs = {'CAND_DIM': nCandDim, 'MAX_WORKITEM_INDEX': nWorkItems, 
            'LOG2_MAX_WORKITEM_INDEX': int(np.ceil(np.log2(nWorkItems))), 'FOODS': nCandidates}

code_full = '\n'.join(['#define %s %d' % (s,n) for s,n in hashDefs.items()]) + '\n' + \
            open('kernel.cl','r').read()

build = cl.Program(con,code_full).build()
employee2 = build.employee2
scout_onlooker = build.scout_onlooker_bee
combine_should_employ = build.combine_should_employ

candidates = np.random.rand(nCandidates*nCandDim).astype(np.float32)*np.pi
assert(len(candidates) == (nCandidates * nCandDim))

rands_e = np.random.rand(np.prod(employee_g)*max_iteration).astype(np.float32)
out = np.ones_like(candidates, dtype = np.float32) * -1
haveChanged = np.zeros(nCandidates, dtype = np.int32)

cl_candidates = cl.Buffer(con, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf= candidates)
cl_out = cl.Buffer(con, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf= out)
cl_objectives = cl.Buffer(con, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf= out)
cl_shouldemploy = cl.Buffer(con, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf= np.ones(nCandidates, dtype = np.uint8))
cl_shouldemployonlooker = cl.Buffer(con, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf= np.ones(nCandidates, dtype = np.uint8))
cl_shouldemployscout = cl.Buffer(con, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf= np.ones(nCandidates, dtype = np.uint8))
cl_haveChanged = cl.Buffer(con, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf= haveChanged)
cl_rands_e = cl.Buffer(con, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf= rands_e)

scout_g, scout_l = (nCandidates,2,1), (nCandidates,1,1)
cl_rands_s = cl.Buffer(con, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf= np.random.rand(nCandidates*max_iteration).astype(np.float32))
queue = cl.CommandQueue(con)

employee2.set_scalar_arg_dtypes([None,None,None,np.int32,np.float32,None,None,None])
scout_onlooker.set_scalar_arg_dtypes([None,None,None,np.int32,np.int32,None,None])

combine_g, combine_l = (nCandidates,1,1), None

res_candidates = np.empty((max_iteration,nCandidates,nCandDim), dtype = np.float32)

for iteration in range(max_iteration):
    employee2(queue, employee_g, employee_l, 
              cl_candidates, cl_shouldemploy, cl_rands_e, np.int32(iteration), np.float32(onlooker_acceptable_change),
              cl_haveChanged, cl_objectives, cl_candidates)
    
    scout_onlooker(queue, scout_g, scout_l,
                   cl_objectives,cl_rands_s, cl_haveChanged, np.int32(iteration), np.int32(scout_iterations), 
                   cl_shouldemployonlooker, cl_shouldemployscout)
    
    combine_should_employ(queue, combine_g, combine_l,
                          cl_shouldemployonlooker, cl_shouldemployscout, cl_shouldemploy)
    
    wrkr = cl.enqueue_copy(queue, res_candidates[iteration], cl_candidates)
    
import matplotlib.pyplot as plt
fig,ax = plt.subplots(1,1)

wrkr.wait()

for i,cand in enumerate(res_candidates):
    xx = [i] * cand.shape[0]
    ax.plot(xx, cand[:,0], '.r')
    ax.plot(xx, cand[:,1], '.b')
    
plt.show()
