#define CAND_DIM_AT(candindex,dimindex) (candindex*CAND_DIM+dimindex)

__kernel void employee2 (
    __global float *candidates,
    __global char *shouldEmploy,
    __global float *rands,
    int itr,
    float acceptableObjChange,
    __global int *haveChanged,
    __global float *objectives,
    __global float *out) {
 
    int thisCand = get_group_id(0);

    int dim = get_local_id(0);
    int fsource = get_local_id(1);
    
    __local bool should_employ;
    should_employ = shouldEmploy[thisCand];
    float min_food, max_food, randVal;

    float mypoint2[CAND_DIM], myscalar, obj;
    for (int i=0; i<CAND_DIM; i++) mypoint2[i] = candidates[CAND_DIM_AT(thisCand,i)];

    __local float objValues[MAX_WORKITEM_INDEX];
    int flatindex = get_local_id(0) + get_local_size(0)*(get_local_id(1) + get_local_id(2)*get_local_size(1));
    unsigned int globalindex = get_global_id(0) + get_global_size(0)*(get_global_id(1) + get_global_id(2)*get_global_size(1));
    unsigned int globalsize = get_global_size(0) * get_global_size(1) * get_global_size(2);

    barrier(CLK_GLOBAL_MEM_FENCE);

    __local int minI;
    float min;
    int i;
    
    if (should_employ && dim < CAND_DIM && flatindex < MAX_WORKITEM_INDEX) {
        /* valid work item */

        myscalar = candidates[CAND_DIM_AT(thisCand,dim)];
        // min_food = minMaxBounds[2*dim];        max_food = minMaxBounds[2*dim+1];

        /* Ensure that the first local WI is same as original by multiplying with flatindex */
        randVal = rands[globalsize*itr + globalindex];
        // randVal = flatindex * ((seed * 0x5DEECE66DL + 0xBL) & ((1L << 48) - 1));

        mypoint2[dim] -= randVal*(candidates[CAND_DIM_AT(fsource,dim)] - myscalar);
        obj = 1 - (mypoint2[0]*mypoint2[0] + mypoint2[1]*mypoint2[1]);         /* objective function */
        obj = sin(mypoint2[0]) + cos(mypoint2[1]);
        //obj = 1-obj+obj*obj-obj*obj*obj+obj*obj*obj*obj;

        objValues[flatindex] = obj;
        /*    
        barrier(CLK_LOCAL_MEM_FENCE);
        min = objValues[0];
        minI = 0;        
        if (flatindex == 0) {          
            for (i=0; i < MAX_WORKITEM_INDEX; i++) {
                if (objValues[i] < min) {
                    minI = i; min = objValues[i];
                }
            }        
        }*/
        
        // minimum calculate function
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int op1,offset, minItr=0; minItr <= LOG2_MAX_WORKITEM_INDEX; minItr++) {
            offset = 1 << minItr;
            op1 = 2*offset*flatindex;
            if (op1+offset < MAX_WORKITEM_INDEX) {
#define MIN(x,y) x < y ? x : y
                objValues[op1] = MIN(objValues[op1],objValues[op1+offset]);
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        
        barrier(CLK_LOCAL_MEM_FENCE);

        if (obj == objValues[0]) {
            for (i=0; i<CAND_DIM; i++) out[CAND_DIM_AT(thisCand,i)] = mypoint2[i];    
            objectives[thisCand] = obj;
        }
        
        /* For the onlooker bees */
        if (flatindex == 0) {
            // one of the objectives will be same as previous (food == candidate)
            if (fabs(obj - objValues[0]) > acceptableObjChange)
                haveChanged[thisCand] = itr;
        }
    }
}

/*
 G = (4,2,1);    L = (2,2,1)

WG_ID <- (0,0,0);  (1,0,0) (2,0,0)

 L_ID <- (0,0,0);  (0,1,0), .... (1,1,0)
 
 thisCand = 0
 objValues[4]
 
==--- 
 dim = 0
 fsource = 0

 mypoint2[0] = candidates[0]
 mypoint2[1] = candidates[1]
 
==
 flatindex = 0
 
 myscalar = candidates[0]
 randVal = 0
 
 mypoint2[0] = candidates[0]
 
 objValues[0] = 1 - |mypoint2|^2
 
==
 fsource = 1
 flatindex = 1
 globalindex = 1
 
 randVal = rands[1]
 mypoint2[0] += randVal*(candidates[0] - candidates[2])
 objValues[1] = 1 - |mypoint2|^2
 
==---
 dim = 1
 fsource = 0
 
 mypoint2[0] = candidates[0]
 mypoint2[1] = candidates[1]
 
 flatindex = 2
 globalindex = 2
 
 randVal = rands[2]*2
 
 mypoint2[1] += randVal*(candidates[1] - candidates[1])
 objValues[2] = 1 - |mypoint2|^2
 
==
 fsource = 1
 mypoint2[0] = candidates[0]
 mypoint2[1] = candidates[1]
 
 flatindex = 3
 globalindex = 3
 randVal = rands[3]*3
 
 mypoint3[1] += randVal*(candidates[1] - candidates[3])
 objValues[3] = 1 - |mypoint2|^2
 
=====
 
*/
    

__kernel void employ_bee (
    __global float *candidates, 
    __global float *objectives,
    __global const float *rands,
    int randGlobalOffset,
    __global const float *minMaxBounds, 
    int itr, 
    __global int *objectiveEvaluations,
    __global const char *shouldEmploy,
    __global int *haveChanged, 
    float acceptableObjChange,
    __global float *rough,
    __global float *rough2) 
 {
    const int dim = get_local_id(0);
    const int fsource = get_local_id(1);
//  const int rpoint = get_local_id(2);
    const int thisCand = get_group_id(0);
    __local bool should_employ;
    should_employ = shouldEmploy[thisCand];
    float min_food, max_food, randVal;

    float mypoint2[CAND_DIM], myscalar, obj;
    for (int i=0; i<CAND_DIM; i++) mypoint2[i] = candidates[thisCand+i];

    __local float objValues[MAX_WORKITEM_INDEX];
    int flatindex = 0;

//  __local int itr;
//  long int seed = get_global_id(0) * itr;
    flatindex = get_local_id(0) + get_local_size(0)*(get_local_id(1) + get_local_id(2)*get_local_size(1));

    randGlobalOffset = MAX_WORKITEM_INDEX*CAND_DIM*itr;

//  if (flatindex == 0) itr = *iteration;

//  if (flatindex == thisCand && flatindex == 0) printf("ITR%d/n", itr);
    
    barrier(CLK_GLOBAL_MEM_FENCE);

    if (should_employ && dim < CAND_DIM && flatindex < MAX_WORKITEM_INDEX) {
        /* valid work item */

        myscalar = candidates[thisCand*CAND_DIM+dim];
        min_food = minMaxBounds[2*dim];
        max_food = minMaxBounds[2*dim+1];

        /* Ensure that the first local WI is same as original by multiplying with flatindex */
        randVal = rands[get_global_id(0) + randGlobalOffset] * flatindex;
        //randVal = flatindex * ((seed * 0x5DEECE66DL + 0xBL) & ((1L << 48) - 1));

        obj = 10000.0;

        mypoint2[dim] = randVal*(myscalar - candidates[fsource+dim]);
        obj = 1 - (mypoint2[0]*mypoint2[0] + mypoint2[1]*mypoint2[1]);         /* objective function */
        //obj = 1-obj+obj*obj-obj*obj*obj+obj*obj*obj*obj;

        objValues[flatindex] = obj;

        // minimum calculate function
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int op1,offset, minItr=0; minItr <= LOG2_MAX_WORKITEM_INDEX; minItr++) {
            offset = 1 << minItr;
            op1 = 2*offset*flatindex;
            if (op1+offset < MAX_WORKITEM_INDEX) {
                objValues[op1] = min(objValues[op1],objValues[op1+offset]);
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        
        if (thisCand == 0) rough2[flatindex] = itr;
        
        if (objValues[0] == obj) {
            /* i am minimum*/
            for (int i=0; i<CAND_DIM; i++) candidates[thisCand+i] = mypoint2[i];
            objectives[thisCand] = itr;
        }

        /* For the onlooker bees */
        if (flatindex == 0) {
            if (fabs(obj - objValues[0]) > acceptableObjChange)
                haveChanged[thisCand] = 0; //itr
        }
    }
}


__kernel void next_iterate(
    __global int* iteration) 
{
    *iteration = *iteration + 1;
}

__kernel void scout_onlooker_bee(
    __global float *objectives,
    __global float *rands,
    __global int *haveChanged,
    int iteration,
    int scout_iterations, 
    __global char *shouldEmployOnlooker,
    __global char *shouldEmployScout) 
{
    const int thisCand = get_local_id(0);
    float rnd = rands[FOODS*iteration + thisCand];

    __local float myObjectives[FOODS];
    __local float objSum;
    int i;

    float obj;

    if (get_group_id(1) == 0) {
        /* onlooker bee */
        myObjectives[thisCand] = objectives[thisCand];
        obj = myObjectives[thisCand];
  
        barrier(CLK_GLOBAL_MEM_FENCE);
  
        if (thisCand == 0) {
            objSum = 0;
            for (i = 0; i < FOODS; i++)
                objSum += myObjectives[i];
        }
  
        barrier(CLK_LOCAL_MEM_FENCE);
        float nectar  = obj/objSum;
        shouldEmployOnlooker[thisCand] = rnd < nectar;   
    } else {
        /* scout bee */
        shouldEmployScout[thisCand] = (iteration - haveChanged[thisCand]) > scout_iterations;
    }
}

__kernel void combine_should_employ(
    __global char *shouldEmployOnlooker,
    __global char *shouldEmployScout,
    __global char *shouldEmploy) 
{
    unsigned int globalindex = get_global_id(0) + get_global_size(0)*(get_global_id(1) + get_global_id(2)*get_global_size(1));
    
    shouldEmploy[globalindex] = shouldEmployScout[globalindex] || shouldEmployOnlooker[globalindex];
}