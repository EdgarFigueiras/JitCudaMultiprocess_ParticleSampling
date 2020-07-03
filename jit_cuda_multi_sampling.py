#!/usr/bin/env python
"""
    dnolivieri:  particle sampling method
    efigueiras:  jit kernel and paralelization process
"""
from __future__ import print_function, absolute_import

from numba import cuda, njit, prange, jit
import numpy as np

import argparse
import scipy
from matplotlib import cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import glob
import codecs, json
import scipy.io as sio
import sys, os
import psutil
from multiprocessing import Process, Value, Array
import math
import time


#Multiply by a factor all the values in a matrix.
#Used to increase the psi values
@cuda.jit
def cuda_arraymult(arr):
    increase_factor = 100
    idx = cuda.grid(1)

    #Obtain the total size of the 3d matrix
    i = idx // (arr.shape[1] * arr.shape[2])
    j = (idx // arr.shape[2]) % arr.shape[1]
    k = (idx % arr.shape[2])


    if idx < arr.size:
        #arr[i, j, k] *= increase_factor
        cuda.atomic.add(arr, (i, j, k), arr[i, j, k]*increase_factor)
    cuda.syncthreads()

#Multiply the values of 2 matrix of the same shape
#Used to increase the psi values
@cuda.jit
def cuda_2arraymult(arr, arr2):
    idx = cuda.grid(1)

    #Obtain the total size of the 3d matrix
    i = idx // (arr.shape[1] * arr.shape[2])
    j = (idx // arr.shape[2]) % arr.shape[1]
    k = (idx % arr.shape[2])

    if idx < arr.size:
        arr[i, j, k] *= arr2[i, j, k]


@jit('void(float64[:,:,:])')
def jit_mult(arr):
    for i in range(len(arr)):
        for j in range(len(arr[0])):
            for k in range(len(arr[0][0])):
                arr[i][j][k] *= 100

#Monte-Carlo Qpsi calculation using jit operations
@njit
def jit_Qpsi(Nsamples, gpoints, psisqr, xcm):
    Qpsi = np.zeros(shape=(Nsamples,4))
    cnt=0
    while cnt < Nsamples:
        irand = np.random.randint(0,len(gpoints))
        qr = np.random.rand()
        ix = gpoints[irand,0]
        jx = gpoints[irand,1]
        kx = gpoints[irand,2]
        if qr < psisqr[ix,jx,kx]:
            deltax= ((np.random.rand() * 2) - 1) - xcm[0]
            deltay= ((np.random.rand() * 2) - 1) - xcm[1]
            deltaz= ((np.random.rand() * 2) - 1) - xcm[2]
            Qpsi[cnt][0] = ix+deltax
            Qpsi[cnt][1] = jx+deltay
            Qpsi[cnt][2] = kx+deltaz
            Qpsi[cnt][3] = psisqr[ix,jx,kx]
            cnt+=1
    return Qpsi

#Principal class
class pSampleMethod:
    def __init__(self):
        outdata_folder="resdata"
        output_folder="outdir"
        self.outdata_folder = outdata_folder
        if not os.path.exists(outdata_folder):
            os.makedirs(outdata_folder)
        self.output_folder = output_folder
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)


    #Creates the sampling distribution generating random particles
    def get_sampling_dist(process, self, psisqr, Nsamples, threshold, xcm):
        trshold = float(threshold)
        ybar = np.where( psisqr > trshold )
        gpoints = [[c,v,w] for c,v,w in zip(ybar[0],ybar[1], ybar[2])]
        gpoints = np.array(gpoints)

        Qpsi = jit_Qpsi(int(Nsamples), gpoints, psisqr, xcm)
        return Qpsi.tolist()


    #Transforms .NPZ files into .3d files
    def transformNPZMat(self, process, Nsamples, threshold, tmin, tmax):
        fbar=glob.glob("./" + self.outdata_folder + "/*.npz")
        fbar.sort()

        trshold = float(threshold)
        final_arr=[]
        cnt=0
        xcm=np.array([0.,0.,0.])

        for fname in fbar:
            #print('----get file:', fname)
            npzfile = np.load(fname)
            psisqr=None
            if mattype=="npz":
                psisqr = npzfile['psi']
            else:
                psisqr = npzfile['arr_0']

            #jit
            jit_mult(psisqr)

            #cuda.jit
            #threads_per_block = 1024
            #blocks_per_grid = int(np.ceil(np.prod(psisqr.shape) / threads_per_block))
            #cuda_arraymult[blocks_per_grid, threads_per_block](psisqr)

            if (cnt==0):
                ybar = np.where( psisqr > trshold )
                gpoints = [[c,v,w] for c,v,w in zip(ybar[0],ybar[1], ybar[2])]
                gpoints = np.array(gpoints)
                xcm=np.mean(gpoints, axis=0)
            if cnt < int(tmin):
                #print(fname, "...not included")
                cnt+=1
                continue
            elif cnt > int(tmax):
                break

            if((tmax-tmin) > 0):
                print('Process:', process, ' completed: ', int(((cnt-tmin)/(tmax-tmin))*100), '%')
            else:
                print('Process:', process, ' completed: 100%')
            Qpsi=self.get_sampling_dist(process, psisqr, Nsamples, threshold, xcm)
            final_arr.append(Qpsi)
            cnt+=1

        #.3d file generation
        f = open(self.output_folder + '/3dData' + str(process) + '.3d', 'wb+')
        final_arr = np.array(final_arr)
        np.savez(f, final_arr)

        f.close()

    #Transforms .MAT files into .NPZ files
    def convertMatlabFiles(self, process, nstep, tmin, tmax):
        fbar=glob.glob(self.outdata_folder + "/*.mat")
        fbar.sort()
        cnt=0
        nstepsplot = int(nstep)

        for fname in fbar:
            print('----file:', fname)
            z = sio.loadmat(fname)
            psisqr = z['psi2_evolution']
            if cnt < int(tmin):
                print(fname, "...not included")
                cnt+=1
                continue
            elif cnt> int(tmax):
                break
            if np.mod(cnt,nstepsplot)==0:
                npz_name=fname.replace(".mat", ".npz")
                f = open(npz_name, 'wb+')
                np.savez(f, psisqr)
                f.close()

            cnt+=1

            if((tmax-tmin) > 0):
                print('Process:', process, ' completed: ', int(((cnt-tmin)/(tmax-tmin))*100), '%')
            else:
                print('Process:', process, ' completed: 100%')

    #Base multiprocessing function
    def run_multi_process(self, processes, mat_type, Nsamples, threshold, stepsize, tmin, tmax):
        cores = int(processes)
        min_step = int(tmin)
        max_step = int(tmax)
        total_steps = max_step - min_step + 1 #Starts with file 0
        #steps_by_core = int(math.ceil(total_steps / cores))
        steps_by_core = (total_steps // cores)
        print("Cores: " + str(cores))
        print("Total steps: " + str(total_steps))
        print("Steps by core: " + str(steps_by_core))

        start = time.time()

        #If mat type selected converts this files to NPZ
        if (mat_type=="mat"):
            print("Converting MAT files to NPZ")
            first_step = min_step
            last_step = first_step + steps_by_core
            processes=[]
            for process in range(cores):
                print('Process:', process, ' --- First step: ', first_step, '--Last step: ', last_step)
                p = Process(target=self.convertMatlabFiles, args=(process, stepsize, first_step, last_step))
                p.start()
                processes.append(p)
                first_step = last_step + 1
                last_step = last_step + steps_by_core
                if (last_step > max_step):
                    last_step = max_step

            #Joins all processes to wait for the last process to finish
            for p in processes:
                p.join()

        if (mat_type=="npz"):
            print("Loading NPZ files")

        #Generates the particles using preexisting NPZ files or converted .MAT files
        print("Generating particles from NPZ files")
        first_step = min_step
        last_step = first_step + steps_by_core
        processes=[]
        for process in range(cores):
            print('Process:', process, ' --- First step: ', first_step, '--Last step: ', last_step)
            p = Process(target=self.transformNPZMat, args=(process, Nsamples, threshold, first_step, last_step))
            p.start()
            processes.append(p)
            first_step = last_step + 1
            last_step = last_step + steps_by_core
            if (last_step > max_step):
                last_step = max_step

        #Joins all processes to wait for the last process to finish
        for p in processes:
            p.join()

        end = time.time()
        print("Elapsed time: ", end - start)

        #Call the .json file generator, will convert .3s to a QMWebJS compatible .json file
        self.json_generator(cores, total_steps, Nsamples)


    #Generates .json file compatible with QMWebJS
    def json_generator(self, num_files, total_steps, particles):
        complete_array = []

        for n_file in range(num_files):
            #.json file generation
            file_with_binary_data = open(self.output_folder + '/3dData' + str(n_file) + '.3d', 'rb+')
            #Gets the binary data as an array
            array_with_all_data = np.load(file_with_binary_data, allow_pickle=True)
            #Matrix with the data of the 3D grid
            if(n_file == 0):
                complete_array = array_with_all_data['arr_0']
            else:
                complete_array = np.concatenate((complete_array, array_with_all_data['arr_0']), axis=0)

            #Remove temporal 3d files
            os.remove(self.output_folder + '/3dData' + str(n_file) + '.3d')

        #.3d file generation
        f = open(self.output_folder + '/3dData.3d', 'wb+')
        #complete_array = np.array(complete_array)
        np.savez(f, complete_array)

        f.close()

        #Prints the final shape of the .json files
        print('Array shape: ', complete_array.shape)
        print("Generating the .json file...")
        json_list = complete_array.round(1).tolist()
        out_file = (self.output_folder + '/3dData.json')

        #Creates the .json file
        json.dump(json_list,
                  codecs.open(out_file, 'w',encoding='utf-8'),
                  separators=(',', ':'),
                  sort_keys=True, indent=4)

        print("Finished!")


def check_arg(args=None):
    parser = argparse.ArgumentParser(description='The Particle Sampling')
    parser.add_argument('-m', '--mattype',
                        help='The simulator output type: either .mat (mat) or .npz (npz)',
                        required=True,
                        default='npz')

    parser.add_argument('-t', '--threshold',
                        help='The amplitude threshold value for selected particles',
                        required=False,
                        default=0.05)

    parser.add_argument('-n', '--nparticles',
                        help='The number of sampled particles',
                        required=False,
                        default=5000)

    parser.add_argument('-t0', '--tmin',
                        help='The minimum time point to sample from',
                        required=False,
                        default=0)

    parser.add_argument('-tf', '--tmax',
                        help='The maximum time point to samples from',
                        required=False,
                        default=100)

    parser.add_argument('-s', '--stepsize',
                        help='The timestep for sampling',
                        required=False,
                        default=1)

    parser.add_argument('-p', '--processes',
                        help='The number of processes that will be used for paralelization (uses the number of physicall cores by default)',
                        required=False,
                        default=psutil.cpu_count())

    results = parser.parse_args(args)
    return (results.mattype, results.threshold, results.nparticles, results.tmin, results.tmax, results.stepsize, results.processes)



#-----------------------------------
if __name__ == '__main__':
    mattype, threshold, nparticles, tmin, tmax, stepsize, processes  =  check_arg(sys.argv[1:])

    print('mattype =',mattype)
    print('threshold =',threshold)
    print('nparticles =',nparticles)
    print('tmin =',tmin)
    print('tmax =',tmax)
    print('stepsize =',stepsize)
    print('processes =',processes)

    P = pSampleMethod()
    P.run_multi_process(processes, mattype, nparticles, threshold, stepsize, tmin, tmax)
