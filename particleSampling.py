#!/usr/bin/env python
"""
    dnolivieri:  particle sampling method
    converts the .npz/.mat file directly to the
"""
import argparse
import numpy as np
import scipy
from matplotlib import cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import glob
import codecs, json
import scipy.io as sio
import sys, os
import time

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



    def get_sampling_dist(self, psisqr, Nsamples, threshold, xcm):
        trshold = float(threshold)
        ybar = np.where( psisqr > trshold )
        gpoints = [[c,v,w] for c,v,w in zip(ybar[0],ybar[1], ybar[2])]
        gpoints = np.array(gpoints)
        Qpsi=[]
        cnt=0
        while cnt < int(Nsamples):
            irand= np.random.randint(0,len(gpoints))
            qr=np.random.rand()
            ix = gpoints[irand,0]
            jx = gpoints[irand,1]
            kx = gpoints[irand,2]
            if qr < psisqr[ix,jx,kx]:
                deltax=np.random.choice([-1,1])*np.random.rand()/2. - xcm[0]
                deltay=np.random.choice([-1,1])*np.random.rand()/2. - xcm[1]
                deltaz=np.random.choice([-1,1])*np.random.rand()/2. - xcm[2]
                Qpsi.append([ix+deltax, jx+deltay, kx+deltaz, psisqr[ix,jx,kx] ])
                cnt+=1
        return Qpsi


    def transformNPZMat(self, Nsamples, threshold, tmin, tmax):
        fbar=glob.glob("./" + self.outdata_folder + "/*.npz")
        fbar.sort()

        trshold = float(threshold)
        final_arr=[]
        cnt=0
        xcm=np.array([0.,0.,0.])
        for fname in fbar:
            print('----get file:', fname)
            npzfile = np.load(fname)
            psisqr=None
            if mattype=="npz":
                psisqr = npzfile['psi']
            else:
                psisqr = npzfile['arr_0']
            psisqr *=100.
            if (cnt==0):
                ybar = np.where( psisqr > trshold )
                gpoints = [[c,v,w] for c,v,w in zip(ybar[0],ybar[1], ybar[2])]
                gpoints = np.array(gpoints)
                xcm=np.mean(gpoints, axis=0)
            if cnt < int(tmin):
                print(fname, "...not included")
                cnt+=1
                continue
            elif cnt > int(tmax):
                break

            Qpsi=self.get_sampling_dist(psisqr, Nsamples, threshold, xcm)
            final_arr.append(Qpsi)
            cnt+=1

        #.3d file generation
        f = open(self.output_folder + '/3dData.3d', 'wb+')
        final_arr = np.array(final_arr)
        np.savez(f, final_arr)

        #.json file generation
        json_list = final_arr.round(1).tolist()
        out_file = (self.output_folder + '/3dData.json')

        json.dump(json_list,
                  codecs.open(out_file, 'w',encoding='utf-8'),
                  separators=(',', ':'),
                  sort_keys=True, indent=4)

        f.close()


    def convertMatlabFiles(self, nstep, tmin, tmax):
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


    def run(self, mat_type, Nsamples, threshold, stepsize, tmin, tmax):

        start = time.time()

        if (mat_type=="npz"):
            print("Generating particles from NPZ files")
            self.transformNPZMat(Nsamples, threshold, tmin, tmax)

        if (mat_type=="mat"):
            print("Generating particles from MAT files")
            self.convertMatlabFiles( stepsize, tmin, tmax)
            self.transformNPZMat(Nsamples, threshold, tmin, tmax)

        end = time.time()
        print("Elapsed time: ", end - start)

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
                        default=500)

    parser.add_argument('-s', '--stepsize',
                        help='The timestep for sampling',
                        required=False,
                        default=1)

    results = parser.parse_args(args)
    return (results.mattype, results.threshold, results.nparticles, results.tmin, results.tmax, results.stepsize)



#-----------------------------------
if __name__ == '__main__':
    mattype, threshold, nparticles, tmin, tmax, stepsize  =  check_arg(sys.argv[1:])

    print('mattype =',mattype)
    print('threshold =',threshold)
    print('nparticles =',nparticles)
    print('tmin =',tmin)
    print('tmax =',tmax)
    print('stepsize =',stepsize)


    P = pSampleMethod()
    P.run(mattype, nparticles, threshold, stepsize, tmin, tmax)
