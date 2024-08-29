#!/bin/python
# import matplotlib.pyplot as plt
import numpy as np
import os, sys, time
import numpy.random as rn
from edbn.Methods.SDL.sdl import transform_data

icon=['bx','ro','g*','c<']


class Sampler():
    """
    Sample data from a quadrant and label according to sample being within or outside unit circle.
    Normalize samples so labels are equally distributed. 
    """
    def __init__(self,
                data=0,
                alpha=1.0, 
                verbose=False,
                ntasks=2,
                dim=4,
                discriminator_offset=0.01, 
                distribution_offset=0.01, 
                uniform_width=1.25, 
                nsamples=50000,
                ntrain=4000,
                ntest=500):
        # seed
        rn.seed(512)
        stime=time.time()
        tasks={}
        
        ntrain=len(data.train.get_data())
        ntest=len(data.test_orig.get_data())
        # sample points for four tasks related to four quadrants
        # x, y, vals= transform_data(data.train, [a for a in data.train.attributes() if a != data.train.time and a != data.train.trace])
        x, y, vals= transform_data(data.logfile, [a for a in data.logfile.attributes() if a != data.logfile.time and a != data.logfile.trace])
        samples=[]
        #for i in range(len(data.test_orig.get_data())):
        for i in range(len(data.logfile.get_data())):
            sample = []
            for j in range(len(x)):
                # sample.append(np.array([x[j][i]]))
                sample.append(x[j][i])
            samples.append(np.array(sample))
        # for q in range(ntasks):
        #     if q == 0: # 1th task
        #         tasks[q]=np.asarray(samples[:150])
        #     elif q == 1: # 2nd task
        #         tasks[q]=np.asarray(samples[150:300])

        inputs={}
        labels={}

        # Step 3: Sample correct distribution given certain alpha
        for q in range(ntasks):
            inputs[q]=samples[:ntrain]
            labels[q]=np.argmax(y,axis=1)[:ntrain]
        # draw some extra test data to evaluate
        # label points according to distance to center
        test_inputs={}
        test_labels={}
        for q in range(ntasks):
            test_inputs[q]=samples[-ntest:]
            test_labels[q]=np.argmax(y,axis=1)[-len(data.test_orig.get_data()):]

        # self.tasks=tasks
        self.inputs=inputs
        self.labels=labels
        self.test_inputs=test_inputs
        self.test_labels=test_labels