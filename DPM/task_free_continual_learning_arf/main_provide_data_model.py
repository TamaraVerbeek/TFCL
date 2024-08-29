#!/usr/bin/python
import sys
sys.path.append('/Users/meruozhu/Downloads/MP_data/MP_codes/MP')
from DPM.task_free_continual_learning_arf.method_provide_model_test import Task_free_continual_learning_provide_model
from DPM.task_free_continual_learning_arf.sampler_provide_model import Sampler
from edbn.Methods.SDL.sdl import transform_data

import matplotlib.pyplot as plt
import numpy as np
import os, sys, time
import matplotlib.patches as mpatches

import time
import pandas as pd
import numpy as np
from scipy.stats import percentileofscore
from Data.data import Data
from edbn.Utils.LogFile import LogFile
import edbn.Predictions.setting as setting
from edbn import Methods
from collections import deque
from itertools import islice
from collections import OrderedDict
from PrefixTreeCDD.PrefixTreeClass import PrefixTree
import PrefixTreeCDD.settings as settings
from PrefixTreeCDD.CDD import Window
from skmultiflow.drift_detection import ADWIN, PageHinkley
import math
from numpy import log as ln
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import utils as ku
import numpy.random as rn
from skmultiflow.meta import AdaptiveRandomForestClassifier

# from method import Task_free_continual_learning
# from sampler import Sampler

ntasks=2

def experiment(data, learning_object, tags=['Online Continual','Online Continual No Hardbuffer','Online','Online No Hardbuffer']):
    training_losses={}
    test_losses={}
    loss_window_means={}
    update_tags={}
    loss_window_variances={}
    settings={'Online Continual':(True, True),
             'Online Continual No Hardbuffer':(False, True),
             'Online':(True, False),
             'Online No Hardbuffer':(False, False)}
    colors={'Online Continual':'C2',
             'Online Continual No Hardbuffer':'C3',
             'Online':'C4',
             'Online No Hardbuffer':'C5'}

    for tag in tags:
        print("\n{0}".format(tag))
        # losses, loss_window_means, update_tags, loss_window_variances, test_loss=experiment(use_hard_buffer,continual_learning)
        results=learning_object.method(data,
                                        use_hard_buffer=settings[tag][0],
                                        continual_learning=settings[tag][1])
        training_losses[tag], loss_window_means[tag], update_tags[tag], loss_window_variances[tag], test_losses[tag] = results
    # Plot loss window mean, variance and update for each tag
    #if False and 'Online Continual' in tags or 'Online Continual No Hardbuffer' in tags:
    # if 'Online Continual' in tags or 'Online Continual No Hardbuffer' in tags:
    #     for dataname in ['loss_window_means','update_tags','loss_window_variances']:
    #         legend=[]
    #         plt.title(dataname)
    #         #for i in range(ntasks): plt.axvline(x=(i+1)*ntrain,color='gray')
    #         for tag in tags:    
    #             plt.plot(eval(dataname)[tag],color=colors[tag])
    #             legend.append(mpatches.Patch(color=colors[tag], label=tag))
    #         plt.legend(handles=legend)
    #         plt.axis('off')
    #         plt.show()
    #print loss_window_means
    
    # Plot training loss
    if False:
        legend=[]
        plt.title('training accuracy')
        #for i in range(ntasks): plt.axvline(x=(i+1)*ntrain,color='gray')
        for tag in tags:    
            plt.plot(training_losses[tag][::10],color=colors[tag])
            legend.append(mpatches.Patch(color=colors[tag], label=tag))
        plt.legend(handles=legend)
        plt.axis('off')
        plt.show()
        
    # Plot test loss
    # if False:
        subsample=1
        for task in range(2):
            legend=[]
            plt.title('test loss task {0}'.format(task))
            plt.ylim((0,1))
            #for i in range(ntasks): plt.axvline(x=(i+1)*ntrain,color='gray')
            for tag in tags:  
                plt.plot(np.arange(0,len(test_losses[tag][task]),subsample),test_losses[tag][task][::subsample],color=colors[tag])
                legend.append(mpatches.Patch(color=colors[tag], label=tag))
            plt.legend(handles=legend)
            plt.axis('off')
            plt.show()
            print(tag,task,test_losses[tag][task][-1]*100)

    # Get final average accuracy for each tag:
    for tag in tags:
        print("{0}: {1}".format(tag,np.mean([test_losses[tag][task][-1] for task in range(ntasks)])))
        for task in range(ntasks): 
            print("{0}: task {1}: {2}".format(tag,task,test_losses[tag][task][-1]))
            plt.figure(figsize=(10, 6))  # create a new figure with specified size
            plt.plot(test_losses[tag][task], marker='o')  # plot the test_losses list
            plt.title('Test Accuracy')  # set the title of the plot
            plt.xlabel('Epoch')  # set the label for x-axis
            plt.ylabel('Accurcy')  # set the label for y-axis
            plt.grid(True)  # add a grid
            plt.show()  
    return [np.mean([test_losses[tag][task][-1] for task in range(ntasks)]) for tag in sorted(tags)]


def main():

    # number or tasks or quadrants
    ntasks=2
    dim=4
    file = '/Users/meruozhu/Downloads/MP_data/MP_codes/MP/Data/Helpdesk.csv'
    data = pd.read_csv(file, low_memory=False)
    timeformat = "%Y-%m-%d %H:%M:%S"
    numEvents = data.shape[0]
    print("Num events is {}".format(numEvents))

    dataName = 'Helpdesk_mini'
    log = LogFile(filename=file, delim=",", header=0, rows=None, time_attr="completeTime", trace_attr="case",
                        activity_attr='event', convert=False)
    d = Data(dataName,
                LogFile(filename=file, delim=",", header=0, rows=None, time_attr="completeTime", trace_attr="case",
                        activity_attr='event', convert=False))
    d.logfile.keep_attributes(['event', 'role', 'completeTime'])
    m = Methods.get_prediction_method("SDL")
    s = setting.STANDARD
    trainPerc = 0.8
    s.train_percentage = trainPerc * 100
    # # #
    d.prepare(s)
    print('data prepared done')
    pretrained_model_path = '/Users/meruozhu/Downloads/MP_data/MP_codes/MP/DPM/pretrained_model/Helpdesk'
    #if os.path.exists(pretrained_model_path):
    # if False:
    # # load the pretrained model if it exists
    #     basic_model = tf.keras.models.load_model(pretrained_model_path)
    #     print('load model done')
    # else:
    #     # d.create_batch("normal", timeformat)
    #     is_written = 0

    #     start_time = time.time()
    #     print("Test Context Data")
    #     print(d.test_orig.contextdata)
    #     basic_model = m.train(d.train)
    #     print('train model done')
    basic_model=AdaptiveRandomForestClassifier()
    data_sampler = Sampler(data=d)
    print('make sampler done')
    learning_object=Task_free_continual_learning_provide_model(verbose=False,
                                                        seed=123,
                                                        dev='cpu',
                                                        dim=4,
                                                        hidden_units=100,
                                                        learning_rate=0.005,
                                                        ntasks=2,
                                                        gradient_steps=5,
                                                        loss_window_length=5,
                                                        loss_window_mean_threshold=0.2,
                                                        loss_window_variance_threshold=0.1,                                                         
                                                        MAS_weight=0.5,
                                                        recent_buffer_size=100,
                                                        hard_buffer_size=20,model=basic_model)
    print('get tfcl done')
    #tags=['Online No Hardbuffer', 'Online Continual']
    tags=['Online Continual']
    print(experiment(data_sampler, learning_object, tags))




if __name__ == '__main__':
    main()    
