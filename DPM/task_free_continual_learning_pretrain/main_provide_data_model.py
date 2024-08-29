import sys
sys.path.append('/Users/meruozhu/Downloads/MP_data/MP_codes/MP')
from DPM.task_free_continual_learning.method_provide_model_test import Task_free_continual_learning_provide_model
from DPM.task_free_continual_learning.sampler_provide_model import Sampler

import matplotlib.pyplot as plt
import numpy as np
import os, sys, time
import matplotlib.patches as mpatches

import pandas as pd
from Data.data import Data
from edbn.Utils.LogFile import LogFile
import edbn.Predictions.setting as setting
from edbn import Methods

# from method import Task_free_continual_learning
# from sampler import Sampler

ntasks=1

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
        training_losses, loss_window_means[tag], update_tags[tag], loss_window_variances[tag],future_losses, prediction_results = results
  
    if False:
        legend=[]
        plt.title('training accuracy')
        #for i in range(ntasks): plt.axvline(x=(i+1)*z,color='gray')
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
    #return [np.mean([test_losses[tag][task][-1] for task in range(ntasks)]) for tag in sorted(tags)]
    tag='Online Continual'
    return training_losses, future_losses, loss_window_means[tag], update_tags[tag], loss_window_variances[tag], prediction_results


def main(file,recent_buffer_size,hard_buffer_size,MAS_weight):
    # Start measuring the running time
    start_time = time.time()

    data = pd.read_csv(file, low_memory=False)
    timeformat = "%Y-%m-%d %H:%M:%S"
    numEvents = data.shape[0]
    print("Num events is {}".format(numEvents))

    # Extract the filename from the file path
    filename = os.path.basename(file)

    # Remove the file extension from the filename
    dataName = os.path.splitext(filename)[0]

    #dataName = 'Helpdesk_mini'
#    log = LogFile(filename=file, delim=",", header=0, rows=None, time_attr="completeTime", trace_attr="case",
#                        activity_attr='event', convert=False)
    d = Data(dataName,
                LogFile(filename=file, delim=",", header=0, rows=None, time_attr="completeTime", trace_attr="case",
                        activity_attr='event', convert=False))
    if 'role' in d.logfile.get_data().columns:
        d.logfile.keep_attributes(['event', 'role', 'completeTime'])
    else:
        d.logfile.keep_attributes(['event', 'completeTime'])
    m = Methods.get_prediction_method("SDL")
    s = setting.STANDARD
    trainPerc = 0.01
    s.train_percentage = trainPerc * 100
    # # #
    d.prepare(s)
    print('data prepared done')
   
    start_time = time.time()
    print("Test Context Data")
    #print(d.test_orig.contextdata)
    basic_model = m.train(d.train,{"epochs": 100, "early_stop": 10})
    print('train model done')
    
    connect_symbol="-"
    if '/' in d.logfile.get_data()['completeTime'][0]:
        connect_symbol='/'
    formats = [f"%Y{connect_symbol}%m{connect_symbol}%d %H:%M:%S%z", f'%Y{connect_symbol}%m{connect_symbol}%d %H:%M:%S',f'%Y{connect_symbol}%m{connect_symbol}%d %H:%M:%S.%f']

    for timeformat in formats:
        try:
            pd.to_datetime(d.logfile.get_data()['completeTime'], format=timeformat,exact=True)
            print(d.logfile.get_data()['completeTime'])
            print("The time format is:", timeformat)
            break
        except ValueError:
            continue
    print('timeformat',timeformat)
    # if 'role' in d.logfile.get_data().columns:
    #     if dataName == "DomesticDeclarations":
    #         first_occurrence_index = 9878
    #         first_occurrence_ratio = 17.5
    #     elif dataName == 'InternationalDeclarations':
    #         first_occurrence_index = 12467
    #         first_occurrence_ratio = 17.2
    #     elif dataName == 'PermitLog':
    #         first_occurrence_index = 13677
    #         first_occurrence_ratio = 15.7
    #     elif dataName == 'PrepaidTravelCost_new':
    #         first_occurrence_index = 2367
    #         first_occurrence_ratio = 12.9
    #     elif dataName == 'RequestForPayment_new':
    #         first_occurrence_index = 4913
    #         first_occurrence_ratio = 13.3
    #     # elif dataName == 'Road_Traffic_Fine_Management_Process':
    #     d.add_data_to_test_orig([first_occurrence_ratio,50],timeformat,ratio)
    #     # d.add_data_to_test_orig([17.5,50],timeformat,ratio)

    data_sampler = Sampler(data=d)
    print('make sampler done')
    learning_object=Task_free_continual_learning_provide_model(verbose=False,
                                                        seed=123,
                                                        dev='cpu',
                                                        dim=4,
                                                        hidden_units=100,
                                                        learning_rate=0.005,
                                                        ntasks=1,
                                                        gradient_steps=10,
                                                        loss_window_length=5,
                                                        loss_window_mean_threshold=0.2,
                                                        loss_window_variance_threshold=0.1,                                                         
                                                        MAS_weight=MAS_weight,
                                                        recent_buffer_size=recent_buffer_size,
                                                        hard_buffer_size=hard_buffer_size,model=basic_model)
    print('get tfcl done')
    #tags=['Online No Hardbuffer', 'Online Continual']
    tags=['Online Continual']
    training_losses, future_losses,loss_window_means, update_tags, loss_window_variances, prediction_results = experiment(data_sampler, learning_object, tags)
    # End measuring the running time
    end_time = time.time()
    # Calculate the elapsed time
    running_time = end_time - start_time

    # Print the elapsed time
    print(f"Running time: {running_time} seconds")
   
    return training_losses, future_losses, running_time, loss_window_means, update_tags, loss_window_variances, prediction_results




# if __name__ == '__main__':
#     main()    