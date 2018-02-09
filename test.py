# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 15:44:00 2018

@author: hanze
"""

## Load from assistant data
import os, time, pickle
import numpy as np
import scipy.sparse

#from datasets.utils import convert_to_pairwise_sequence

from algs.utils import get_adjacent_state_matrix
from algs.markov import Markov_HR, markov_get_transition_matrix
from algs.cfirl import CF_comp_data_structs, CF_HR_quicker_v2
from algs.nearestneigh import NN_HR_transmat, NN_HR_new, nn_cosine_structs,\
    nn_euclidean_structs
from algs.lstm import train_model, specify_model, RNN_HR



def run_lstm_exp(train_data_file, test_data_file, fixed_len, num_states, max_seq_len, num_neurs, hitratelvl, savefile):
    if isinstance(train_data_file, basestring) and isinstance(test_data_file, basestring):
        data, data_test = load_data(train_data_file, test_data_file)
    else:
        data = train_data_file
        data_test = test_data_file
    
    NUM_EPOCHS = 20
    
#     model_fname = 'results/tempvars/lastmodel_lstm_neur{:d}' + str(num_neurs)
    model_fname = savefile + '_neurs{:d}_msl{:d}_epochs{:d}'.format(num_neurs, max_seq_len, NUM_EPOCHS)
    model = specify_model(num_states, max_seq_len, num_neurs, dropout_rate=0.2, recc_dropout_rate=0.2)    
    model, _ = train_model(data, model, max_seq_len, num_states, NUM_EPOCHS, model_fname)
    
    RNN_hr_all, RNN_hr_len = RNN_HR(model, max_seq_len, data_test, fixed_len, num_states, hitratelvl)
        
    RNN_hr_all = np.asarray(RNN_hr_all)
    RNN_hr = np.zeros((2, len(fixed_len)))
    for i, t in enumerate(RNN_hr_len):
        RNN_hr[i,:] = np.array(t)
    
#     print('{:s}'.format(''.join(40 * ['-'])))
#     print('{:10s}{:>8s}{:10.2f}{:10.2f}'.format('cfirl', 'all', 100 * cfirl_hr_all[0], 100 * cfirl_hr_all[1]))
#     for i, l in enumerate(fixed_len):
#         print('{:10s}{:>8d}{:10.2f}{:10.2f}'.format('cfirl', l, 100 * cfirl_hr[0][i], 100 * cfirl_hr[1][i]))
#     print('{:s}\n'.format(''.join(40 * ['-'])))
  
    # save results
    if savefile != None and savefile != '':
        f = open(savefile, 'wb')
        pickle.dump([RNN_hr_all, RNN_hr], f)
        f.close()
        
        # over all states
        fname = savefile + '_all'
        np.savetxt(fname, 100 * np.array(RNN_hr_all), fmt='%.4f', delimiter=',', header='HR@1, HR@2')
        # length specific 
        fname = savefile + '_len'
        np.savetxt(fname, 100 * RNN_hr, fmt='%.4f', delimiter=',', header=str(fixed_len))


def  get_global_sequence(skillset_n):
    glb_seq = []
    for user in skillset_n:
        for assignment in user:
            glb_seq += [assignment]
    return glb_seq
        
        
        
if __name__ == '__main__':
    MAX_SEQ_LEN = 10
    NUM_STATES = 2
    ## Location of the kt dataset
    #C:\Users\hanze\Desktop\Research\cfirl-master\datasets
    ## Load KT dataset
    f = open(r'C:\Users\hanze\Desktop\Research\cfirl-master\datasets\ast_skmult.pkl', 'rb')
    data = pickle.load(f,encoding='latin1')
    f.close()
    
    ## Get skillset 0 
    skillset0 = data[0]
    ## convert to global model
    glb_dataset_0 = get_global_sequence(skillset0)
    glb_dataset_first_50 = glb_dataset_0[:800]
    glb_dataset_test = glb_dataset_0[800:880]
    
    ## covert to i
    
    x, y = prepare_data(glb_dataset_first_50, MAX_SEQ_LEN, num_states=NUM_STATES)
    
#     
    model = specify_model(NUM_STATES, MAX_SEQ_LEN, 20)
#     clear()
    model, hist = train_model(glb_dataset_first_50, model, MAX_SEQ_LEN, 2, 10, 'test_1')
    model.load_weights('test_1')
#     print(hist)
    
#     test_data = np.random.randint(low=0, high=6, size=(5,10))
#     test_data = [[1, 2, 3,2,3], [4,3,1,0,0], [3, 3, 3, 2,1], [1, 2, 1, 1,3]]
#     hr_all, hr_lem = LSTM_HR(model, MAX_SEQ_LEN, test_data, [2, 4], 6, 2)
#     print(hr_all, hr_lem)
     
    gr = validate_params(x, y, NUM_STATES, MAX_SEQ_LEN)
    print(gr)
    
    RNN_hr_all, RNN_hr_len = RNN_HR(model, MAX_SEQ_LEN, glb_dataset_test, [7,8], NUM_STATES, 2)
    
    RNN_hr_all = np.asarray(RNN_hr_all)
    RNN_hr = np.zeros((2, len(fixed_len)))
    for i, t in enumerate(RNN_hr_len):
        RNN_hr[i,:] = np.array(t)
    
#     print('{:s}'.format(''.join(40 * ['-'])))
#     print('{:10s}{:>8s}{:10.2f}{:10.2f}'.format('cfirl', 'all', 100 * cfirl_hr_all[0], 100 * cfirl_hr_all[1]))
#     for i, l in enumerate(fixed_len):
#         print('{:10s}{:>8d}{:10.2f}{:10.2f}'.format('cfirl', l, 100 * cfirl_hr[0][i], 100 * cfirl_hr[1][i]))
#     print('{:s}\n'.format(''.join(40 * ['-'])))
  
    # save results
    '''
    if savefile != None and savefile != '':
        f = open(savefile, 'wb')
        pickle.dump([RNN_hr_all, RNN_hr], f)
        f.close()
        
        # over all states
        fname = savefile + '_all'
        np.savetxt(fname, 100 * np.array(RNN_hr_all), fmt='%.4f', delimiter=',', header='HR@1, HR@2')
        # length specific 
        fname = savefile + '_len'
        np.savetxt(fname, 100 * RNN_hr, fmt='%.4f', delimiter=',', header=str(fixed_len))
'''
        
    
    