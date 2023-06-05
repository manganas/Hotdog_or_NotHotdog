import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import pickle

from typing import Tuple

def get_pkl_dicts(dicts_dir):
    dict_dict = {}
    for file_ in Path(dicts_dir).glob('**/*.pkl'):
        # print(file_.stem)
        with open(file_, 'rb') as handle:
            tmp = pickle.load(handle)
            dict_dict[file_.stem] = tmp

    return dict_dict
        
def plot_indiv(exp_name:str, out_dict:dict, output_dir:str, figsize: None | Tuple[int]=None)->None:
    train_loss = out_dict['train_loss']
    test_loss = out_dict['test_loss']

    train_acc = out_dict['train_acc']
    test_acc = out_dict['test_acc']


    epochs = np.arange(len(train_loss))+1

    fig = plt.figure(figsize=figsize, constrained_layout=True)
    # plt.tight_layout()
    
    plt.subplot(1,2,1)
    plt.plot(epochs, train_loss, label='train loss')
    plt.plot(epochs, test_loss, 'r--', label='test_loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Losses vs epochs')
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(epochs, train_acc, label='train acc')
    plt.plot(epochs, test_acc, 'r--', label='test acc')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy %')
    plt.title('Accuracy vs epochs')
    plt.legend()

    plt.suptitle(exp_name[9:])

    plt.savefig(f"{output_dir}/{exp_name[9:]}.pdf")

    plt.show()
    
    

def plot_comp_experiments(dict_dict:dict, metric:str, output_dir:str, test=True):

    if metric.lower().strip()=='loss':
        mtrc='loss'
    elif metric.lower().strip()=='acc':
        mtrc = 'acc'
    
    else:
        raise NotImplementedError
    
    if not test:
        type_='train'
    else:
        type_ = 'test'
    
    key_ = f"{type_}_{mtrc}"

    names = []
    vals = []

    for k, v in dict_dict.items():
        names.append(k)
        vals.append(v[key_])

    epochs = np.arange(len(vals[0]))+1
    
    fig = plt.figure()
    for i in range(len(names)):
        plt.plot(epochs, vals[i], label=names[i][9:])

    plt.legend()
    plt.title(f"{type_.title()} {mtrc}")
    
    plt.savefig(f'{output_dir}/{type_}_{mtrc}.pdf')
    plt.show()
    

def main():
    
    OUTPUT_DIR = Path.cwd()/'outputs'
    SAVE_DIR = Path.cwd()/'figures' # can change to global figures
    SAVE_DIR.mkdir(exist_ok=True)

    # Get all the dictionaries I have in the outputs dir
    dict_dict = get_pkl_dicts(OUTPUT_DIR)
    
    # For each dict key, an experiment, plot the corresponding acc and losses
    for k, v in dict_dict.items():
        plot_indiv(k,v, SAVE_DIR)

    # Plot accuracies for test set together for comparison
    plot_comp_experiments(dict_dict, 'acc', SAVE_DIR, test=True)
    

if __name__=='__main__':
    main()