import matplotlib.pyplot as plt
from population import *
from utils import *
import os
import argparse
import jsonpickle
from collections import defaultdict

def plot_setting(name, path, save_path):
    values = []
    d = defaultdict()
    max_Hv = 0
    val = None
    bins = 0
    present_counter = 0
    for filename in os.listdir(path):
        if filename.endswith(".json") and filename.find('population') == -1:
            f = open(path + '\\' + filename).read()
            candidate = jsonpickle.decode(f)
            config = candidate['config']
            try:
                v = config[name]
                if max_Hv < candidate['HV']:
                    val = v
                    max_Hv = candidate['HV']
                values.append(v)
                if v in d.keys():
                    d[v] += 1
                else:
                    d[v] = 1
                    bins += 1
                present_counter += 1
            except:
                continue
        else:
            continue
    if name == 'criterion' or name == 'optimizer':
        pass
    else:
        print(name)
        fig, ax = plt.subplots()
        if name == 'global_avg_pooling' or name == 'use_BN':
            values = [(1 if v else 0) for v in values]
            labels = ['False', 'True']
            plt.xticks([0.25,0.75], labels)
            ax.set_xticklabels(labels)
        plt.title(name + ' present in ' + str(present_counter) + ' configs')
        plt.hist(values, bins=bins)
        txt = "Best HV (%d) for value: %d" % (max_Hv, val)
        plt.figtext(0.0, 0.00, txt, wrap=True, fontsize=10)
        plt.savefig(fname= save_path + '\\' + name)
        plt.clf()

if __name__ == "__main__":

    cmdline_parser = argparse.ArgumentParser('AutoML SS20 final project')
    cmdline_parser.add_argument('-f', '--model_path',
                                default=r'.\src\models',
                                help='Path where models are stored',
                                type=str)
    cmdline_parser.add_argument('-s', '--stats_path',
                                default=r'.\src\stats',
                                help='Path to plot stats',
                                type=str)
    args, unknowns = cmdline_parser.parse_known_args()
    hyperparams = {
        'optimizer',
        'criterion',
        'n_conv_layers',
        'n_channels_conv_0',
        'n_channels_conv_1',
        'n_channels_conv_2',
        'kernel_size',
        'global_avg_pooling',
        'use_BN',
        'n_fc_layers',
        'n_channels_fc_0',
        'n_channels_fc_1',
        'n_channels_fc_2'}

    for para in iter(hyperparams):
        plot_setting(para, args.model_path, args.stats_path)