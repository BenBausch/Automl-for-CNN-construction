import os
import argparse
import logging
import time
import re
import numpy as np
import jsonpickle
from sklearn.model_selection import StratifiedKFold   # We use 3-fold stratified cross-validation

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from torchsummary import summary
from torchvision.datasets import ImageFolder

from cnn import torchModel
from population import *
from utils import get_optim, get_crit
from early_stopping import EarlyStopping

def init_population(population,
               img_width,
               img_height,
               batch_size,
               num_epochs,
               learning_rate,
               train_criterion,
               cv,
               data,
               device,
               save_model_str):
    """
    Initializes population with 5 different networks, that should be spread over the config space as far as possible.
    """
    # max amount of parameters
    config1 = {
        'optimizer': 0.2,
        'criterion': 0.2,
        'lr': learning_rate,
        'n_conv_layers': 3,
        'n_channels_conv_0': 2048,
        'n_channels_conv_1': 2048,
        'n_channels_conv_2': 2048,
        'kernel_size': 3,
        'global_avg_pooling': False,
        'use_BN': True,
        'n_fc_layers': 1,
        'n_channels_fc_0': 50,
        'n_channels_fc_1': 50,
        'n_channels_fc_2': 50}

    # min amount of paras for max num layers
    config2 = {
    'optimizer': 0.4,
    'criterion': 0.4,
    'lr': learning_rate,
    'n_conv_layers': 3,
    'n_channels_conv_0': 64,
    'n_channels_conv_1': 64,
    'n_channels_conv_2': 64,
    'kernel_size': 1,
    'global_avg_pooling': True,
    'use_BN': False,
    'n_fc_layers': 3,
    'n_channels_fc_0': 1,
    'n_channels_fc_1': 1,
    'n_channels_fc_2': 1}

    # config3 and config4 +- medium amount of parameters overall
    # differ in the other settings
    config3 = {
    'optimizer': 0.6,
    'criterion': 0.6,
    'lr': learning_rate,
    'n_conv_layers': 2,
    'n_channels_conv_0': 1200,
    'n_channels_conv_1': 1200,
    'kernel_size': 5,
    'global_avg_pooling': True,
    'use_BN': False,
    'n_fc_layers':1,
    'n_channels_fc_0': 273}

    config4 = {
    'optimizer': 0.8,
    'criterion': 0.8,
    'lr': learning_rate,
    'n_conv_layers':1,
    'n_channels_conv_0':900,
    'kernel_size':1,
    'global_avg_pooling': False,
    'use_BN': True,
    'n_fc_layers':2,
    'n_channels_fc_0':1,
    'n_channels_fc_1': 50}

    # default config of the project
    config5 = {
        'optimizer': 0.4,
        'criterion': 0.4,
        'lr': learning_rate,
        'n_conv_layers': 2,
        'n_channels_conv_0': 457,
        'n_channels_conv_1': 511,
        'n_channels_conv_2': 38,
        'kernel_size': 5,
        'global_avg_pooling': True,
        'use_BN': False,
        'n_fc_layers': 3,
        'n_channels_fc_0': 27,
        'n_channels_fc_1': 17,
        'n_channels_fc_2': 273}

    configs = [config1, config2, config3, config4, config5]

    print('Initalizing the population: ')

    for i, config in enumerate(configs):
        default = False
        if i == 4:
            default = True
        train_loop(
               population,
               img_width,
               img_height,
               batch_size,
               num_epochs,
               learning_rate,
               train_criterion,
               cv,
               data,
               config,
               device,
               save_model_str,
               default)


def load_last_population(path_to_last_pop,
                     path_to_models,
                     population,
                     population_number):
    f = open(path_to_last_pop + "\\" + "population" + str(population_number) + ".json").read()
    population_dict = jsonpickle.decode(f)
    # load all the models that are in the population
    candidate_list = []
    print("Searching for candiates: ")
    print(population_dict['candidate_ids'])
    cans = population_dict['candidate_ids']
    default_in_cans = False
    for i in cans:
        if i == 4:
            default_in_cans = True
            break
    # append default config
    if not(default_in_cans):
        cans.append(4)
    # find max model id
    max_id = 0
    for filename in os.listdir(path_to_models):
        if filename.endswith(".json") and filename.find('model') != -1:
            id_found = int(re.findall('[0-9]+', filename)[0])
            if id_found > max_id:
                max_id = id_found
    print("Max id found: %d" % max_id)
    for i in population_dict['candidate_ids']:
        for filename in os.listdir(path_to_models):
            if filename.endswith(".json") and filename.find('model' + str(i) + ".") != -1:
                print('Found Candiadate : ' + filename)
                f = open(path_to_models + '\\' + filename).read()
                c_d= jsonpickle.decode(f)
                # load the torch model of the candidate:
                model = torch.load(path_to_models + '\\' + filename[0:-5])
                candidate = Candidate(score=c_d['score'],
                                      size=c_d['size'],
                                      config=c_d['config'],
                                      model=model,
                                      HV=c_d['HV'],
                                      default=c_d['default'],
                                      id=c_d['id'],
                                      max_id=max_id
                                      )
                # only add default to cans is previously has been in cans
                if i == 4 and default_in_cans:
                    population.default = candidate
                    candidate_list.append(candidate)
                elif i == 4:
                    population.default = candidate
                else:
                    candidate_list.append(candidate)
    print(candidate_list)
    population.candidates = candidate_list
    population.use_size = population_dict['use_size']
    population.randomize = population_dict['randomize']
    population.generations_since_last_change_pareto = population_dict['gslcp']
    population.HV = population_dict['HV']


def train_loop(population,
               img_width,
               img_height,
               batch_size,
               num_epochs,
               learning_rate,
               train_criterion,
               cv,
               data,
               model_config,
               device,
               save_model_str,
               default):
    """
    Trains model and additionally it adds the model wrapped as candidate member to the population.
    :param cv: crossvalidation algorithm
    :param data:
    :param config:
    :return:
    """
    score = []
    model = None
    for train_idx, valid_idx in cv.split(data, data.targets):

        early_stopping = EarlyStopping(patience=10, verbose=False)

        train_data = Subset(data, train_idx)
        test_dataset = Subset(data, valid_idx)

        # image size
        input_shape = (3, img_width, img_height)

        # Make data batch iterable
        # Could modify the sampler to not uniformly random sample
        train_loader = DataLoader(dataset=train_data,
                                  batch_size=batch_size,
                                  shuffle=True)
        test_loader = DataLoader(dataset=test_dataset,
                                 batch_size=batch_size,
                                 shuffle=False)

        model = torchModel(model_config,
                           input_shape=input_shape,
                           num_classes=len(data.classes)).to(device)

        # THIS HERE IS THE SECOND OBJECTIVE YOU HAVE TO OPTIMIZE
        # THIS HERE IS THE SECOND OBJECTIVE YOU HAVE TO OPTIMIZE
        # THIS HERE IS THE SECOND OBJECTIVE YOU HAVE TO OPTIMIZE
        total_model_params = np.sum(p.numel() for p in model.parameters())
        # THIS HERE IS THE SECOND OBJECTIVE YOU HAVE TO OPTIMIZE
        # THIS HERE IS THE SECOND OBJECTIVE YOU HAVE TO OPTIMIZE
        # THIS HERE IS THE SECOND OBJECTIVE YOU HAVE TO OPTIMIZE
        # instantiate optimizer
        optimizer = get_optim(model_config['optimizer'])(model.parameters(),
                                    lr=learning_rate)

        # Just some info for you to see the generated network.
        logging.info('Generated Network:')
        summary(model, input_shape,
                device='cuda' if torch.cuda.is_available() else 'cpu')

        # Train the model
        for epoch in range(num_epochs):
            logging.info('#' * 50)
            logging.info('Epoch [{}/{}]'.format(epoch + 1, num_epochs))

            train_score, train_loss = model.train_fn(optimizer, train_criterion, train_loader, device)
            test_score = model.eval_fn(test_loader, device)

            logging.info('Split-Train accuracy %f', train_score)
            logging.info('Split-Test accuracy %f', test_score)

            early_stopping(test_score)

            if early_stopping.early_stop:
                print("Early stopping at iteration: %d" % epoch)
                break

    # THIS HERE IS PART OF THE FIRST OBJECTIVE YOU HAVE TO OPTIMIZE
    # THIS HERE IS PART OF THE FIRST OBJECTIVE YOU HAVE TO OPTIMIZE
    # THIS HERE IS PART OF THE FIRST OBJECTIVE YOU HAVE TO OPTIMIZE
    score.append(test_score)
    # THIS HERE IS PART OF THE FIRST OBJECTIVE YOU HAVE TO OPTIMIZE
    # THIS HERE IS PART OF THE FIRST OBJECTIVE YOU HAVE TO OPTIMIZE
    # THIS HERE IS PART OF THE FIRST OBJECTIVE YOU HAVE TO OPTIMIZE
    ref_point = [np.log10(10 ** 8), 0]
    hv = population.computeHV2D([[np.log10(total_model_params), 1 - np.mean(score)]], ref_point)

    candidate = Candidate(1 - np.mean(score),
                          total_model_params,
                          config=model_config,
                          model=model,
                          HV=hv)
    if default:
        population.default = candidate
    if save_model_str:
        # Save the model checkpoint can be restored via "model = torch.load(save_model_str)"
        print('Saving model!')
        if os.path.exists(save_model_str):
            save_model_str += r'\model'
            save_model_str += ''.join(str(candidate.id))
        torch.save(model.state_dict(), save_model_str)
        j = jsonpickle.encode(candidate.__dict__())
        with open(save_model_str + '.json', 'w') as outfile:
            outfile.write(j)


    population.add_candidate(candidate)

    # RESULTING SCORES FOR BOTH OBJECTIVES
    # RESULTING SCORES FOR BOTH OBJECTIVES
    # RESULTING SCORES FOR BOTH OBJECTIVES
    print('Resulting Model Score:')
    print('negative acc [%] | #num model parameters')
    print(1 - np.mean(score), total_model_params)


def main(data_dir,
         num_epochs=10,
         num_generations=2,
         batch_size=50,
         learning_rate=0.001,
         train_criterion=torch.nn.CrossEntropyLoss,
         model_optimizer=torch.optim.Adam,
         data_augmentations=None,
         save_model_str=None,
         path_to_population=None,
         population_number=-1,
         ):
    """
    Outter loop for the hyperparameter optimization and NAS.

    :param model_config:
    :param data_dir:
    :param num_epochs:
    :param batch_size:
    :param learning_rate:
    :param train_criterion:
    :param model_optimizer:
    :param data_augmentations:
    :param save_model_str:
    :return:
    """
    start = time.time()
    # Device configuration
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    img_width = 16
    img_height = 16
    # data_augmentations = [transforms.Resize([img_width, img_height]),
    #                      transforms.ToTensor()]
    if data_augmentations is None:
        # You can add any preprocessing/data augmentation you want here
        data_augmentations = transforms.ToTensor()
    elif isinstance(data_augmentations, list):
        data_augmentations = transforms.Compose(data_augmentations)
    elif not isinstance(data_augmentations, transforms.Compose):
        raise NotImplementedError

    # Load the dataset
    data = ImageFolder(data_dir, transform=data_augmentations)

    cv = StratifiedKFold(n_splits=3, random_state=42, shuffle=True)  # to make CV splits consistent

    # instantiate training criterion and population
    train_criterion = train_criterion().to(device)

    population = Population()
    if path_to_population is None:
        init_population(population,
                   img_width,
                   img_height,
                   batch_size,
                   num_epochs,
                   learning_rate,
                   train_criterion,
                   cv,
                   data,
                   device,
                   save_model_str)
    else:
        print("Loading last population")
        load_last_population(path_to_population, save_model_str, population, population_number)

    use_size = True
    for g in range(population_number + 1,num_generations):
        if time.time() - start >= 86400:
            population.plot_pareto_set(population.compute_pareto_set(), g)
            break
        num_children = 5

        # if Hypervolume did not change within last few gens (controled in population.plot_paerto front)
        # then randomize new children
        if True:
            # get 5 parents and randomly select parent tuples for
            if g % 3 == 0:
                use_size = True
                population.use_size = use_size
            else:
                use_size = False
                population.use_size = use_size
            if use_size:
                print('Sample Size')
                candidates = list(population.sample_by_dist_prop_to_size_then_tournament(num_children))
            else:
                print('Sample HV')
                candidates = list(population.sample_by_hypervolume(num_children))
            index_tupels = []
            for i in range(len(candidates)):
                for j in range(i + 1, len(candidates)):
                    index_tupels.append((i,j))
            indexes = np.random.choice([i for i in range(len(index_tupels))], len(candidates), replace=False)
            choice = [index_tupels[i] for i in indexes]
            for c in choice:
                if time.time() - start >= 86400:
                    break
                child_config = population.produce_child(candidates[c[0]], candidates[c[1]])
                train_loop(population,
                           img_width,
                           img_height,
                           batch_size,
                           num_epochs,
                           learning_rate,
                           train_criterion,
                           cv,
                           data,
                           child_config,
                           device,
                           save_model_str,
                           False)
        else:
            #pareto = list(map(lambda x: x, iter(population.compute_pareto_set())))
            for i in range(num_children):
                #parent = np.random.choice(pareto)
                child_config = population.sample_child_uniformly()
                train_loop(population,
                           img_width,
                           img_height,
                           batch_size,
                           num_epochs,
                           learning_rate,
                           train_criterion,
                           cv,
                           data,
                           child_config,
                           device,
                           save_model_str,
                           False)
        if save_model_str:
            # Save the model checkpoint can be restored via "model = torch.load(save_model_str)"
            print('Saving Population!')
            if os.path.exists(save_model_str):
                a = save_model_str
                a += r'\population'
                a += ''.join(str(g))
            j = jsonpickle.encode(population.__dict__())
            with open(a + '.json', 'w') as outfile:
                outfile.write(j)

        population.plot_pareto_set(population.compute_pareto_set(), g)
        population.kill_weak_candidates()


if __name__ == "__main__":

    loss_dict = {'cross_entropy': torch.nn.CrossEntropyLoss,
                 'mse': torch.nn.MSELoss}
    opti_dict = {'adam': torch.optim.Adam,
                 'adamw': torch.optim.AdamW,
                 'adad': torch.optim.Adadelta,
                 'sgd': torch.optim.SGD}

    # encoding of the categorical hyper parameters into continuous space i = included e=excluded
    # example 0i-0.25i for adam, 0.25e-0.5i adamw, 0.5e-0.75i adad, 0.75e-1i sgd

    cmdline_parser = argparse.ArgumentParser('AutoML SS20 final project')

    cmdline_parser.add_argument('-e', '--epochs',
                                default=50,
                                help='Number of epochs',
                                type=int)
    cmdline_parser.add_argument('-g', '--generations',
                                default=30,
                                help='Number of generations of generated of spring',
                                type=int)
    cmdline_parser.add_argument('-b', '--batch_size',
                                default=200,
                                help='Batch size',
                                type=int)
    cmdline_parser.add_argument('-D', '--data_dir',
                                default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                     '..', 'micro17flower'),
                                help='Directory in which the data is stored (can be downloaded)')
    cmdline_parser.add_argument('-l', '--learning_rate',
                                default=2.244958736283895e-05,
                                help='Optimizer learning rate',
                                type=float)
    cmdline_parser.add_argument('-L', '--training_loss',
                                default='cross_entropy',
                                help='Which loss to use during training',
                                choices=list(loss_dict.keys()),
                                type=str)
    cmdline_parser.add_argument('-o', '--optimizer',
                                default='adamw',
                                help='Which optimizer to use during training',
                                choices=list(opti_dict.keys()),
                                type=str)
    cmdline_parser.add_argument('-m', '--model_path',
                                default=r'.\src\models',
                                help='Path to store model',
                                type=str)
    cmdline_parser.add_argument('-p', '--path_to_pop',
                                default=r'.\src\models',
                                help='Path to stored last population (only folder)',
                                type=str)
    cmdline_parser.add_argument('-n', '--pop_num',
                                default=9,
                                help='Number of last population',
                                type=str)
    cmdline_parser.add_argument('-v', '--verbose',
                                default='INFO',
                                choices=['INFO', 'DEBUG'],
                                help='verbosity')
    args, unknowns = cmdline_parser.parse_known_args()
    log_lvl = logging.INFO if args.verbose == 'INFO' else logging.DEBUG
    logging.basicConfig(level=log_lvl)

    if unknowns:
        logging.warning('Found unknown arguments!')
        logging.warning(str(unknowns))
        logging.warning('These will be ignored')

    # architecture parametrization
    main(
        data_dir=args.data_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        num_generations=args.generations,
        learning_rate=args.learning_rate,
        data_augmentations=None,  # Not set in this example
        save_model_str=args.model_path,
        path_to_population=args.path_to_pop,
        population_number=args.pop_num
    )
