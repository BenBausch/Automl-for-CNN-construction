from itertools import count
import numpy as np
from utils import get_borders
from utils import within_borders

import matplotlib.pyplot as plt


class Candidate():
    """
    Wrapper class for the Cnn models.
    """
    _num_candidates = count(0)

    def __init__(self, score, size, config, model):
        self.id = next(self._num_candidates)
        self.score = score
        self.size = size
        self.model = model
        self.init_configuration(config)

    def init_configuration(self, config):
        self.optimizer = config['optimizer']
        self.criterion = config['criterion']
        self.n_convoultions = config['n_conv_layers']
        self.n_conv_filters = []
        for i in range(self.n_convoultions):
            f = config['n_channels_conv_' + str(i)]
            self.n_conv_filters.append(f)
        self.kernel_size = config['kernel_size']
        self.global_avg_pooling = config['global_avg_pooling']
        self.use_BN = config['use_BN']
        self.n_fc_layers = config['n_fc_layers']
        self.n_fc_channels = []
        for i in range(self.n_fc_layers):
            f = config['n_channels_fc_' + str(i)]
            self.n_fc_channels.append(f)


class Population():
    """
    A population consists of a set of candidates.

    1) initalization outside of class
    2) selecting parents by sample_by_dist_prop_to_size_then_tournament
    3) Variations by
    """

    def __init__(self):
        self.candidates = []

    def add_candidate(self, candidate):
        self.candidates.append(candidate)
        self.candidates.sort(key=lambda x: x.size, reverse=False)

    def sample_by_dist_prop_to_size_then_tournament(self, number_children):
        """
        Samples #number_candidates candidates by the a distribution proportional of their size.
        Small configs in terms of # learnable parameters should be more likely to be sampled.
        For this function the self.candidates has to be sorted by size.
        After sampling, model with best performance is selected.
        Reapts process num
        """
        # select number_candidates from self.candidates without re-sampling
        # get number parameters of each candidate
        # TODO: test this line
        possible_parents = []
        for cand in self.candidates:
            possible_parents.append(cand)
        candidate_set = set()

        if number_children > len(possible_parents):
            for c in self.candidates:
                candidate_set.add(c)
            return c

        for j in range(number_children):

            print('Possible parents by ids:')
            print(list(map(lambda x: x.id, possible_parents)))

            # calculate probs proportional to size of candidate and
            # reverse the list to make small networks more probable
            probs = list(map(lambda x: x.size, possible_parents))
            total_parameters = sum(probs)
            for i in range(len(probs)):
                probs[i] = probs[i] / total_parameters
            print("Number Children left: %d:" % (number_children - (j+1)))
            if number_children - j >= len(possible_parents):
                candidates = np.random.choice(len(possible_parents), len(possible_parents), replace=False, p=probs[::-1])
            else:
                candidates = np.random.choice(len(possible_parents), number_children, replace=False, p=probs[::-1])
            print('Possible choices: ')
            for i in candidates:
                cand = possible_parents[i]
                print('Possible Candidate Id : %d and Score: %d and Size: %d' %
                      (cand.id, cand.score, cand.size))

            # Get the candidate with best performance from the sampled candidates and remove its index from the possible
            # parent candidate of further children (prevent getting same candidate multiple times)
            tournement_candidates = list(map(lambda c:(possible_parents[c],c), candidates))
            best_candidate, index = tournement_candidates[np.argmax(list(map(lambda c: c[0].score, tournement_candidates)))]
            print('Selected Candidate Id : %d and Score: %d and Size: %d' %
                  (best_candidate.id, best_candidate.score, best_candidate.size))
            possible_parents.pop(index)
            candidate_set.add(best_candidate)
        return candidate_set

    def produce_child(self, parent1: Candidate, parent2:Candidate) -> Candidate:
        """
        Children produced by recombination of the parents.
        Depending on hyperparameter different recombination types used.
        """

        config = {}

        # determine number of conv layers of child by uniform crossover
        n_convs = np.random.choice([parent1.n_convoultions, parent2.n_convoultions], p=[0.5, 0.5])
        n_convs = within_borders(n_convs, get_borders('n_conv_layers')[1])
        config['n_conv_layers'] = n_convs

        # determine number filters by intermediate recombination of parents with flooring the value
        # uniform sample value within borders if not available
        for i in range(n_convs):
            try:
                filter_size = int(0.5*(parent1.n_conv_filters[i] + parent2.n_conv_filters[i]))
            except:
                borders = get_borders('n_channels_conv_' + str(i))[1]
                try:
                    filter_size = int(0.5 * (np.random.randint(borders[0], borders[1]) + parent2.n_conv_filters[i]))
                except:
                    filter_size = int(0.5 * (parent1.n_conv_filters[i] + np.random.randint(borders[0], borders[1])))

            config['n_channels_conv_' + str(i)] = within_borders(filter_size, get_borders('n_channels_conv_' + str(i))[1])

        # determine number of fc layers of child by uniform crossover
        n_fcs = np.random.choice([parent1.n_fc_layers, parent2.n_fc_layers], p=[0.5, 0.5])
        n_fcs = within_borders(n_fcs, get_borders('n_fc_layers')[1])
        config['n_fc_layers'] = n_fcs

        # determine number channels by intermediate recombination of parents with flooring the value
        # uniform sample value within borders if not available
        for i in range(n_fcs):
            try:
                num_channels = int(0.5 * (parent1.n_fc_channels[i] + parent2.n_fc_channels[i]))
            except:
                borders = get_borders('n_channels_fc_' + str(i))[1]
                try:
                    num_channels = int(0.5 * (np.random.randint(borders[0], borders[1]) + parent2.n_fc_channels[i]))
                except:
                    num_channels = int(0.5 * (parent1.n_fc_channels[i] + np.random.randint(borders[0], borders[1])))

            config['n_channels_fc_' + str(i)] = within_borders(num_channels, get_borders('n_channels_fc_' + str(i))[1])

        # choose optimizer by simulated binary crossover
        opti = 0.5 * (parent1.optimizer + parent2.optimizer)
        beta = np.random.uniform(-1, 1)
        opti = opti + 0.5 * beta * np.abs(parent1.optimizer - parent2.optimizer)
        config['optimizer'] = within_borders(opti, get_borders('optimizer')[1])

        # choose criterion by simulated binary crossover
        crit = 0.5 * (parent1.criterion + parent2.criterion)
        beta = np.random.uniform(-1, 1)
        crit = crit + 0.5 * beta * np.abs(parent1.criterion - parent2.criterion)
        config['criterion'] = within_borders(crit, get_borders('criterion')[1])

        # choose kernel size as a uniform mutation ( does not depend on parents )
        borders = get_borders('kernel_size')[1]
        config['kernel_size'] = np.random.randint(borders[0], borders[1])

        # choose global average pooling and batch normalization by applying logical AND
        config['global_avg_pooling'] = parent1.global_avg_pooling and parent2.global_avg_pooling
        config['use_BN'] = parent1.use_BN and parent2.use_BN

        return config

    def compute_pareto_set(self):
        dominated = [False for c in self.candidates]
        for i, c1 in enumerate(self.candidates):
            # if c1 already dominated by some candidate no need to check again
            if not dominated[i]:
                for j, c2 in enumerate(self.candidates):
                    if c2.score <= c1.score and c2.size < c1.size:
                       dominated[i] = True
                    elif c2.score < c1.score and c2.size <= c1.size:
                        dominated[i] = True
                    elif c1.score <= c2.score and c1.size < c2.size:
                        dominated[j] = True
                    elif c1.score < c2.score and c1.size <= c2.size:
                        dominated[j] = True
        pareto_set = set()
        for i, c in enumerate(self.candidates):
            if not dominated[i]:
                pareto_set.add(c)
        return pareto_set

    def plot_pareto_set(self, pareto_set):

        # size and score of the base model
        size_b = [self.candidates[4].size]
        score_b = [self.candidates[4].score]


        # sizes/scores of candidates in the pareto front
        sizes_p = list(map(lambda x: np.log(x.size), iter(pareto_set)))
        scores_p = list(map(lambda x: x.score, iter(pareto_set)))

        # sizes/scores of all candidates in population
        sizes_a = list(map(lambda x: np.log(x.size), iter(self.candidates)))
        scores_a = list(map(lambda x: x.score, iter(self.candidates)))

        plt.scatter(score_b, size_b, c='r', alpha=1)
        plt.scatter(scores_a, sizes_a, c='b', alpha=0.5)
        plt.scatter(scores_p, sizes_p, c='g', alpha=1)
        plt.title('Scatter plot pythonspot.com')
        plt.xlabel('size in log scale')
        plt.ylabel('score')
        plt.show()


"""if __name__ == "__main__":
    p = Population()
    for i in range(1,11):
        p.add_candidate(Candidate(10*i, i, 0, 0))
    p.sample_by_dist_prop_to_size_then_tournament(4)"""