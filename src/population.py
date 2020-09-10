from itertools import count
import numpy as np
from utils import get_borders
from utils import within_borders

import matplotlib.pyplot as plt
import pickle
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sb

class Candidate():
    """
    Wrapper class for the Cnn models.
    """
    _num_candidates = count(0)
    _ids_set_after_crash = count(0)

    def __init__(self, score, size, config, model, HV, default=None, id=None, max_id=0):
        if not(id is None):
            self.id = id
            if next(self._ids_set_after_crash) == 0:
                for i in range(max_id):
                    print(next(self._num_candidates))
                self._ids_set_after_crash = True
        else:
            self.id = next(self._num_candidates)
            print("ID :" + str(self.id))
        self.score = score
        self.size = size
        # now set to none since not not needed
        self.model = None
        self.init_configuration(config)
        self.config = config
        self.default = default
        self.HV = HV

    def __dict__(self):
        d = {}
        d['id'] = self.id
        d['score'] = self.score
        d['size'] = self.size
        d['config'] = self.config
        d['default'] = self.default
        #d['model'] = self.model
        d['HV'] = self.HV
        return d

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
        self.use_size = True
        self.randomize = False
        self.generations_since_last_change_pareto = 0
        self.HV = 0
        self.default = None


    def __dict__(self):
        d = {}
        d['use_size'] = self.use_size
        d['randomize'] = self.randomize
        d['gslcp'] = self.generations_since_last_change_pareto
        d['HV'] = self.HV
        candidate_ids = []
        for c in self.candidates:
            candidate_ids.append(c.id)
        d['candidate_ids'] = candidate_ids
        return d



    def add_candidate(self, candidate):
        self.candidates.append(candidate)
        if True:
            # use this line when sampling by size
            self.candidates.sort(key=lambda x: x.size, reverse=False)
        else:
            # use this line when sampling score
            self.candidates.sort(key=lambda x: x.score, reverse=False)

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
            probs = list(map(lambda x: np.log10(x.size), possible_parents))
            total_parameters = sum(probs)
            for i in range(len(probs)):
                probs[i] = probs[i] / total_parameters
            print("Number Children left: %d:" % (number_children - (j)))
            if number_children >= len(possible_parents):
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
            best_candidate, index = tournement_candidates[np.argmin(list(map(lambda c: c[0].score, tournement_candidates)))]
            print('Selected Candidate Id : %d and Score: %d and Size: %d' %
                  (best_candidate.id, best_candidate.score, best_candidate.size))
            possible_parents.pop(index)
            candidate_set.add(best_candidate)
        return candidate_set

    def sample_by_dist_prop_to_score_then_tournament(self, number_children):
        """
        Samples #number_candidates candidates by the a distribution proportional of their score.
        Better configs in terms of score should be more likely to be sampled.
        For this function the self.candidates has to be sorted by score.
        After sampling, model with best performance is selected.
        Reapts process num
        """
        # select number_candidates from self.candidates without re-sampling
        # get number parameters of each candidate
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
            probs = list(map(lambda x: x.score, possible_parents))
            total_parameters = sum(probs)
            for i in range(len(probs)):
                probs[i] = probs[i] / total_parameters
            print("Number Children left: %d:" % (number_children - (j)))
            if number_children >= len(possible_parents):
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
            best_candidate, index = tournement_candidates[np.argmin(list(map(lambda c: c[0].size, tournement_candidates)))]
            print('Selected Candidate Id : %d and Score: %d and Size: %d' %
                  (best_candidate.id, best_candidate.score, best_candidate.size))
            possible_parents.pop(index)
            candidate_set.add(best_candidate)
        return candidate_set

    def sample_by_hypervolume(self, number_children):
        probs = list(map(lambda x: x.HV, self.candidates))
        total_parameters = sum(probs)
        for i in range(len(probs)):
            probs[i] = probs[i] / total_parameters
        if number_children >= len(self.candidates):
            candidates = np.random.choice(len(self.candidates), len(self.candidates), replace=False, p=probs[::-1])
        else:
            candidates = np.random.choice(len(self.candidates), number_children, replace=False, p=probs[::-1])
        print('Selected candidates: ')
        candidate_set = set()
        for i in candidates:
            cand = self.candidates[i]
            candidate_set.add(cand)
            print('Selected Candidate Id : %d and Score: %d and Size: %d' %
                  (cand.id, cand.score, cand.size))
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
            beta = np.random.normal(0.5, 0.5)
            try:
                filter_size = int(beta*(parent1.n_conv_filters[i] + parent2.n_conv_filters[i]))
            except:
                borders = get_borders('n_channels_conv_' + str(i))[1]
                try:
                    filter_size = int(beta * (np.random.randint(borders[0], borders[1]) + parent2.n_conv_filters[i]))
                except:
                    filter_size = int(beta * (parent1.n_conv_filters[i] + np.random.randint(borders[0], borders[1])))

            config['n_channels_conv_' + str(i)] = within_borders(filter_size, get_borders('n_channels_conv_' + str(i))[1])

        # determine number of fc layers of child by uniform crossover
        n_fcs = np.random.choice([parent1.n_fc_layers, parent2.n_fc_layers], p=[0.5, 0.5])
        n_fcs = within_borders(n_fcs, get_borders('n_fc_layers')[1])
        config['n_fc_layers'] = n_fcs

        # determine number channels by intermediate recombination of parents with flooring the value
        # uniform sample value within borders if not available
        for i in range(n_fcs):
            beta = np.random.normal(0.5, 0.5)
            try:
                num_channels = int(beta * (parent1.n_fc_channels[i] + parent2.n_fc_channels[i]))
            except:
                borders = get_borders('n_channels_fc_' + str(i))[1]
                try:
                    num_channels = int(beta * (np.random.randint(borders[0], borders[1]) + parent2.n_fc_channels[i]))
                except:
                    num_channels = int(beta * (parent1.n_fc_channels[i] + np.random.randint(borders[0], borders[1])))

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
        borders = get_borders('global_avg_pooling')[1]
        config['global_avg_pooling'] = np.random.choice(borders, p=[0.5,0.5])
        # increase prob to use bn, since generally known to yield better results
        borders = get_borders('use_BN')[1]
        config['use_BN'] = np.random.choice(borders, p=[0.7,0.3])

        return config

    def mutate_parent(self, parent: Candidate):
        config = {}

        # determine number of conv layers of child by uniform crossover
        borders = get_borders('n_conv_layers')[1]
        n_convs = parent.n_convoultions
        config['n_conv_layers'] = within_borders(n_convs, get_borders('n_conv_layers')[1])

        # determine number filters by intermediate recombination of parents with flooring the value
        # uniform sample value within borders if not available
        for i in range(n_convs):
            borders = get_borders('n_channels_conv_' + str(i))[1]
            filter_size = int(parent.n_conv_filters[i] +
                               np.random.uniform() * np.random.randint(borders[0], borders[1]))
            config['n_channels_conv_' + str(i)] = within_borders(filter_size,
                                                                 get_borders('n_channels_conv_' + str(i))[1])

        # determine number of fc layers of child by uniform crossover
        borders = get_borders('n_fc_layers')[1]
        n_fcs = parent.n_fc_layers
        config['n_fc_layers'] = n_fcs

        # determine number channels by intermediate recombination of parents with flooring the value
        # uniform sample value within borders if not available
        for i in range(n_fcs):
            borders = get_borders('n_channels_fc_' + str(i))[1]
            num_channels = int(parent.n_fc_layers[i] +
                               np.random.uniform() * np.random.randint(borders[0], borders[1]))
            config['n_channels_fc_' + str(i)] = within_borders(num_channels, get_borders('n_channels_fc_' + str(i))[1])

        # choose optimizer by simulated binary crossover
        borders = get_borders('optimizer')[1]
        opti = parent.optimizer + np.random.uniform(0, 0.5) * np.random.uniform(borders[0], borders[1])
        config['optimizer'] = within_borders(opti, get_borders('optimizer')[1])

        # choose criterion by simulated binary crossover
        borders = get_borders('criterion')[1]
        crit = parent.criterion + np.random.uniform(0, 0.5) * np.random.uniform(borders[0], borders[1])
        config['criterion'] = within_borders(crit, get_borders('criterion')[1])

        # choose kernel size as a uniform mutation ( does not depend on parents )
        borders = get_borders('kernel_size')[1]
        config['kernel_size'] = np.random.randint(borders[0], borders[1])

        # choose global average pooling and batch normalization by applying logical AND
        values = get_borders('global_avg_pooling')[1]
        config['global_avg_pooling'] = np.random.choice(values, p=[0.5, 0.5])
        values = get_borders('use_BN')[1]
        config['use_BN'] = np.random.choice(values, p=[0.5, 0.5])

        return config

    def sample_child_uniformly(self):
        config = {}

        # determine number of conv layers of child by uniform crossover
        borders = get_borders('n_conv_layers')[1]
        n_convs = np.random.randint(borders[0], borders[1])
        config['n_conv_layers'] = within_borders(n_convs, get_borders('n_conv_layers')[1])

        # determine number filters by intermediate recombination of parents with flooring the value
        # uniform sample value within borders if not available
        for i in range(n_convs):
            borders = get_borders('n_channels_conv_' + str(i))[1]
            filter_size = np.random.randint(borders[0], borders[1])
            config['n_channels_conv_' + str(i)] = within_borders(filter_size,
                                                                 get_borders('n_channels_conv_' + str(i))[1])

        # determine number of fc layers of child by uniform crossover
        borders = get_borders('n_fc_layers')[1]
        n_fcs = np.random.randint(borders[0], borders[1])
        config['n_fc_layers'] = n_fcs

        # determine number channels by intermediate recombination of parents with flooring the value
        # uniform sample value within borders if not available
        for i in range(n_fcs):
            borders = get_borders('n_channels_fc_' + str(i))[1]
            num_channels = np.random.randint(borders[0], borders[1])
            config['n_channels_fc_' + str(i)] = within_borders(num_channels, get_borders('n_channels_fc_' + str(i))[1])

        # choose optimizer by simulated binary crossover
        borders = get_borders('optimizer')[1]
        opti = np.random.randint(borders[0], borders[1])
        config['optimizer'] = within_borders(opti, get_borders('optimizer')[1])

        # choose criterion by simulated binary crossover
        borders = get_borders('criterion')[1]
        crit = np.random.randint(borders[0], borders[1])
        config['criterion'] = within_borders(crit, get_borders('criterion')[1])

        # choose kernel size as a uniform mutation ( does not depend on parents )
        borders = get_borders('kernel_size')[1]
        config['kernel_size'] = np.random.randint(borders[0], borders[1])

        # choose global average pooling and batch normalization by applying logical AND
        values = get_borders('global_avg_pooling')[1]
        config['global_avg_pooling'] = np.random.choice(values, p=[0.5, 0.5])
        values = get_borders('use_BN')[1]
        config['use_BN'] = np.random.choice(values, p=[0.5, 0.5])

        return config

    def compute_pareto_set(self):
        dominated = self.determine_non_dominated_candidates(self.candidates)
        pareto_set = set()
        for i, c in enumerate(self.candidates):
            if not dominated[i]:
                pareto_set.add(c)
        return pareto_set

    def determine_non_dominated_candidates(self, list_of_candidates, dominated=None):
        if dominated is None:
            dominated = [False for c in list_of_candidates]
        for i, c1 in enumerate(list_of_candidates):
            # if c1 already dominated by some candidate no need to check again
            if not dominated[i]:
                for j, c2 in enumerate(list_of_candidates):
                    if c2.score <= c1.score and c2.size < c1.size:
                       dominated[i] = True
                    elif c2.score < c1.score and c2.size <= c1.size:
                        dominated[i] = True
                    elif c1.score <= c2.score and c1.size < c2.size:
                        dominated[j] = True
                    elif c1.score < c2.score and c1.size <= c2.size:
                        dominated[j] = True
        return dominated

    def plot_pareto_set(self, pareto_set, g):
        """
        Incorporates vital functionality!
        Checks if pareto front stays the same over 3 generations.
        If so attribute to random sample next generation will be activated.
        :param pareto_set:
        :param g:
        :return:
        """
        # plot settings
        sb.set_style('darkgrid')
        sb.set_context("paper", font_scale=1,
                       rc={
                           "grid.linewidth": 2,
                           'axes.labelsize': 10,
                           "axes.titlesize": 10,
                           "legend.fontsize": 10.0,
                           'lines.linewidth': 2,
                           'xtick.labelsize': 10.0,
                           'ytick.labelsize': 10.0,
                       })

        # plotting algorithm results
        # size and score of the base model
        size_b = [np.log10(self.default.size)]
        score_b = [self.default.score]

        # sizes/scores of candidates in the pareto front
        sizes_p = list(map(lambda x: np.log10(x.size), iter(pareto_set)))
        scores_p = list(map(lambda x: x.score, iter(pareto_set)))

        # sizes/scores of all candidates in population
        sizes_a = list(map(lambda x: np.log10(x.size), iter(self.candidates)))
        scores_a = list(map(lambda x: x.score, iter(self.candidates)))

        ref_point = [np.log10(10 ** 8), 0]
        own = list(map(lambda x: x, iter(zip(sizes_p, scores_p))))
        own.sort(key=lambda x: x[0], reverse=True)
        own_HV = self.computeHV2D(own, ref_point)

        # vital functionality
        # increase generation counter if hv stays the same compared to last generation
        # if counter reaches 3 random sample next generation
        if own_HV == self.HV:
            self.generations_since_last_change_pareto += 1
            # perform random search as long as no change in terms of HV
            if self.generations_since_last_change_pareto >= 3:
                self.randomize = True
        else:
            self.generations_since_last_change_pareto = 0
            self.randomize = False

        own.sort(key=lambda x: x[0], reverse=False)
        sizes_p = list(map(lambda x: x[0], own))
        scores_p = list(map(lambda x: x[1], own))

        # plotting baselines
        baseline = np.array([[np.log10(8.80949400e+06), -7.69414740e+01], [np.log10(2.84320000e+04), -5.86384692e+01]])
        plt.scatter(baseline[:, 0], baseline[:, 1], s=100, c='orange')
        plt.plot(baseline[:, 0], baseline[:, 1], c='orange', label='Baseline')

        difandre = np.array([[np.log10(4.27571700e+06), -8.13530869e+01], [np.log10(3.64660000e+04), -8.00280941e+01]])
        plt.scatter(difandre[:, 0], difandre[:, 1], s=100, c='red')
        plt.plot(difandre[:, 0], difandre[:, 1], c='red', label='Difandre')

        # Example known networks
        plt.scatter(np.log10(11.69 * 10 ** 6), -93.87, label='Res18', s=100, marker='X', c='m')
        plt.scatter(np.log10(25.56 * 10 ** 6,), -87.99, label='Res50', s=100, marker='X', c='c')
        plt.scatter(np.log10(44.55 * 10 ** 6), -90.41, label='Res101', s=100, marker='X', c='y')
        plt.scatter(np.log10(61.10 * 10 ** 6), -90.20, label='AlexNet', s=100, marker='X', c='k')

        plt.scatter(sizes_a, scores_a, c='b', alpha=0.5, label='Population')
        plt.plot(sizes_p, scores_p, c='g', alpha=1)
        plt.scatter(sizes_p, scores_p, c='g', alpha=1, label='My Pareto Front')
        plt.scatter(size_b, score_b, c='k', alpha=1, label='Default Config')
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), borderaxespad=0.1)
        plt.title('Pareto Front of Iteration ' + str(g) + ', Pop size: ' + str(len(self.candidates))
                  + ' HV: ' + str(own_HV))
        plt.ylabel('Score')
        plt.xlabel('Number Parameters in Log-Scale')
        name = r'.\src\pareto_fronts\pareto_front_' + str(g)
        plt.tight_layout()
        plt.savefig(fname=name)
        plt.clf()

    def kill_weak_candidates(self):

        population = set(c for c in self.candidates)
        len_pop = len(population)
        new_population = set()
        pop_count = 0
        front = 1
        while (pop_count <= 5 or front <= 2) and pop_count <= 20:
            # keep at least 5 models in the population
            front += 1
            list_pop = list(map(lambda x: x, iter(population)))
            p_front = self.determine_non_dominated_candidates(list_pop)
            for i, v in enumerate(p_front):
                if not(v) and pop_count <= 20:
                    pop_count += 1
                    population.remove(list_pop[i])
                    new_population.add(list_pop[i])
                if pop_count > 20:
                    break
            # if all candidates already in new population
            if pop_count == len_pop:
                break
            # if no new front can be build
            # sample uniformly from the main population
            cont = False
            for c in p_front:
                if c:
                    cont = True
            if not(cont):
                try:
                    rest = np.random.choice(list_pop, min(20 - pop_count, len(list_pop)), replace=False)
                    for c in rest:
                       new_population.add(c)
                    break
                except:
                    break

        new_pop = list(map(lambda x: x, iter(new_population)))
        new_pop.sort(key=lambda x: x.size, reverse=False)
        self.candidates = new_pop
        print(list(map(lambda x: x.id, self.candidates)))

    def computeHV2D(self, front, ref):
        """
        Compute the Hypervolume for the pareto front  (only implement it for 2D)
        :param front: (n_points, m_cost_values) array for which to compute the volume
        :param ref: coordinates of the reference point
        :returns: Hypervolume of the polygon spanned by all points in the front + the reference point
        """
        # We assume all points already sorted
        list_ = [ref]
        for x in front:
            elem_at = len(list_) - 1
            list_.append([list_[elem_at][0], x[1]])  # add intersection points by keeping the x constant
            list_.append(x)
        list_.append([list_[-1][0], list_[0][1]])
        sorted_front = np.array(list_)

        def shoelace(x_y):  # taken from https://stackoverflow.com/a/58515054
            x_y = np.array(x_y)
            x_y = x_y.reshape(-1, 2)

            x = x_y[:, 0]
            y = x_y[:, 1]

            S1 = np.sum(x * np.roll(y, -1))
            S2 = np.sum(y * np.roll(x, -1))

            area = .5 * np.absolute(S1 - S2)

            return area

        return shoelace(sorted_front)
