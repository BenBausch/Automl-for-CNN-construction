from itertools import count
import numpy as np

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

        for i in range(number_children):
            if number_children > len(possible_parents):
                return candidate_set

            print('Possible parents by ids:')
            print(list(map(lambda x: x.id, possible_parents)))

            # calculate probs proportional to size of candidate and
            # reverse the list to make small networks more probable
            probs = list(map(lambda x: x.size, possible_parents))
            total_parameters = sum(probs)
            for i in range(len(probs)):
                probs[i] = probs[i] / total_parameters
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

    def produce_child(self, parent1, parent2):
        pass 

    def compute_pareto_front(self):
        pass


if __name__ == "__main__":
    p = Population()
    for i in range(1,11):
        p.add_candidate(Candidate(10*i, i, 0, 0))
    p.sample_by_dist_prop_to_size_then_tournament(4)