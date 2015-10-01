import numpy as np
from sklearn.utils import check_random_state
from scipy.stats import kendalltau, pearsonr

def weighted_pearson_correlation(u, v, w):
            '''
            Calculates the weighted (with weights `w`) pearson correlation between
            u and v
            
            TODO: make faster?
            '''
            # Normalize w so it sums to 1
            if np.sum(w) != 1.:
                w = w / float(np.sum(w))
            
            u = u.flatten()
            v = v.flatten()
            
            mean_u = w.dot(u)
            var_u = w.dot((u - mean_u)**2)
            
            mean_v = w.dot(v)
            var_v = w.dot((v - mean_v)**2)
            
            cov_uv = np.sum((u - mean_u)*(v - mean_v)*(w))
            rho = cov_uv / np.sqrt(var_u) / np.sqrt(var_v)
            
            return rho

class GAOutlierEnsemble(object):
    """Our very own Genetic Algorithm-based Outlier Ensemble!
    
    
    Parameters
    ----------
    n_iter : integer, default: 10
        How long to run the GA
    
    population_size : integer, default 10
        How many individuals to keep in the population every generation.

    mutate_prob : default 0.5
        The probability of an new individual undergoing mutation.
    
    gene_mutate_prob : default 0.15
        The probability of a gene in a chromosome mutating.
        That is, the probability that member in the ensemble will leave, or that a new member will be added.
    
    random_state : int seed, RandomState instance, or None (default)
        A pseudo-random number generator.
    
    Attributes
    ----------
    
    
    References
    ----------
    None!
    """
    
    def __init__(self, n_iter=2, population_size=10, mutate_prob = 0.5, gene_mutate_prob=0.15, random_state=None, top_k=500):
        self.n_iter = n_iter
        self.population_size = population_size
        self.mutate_prob = mutate_prob
        self.gene_mutate_prob = gene_mutate_prob
        self.random_state = random_state
        
        self.top_k = top_k
    
    def run(self, matrix_of_scores):

        self.rs = check_random_state(self.random_state)
        self.matrix_of_scores = matrix_of_scores
        self.n = matrix_of_scores.shape[1]
        self.num_instances = matrix_of_scores.shape[0]
        
        # pseudo ground truth
        # Create "target" vector (NOT used as the actually result)
        top_k_union = np.unique(np.argsort(-matrix_of_scores, axis=0)[:self.top_k])
        target_vec = np.zeros(self.num_instances)
        target_vec[top_k_union] = 1
        K = len(top_k_union)
        
        # Weights for the weighted Pearson correlation
        w = np.empty(self.num_instances)
        w[:] = K
        w[top_k_union] = self.num_instances-K
        self.target_vec = target_vec
        self.target_vec_w = w
        
        
        self.init_population()
        #print np.mean([len(C) for C in self.population]), np.std([len(C) for C in self.population])
        fitness = []
        for i in range(self.n_iter):
            # Calculate fitness for members in population
            fitness = np.array([self.fitness(C) for C in self.population], dtype=float)
            fitness_prob_dist = fitness / fitness.sum()
            # Select parents from population and produce children
            children = []
            for _ in range(self.population_size):
                # choose 2 members of the population
                parent_ind = self.rs.choice(self.population_size,
                                         2,
                                         replace=False,
                                         p=fitness_prob_dist)
                parentA = self.population[parent_ind[0]]
                parentB = self.population[parent_ind[1]]
                child = self.crossover(parentA, parentB)
                assert len(np.unique(child)) == len(child)
                # Mutate child
                if self.rs.binomial(1, p=self.mutate_prob):
                    child = self.mutate(child)
                    assert len(np.unique(child)) == len(child)
                child = child.astype(int)
                children.append(child)
                
            #Update population
            self.population = children    
            #print np.mean([len(C) for C in self.population]), np.std([len(C) for C in self.population])
        
        # Return from the final population the member with the best fitness
        fitness = np.array([self.fitness(C) for C in self.population], dtype=float)
        return self.population[np.argmax(fitness)]
    
    def agreement_measure(self, s1, s2):
        #from scipy.stats import kendalltau
        #return kendalltau(s1, s2)
        return (pearsonr(s1, s2)[0] - (-1)) / 2.
    
    def agreement_measure_other(self, s1, s2):
        '''
        returns the "agreement" between s1 and s2
        weighted pearson correlation
        '''
        
        # indices of s1,s2 in sorted order, smallest scores first
        s1_sortind = np.argsort(s1)
        s2_sortind = np.argsort(s2)
        
        
        w1 = np.arange(1, len(s1)+1)[s1_sortind]
        w2 = np.arange(1, len(s2)+1)[s2_sortind]
    
        # w1 weights the scores in s1 which are higher with higher weights
        # w2 does the same for s2
        
        ans1 = weighted_pearson_correlation(s1, s2, w1)
        ans2 = weighted_pearson_correlation(s1, s2, w2)
        
        # Rescaled to be between 0,10
        ans1 = (ans1 - (-1)) / 2.
        ans2 = (ans2 - (-1)) / 2.
        
        ans = (ans1 + ans2) / 2.
        try:
            assert 0.0 <= ans <= 1.0
        except:
            raise Warning("WTF something wrong with correlation")
        return ans
    
    def kappa_statistic(self):
        raise Exception("Not Implemented Yet!")
        
    def symmetrical_uncertainty(self):
        raise Exception("Not Implemented Yet!")
        
    def fitness(self, chromosome):
        #return np.abs(len(chromosome)-10)
        # average of the full ensemble
        #full_ensemble_mean = np.mean(self.matrix_of_scores, axis=1)
        #ensemble_mean = np.mean(self.matrix_of_scores[:, chromosome], axis=1)
        # use the merit
        n = len(chromosome)
        
        kcm = np.mean([(weighted_pearson_correlation(self.target_vec, self.matrix_of_scores[:, j], self.target_vec_w) - (-1)) / 2. for j in chromosome])
        kmm = []
        for i in range(len(chromosome)):
            for j in range(i+1, len(chromosome)):
                kmm.append(self.agreement_measure(self.matrix_of_scores[:, i], self.matrix_of_scores[:, j]))
        kmm = np.mean(kmm)
        
        merit = n * kcm / np.sqrt(n + n*(n-1)*kmm)
        return merit
#def merit(matrix_of_scores):
#    '''
#    Measure's the "merit" function as seen in Lior's paper
#    using kendall's tau as the measure of agreement
#    '''
#    #ensemble_mean = np.mean(matrix_of_scores, axis=1)
#    # use the merit
#    n = matrix_of_scores.shape[1]
#    
#    kcm = np.mean([weighted_pearson_correlation(matrix_of_scores[:, j], target_vec, target_vec_w) for j in range(n)])
#    kmm = []
#    for i in range(n):
#        for j in range(i+1, n):
#            kmm.append(kendalltau(matrix_of_scores[:, i], matrix_of_scores[:, j]))
#    kmm = np.mean(kmm)
#    
#    merit = n * kcm / np.sqrt(n + n*(n-1)*kmm)
#    return (merit, kcm, kmm)        
#        
    def crossover(self, parent1, parent2):
        in_both = set(parent1).intersection(set(parent2))
        only_in_one = set(parent1).symmetric_difference(set(parent2))
        which_to_take = self.rs.binomial(n=1, p=0.5, size=len(only_in_one)) == 1
        return np.r_[np.array(list(in_both)), np.array(list(only_in_one))[which_to_take]]
    
    def mutate(self, chromosome):
        # Mutate ensemble
        # Some stay in some leave
        # calculate which to come in
        not_in = set(range(self.n)).difference(set(chromosome))
        # members not_in the ensemble which will join
        joining_ind = self.rs.binomial(n=1, p=self.gene_mutate_prob, size=len(not_in)) == 1
        
        new_ens_members = np.array(list(not_in))[joining_ind]
        
        # get which members stay in the ensemble
        staying_ind = self.rs.binomial(n=1, p=1.0-self.gene_mutate_prob, size=len(chromosome)) == 1
        remaining_ens_members = chromosome[staying_ind]
        
        return np.r_[remaining_ens_members, new_ens_members]
        
    def init_population(self):
        self.population = []
        sizes = self.rs.choice(np.arange(2, self.n), self.population_size)
        for i in range(self.population_size):
            self.population.append(self.rs.choice(self.n, sizes[i], replace=False))
        
        
if __name__ == '__main__':
    matrix_of_scores = np.random.rand(3,100)
    ga = GAOutlierEnsemble(n_iter=25)
    ga.run(matrix_of_scores)
