import argparse
import logging
import os
import random
import string
import time
from random import randint
from typing import Any, Callable, Iterable, List, Literal, Tuple, TypeVar, Union, Callable
from functools import partial
from collections import Counter
import sys
import time


# from jax import random
# import jax.numpy as jnp
# import numpy as np

# import numpyro
# from numpyro.contrib.control_flow import scan
# from numpyro.contrib.indexing import Vindex
# import numpyro.distributions as dist
# from numpyro.examples.datasets import JSB_CHORALES, load_dataset
# from numpyro.handlers import mask
# from numpyro.infer import HMC, MCMC, NUTS
# # from pyro.infer import HMC, MCMC, NUTS
# # from pyro.infer import HMC as pyro_HMC, MCMC as pyro_MCMC, NUTS as pyro_NUTS
# from numpyro.infer.util import Predictive

import pyro
import pyro.contrib.examples.polyphonic_data_loader as poly
# import pyro.distributions as dist # type: ignore
from pyro import poutine
from pyro.infer import SVI, JitTraceEnum_ELBO, TraceEnum_ELBO, TraceTMC_ELBO
from pyro.infer import HMC, MCMC, NUTS
# from pyro.infer.autoguide import AutoDelta
import pyro.distributions as dist
from pyro.infer.autoguide.guides import AutoDelta
from pyro.infer import Predictive
from pyro.ops.indexing import Vindex
from pyro.optim import Adam # type: ignore # "Adam" *is* a known import symbol
from pyro.util import ignore_jit_warnings

import torch
from torch.nn.utils.rnn import pad_sequence

import numpy as np

# import pandas as pd


from read_genomic_datasets import read_genomic_datasets


logging.basicConfig(format="%(relativeCreated) 9d %(message)s", level=logging.DEBUG)

# logger = logging.getLogger(__name__)
# logger.setLevel(logging.INFO)
log = logging.getLogger(__name__)
debug_handler = logging.StreamHandler(sys.stdout)
debug_handler.setLevel(logging.DEBUG)
debug_handler.addFilter(filter=lambda record: record.levelno <= logging.DEBUG)
log.addHandler(debug_handler)


def rand_str(length:int=10):
    return ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(length))

# _sample_dist       = lambda adist: pyro.sample(rand_str(), adist)
# _shape_dist        = lambda adist: pyro.sample(rand_str(), adist).shape
# _sample_dist_infer = lambda adist: pyro.sample(rand_str(), adist, infer={"enumerate": "parallel"})
# _shape_dist_infer  = lambda adist: pyro.sample(rand_str(), adist, infer={"enumerate": "parallel"}).shape

Cate = dist.Categorical # type: ignore # "Categorical" *is* a known member of module dist
Bern = dist.Bernoulli # type: ignore # "Bernoulli" *is* a known member of module dist
Diri = dist.Dirichlet # type: ignore # "Dirichlet" *is* a known member of module dist
Beta = dist.Beta # type: ignore # "Beta" *is* a known member of module dist

eye = torch.eye
# sample = pyro.sample
T = torch.Tensor # tensor constructor

def Eight():
    return T([0,0,0,0,0,0,0,0]).to(torch.long)

def prop_np_array(maybe_arr: Union[torch.Tensor, np.ndarray]):
    if isinstance(maybe_arr, torch.Tensor):
        arr = maybe_arr.cpu().detach().numpy()
    elif isinstance(maybe_arr, np.ndarray):
        arr = maybe_arr
    else:
        raise ValueError("function should receive either np array or torch tensor")
    unique, counts = np.unique(arr, return_counts=True)
    proportions = counts/len(arr)
    return dict(zip(unique, proportions))




unphasing_dict = {
    (0,0): 1,
    (0,1): 2,
    (1,0): 2,
    (1,1): 3
}

# effectively this is the map:
#   (0,0) -->  1
#   (0,1) -->  2
#   (1,0) -->  2
#   (1,1) -->  3
# works on batches of genotypes, e.g.:
# [(0,1), (0,0), (1,0)] -->  [2, 1, 2]
def unphase(genotype: torch.Tensor) -> torch.Tensor:
    return genotype.sum(axis=-1) + 1 # type: ignore # it's actually ok to add Tensor and integer


def validate_full_sequences(sequences, lengths):
    assert len(sequences.shape) == 3
    with ignore_jit_warnings():
        num_sequences, max_length, num_calls_per_site = map(int, sequences.shape)
        assert num_calls_per_site == 2, "there should be 2 calls per each site"
        assert lengths.shape == (num_sequences,)
        assert lengths.max() <= max_length

def validate_unphased_sequences(sequences, lengths):
    assert len(sequences.shape) == 2
    with ignore_jit_warnings():
        num_sequences, max_length = map(int, sequences.shape)
        assert lengths.shape == (num_sequences,)
        assert lengths.max() <= max_length



# Next consider a Factorial HMM with two hidden states.
#
#    w[t-1] ----> w[t] ---> w[t+1]
#        \ x[t-1] --\-> x[t] --\-> x[t+1]
#         \  /       \  /       \  /
#          \/         \/         \/
#        y[t-1]      y[t]      y[t+1]
#
# Note that since the joint distribution of each y[t] depends on two variables,
# those two variables become dependent. Therefore during enumeration, the
# entire joint space of these variables w[t],x[t] needs to be enumerated.
# For that reason, we set the dimension of each to the square root of the
# target hidden dimension.
#
# Note that this is the "FHMM" model in reference [1].
def model_31(sequences, lengths, args, batch_size=None, include_prior=True, mode:Literal["train","validate","phase"]="train"):

    # corresponds to number of possible indices for probs_w and probs_x
    #         and to number of possible values of w and x vars
    hidden_dim = 2   # for both w and x can be 0 and 1
    observed_dim = 4 # can be 0 (unknown), 1 (0/0), 2 (0/1), 3 (1/1)

    with poutine.mask(mask=include_prior):

        """
        Parameters for prior probabilities of finding ref (0) or alt (1) after ref (0) or alt (1).
        Parameters were roughly set in accord to two assumptions:
         - approximate occurence of 1s in sequences is around 7%
         - 1s tend to hang out together.

        In fact, these two variables can be measured before training the model.
        """
        p_0_followedby_0 = 0.95
        p_0_followedby_1 = 0.05
        p_1_followedby_0 = 0.95
        p_1_followedby_1 = 0.05

        w_to_w_probs = torch.Tensor([
            [p_0_followedby_0, p_0_followedby_1],
            [p_1_followedby_0, p_1_followedby_1],
        ])
        x_to_x_probs = torch.Tensor([
            [p_0_followedby_0, p_0_followedby_1],
            [p_1_followedby_0, p_1_followedby_1],
        ])
        assert w_to_w_probs.shape[-1] == hidden_dim
        assert x_to_x_probs.shape[-1] == hidden_dim

        probs_w = pyro.sample(
            "probs_w", dist.Dirichlet(w_to_w_probs).to_event(1) # type: ignore # "Dirichlet" *is* a known member of module dist
        )
        probs_x = pyro.sample(
            "probs_x", dist.Dirichlet(x_to_x_probs).to_event(1) # type: ignore # "Dirichlet" *is* a known member of module dist
        )

        """
        Parameter for prior probability that one of two calls was incorrect
        Literally, prior probability we will observe an unphased genotype that will differ by one call
        from the real underlying genotype. i.e.:
            p_1  = 
                P( '0/1' | '1|1' )  = 
                P( '0/1' | '0|0' )  = 
                P( '0/0' | '1|0' )  = 
                P( '0/0' | '0|1' )  = 
                P( '1/1' | '1|0' )  = 
                P( '1/1' | '0|1' )
        """
        p_1 = 2e-4

        """
        Parameter for prior probability that both two calls were incorrect
        Literally, prior probability we will observe an unphased genotype that will differ by both calls
        from the real underlying genotype. i.e.:
            p_2  = 
                P( '0/0' | '1|1' )  = 
                P( '1/1' | '0|0' )
        """
        p_2 = 2e-5

        """
        Parameter for prior probability that both two calls were correct!
        Literally, prior probability we will observe an unphased genotype that has the same number of alt calls
        as the real underlying genotype. i.e.:
            p_0  = 
                P( '0/0' | '0|0' )  = 
                P( '0/1' | '0|1' )  = 
                P( '1/0' | '1|0' )  = 
                P( '1/1' | '1|1' )
        """
        p_0 = 1 - (2*p_1) - p_2

        """
        Parameter for prior probability that it's a "." (missing call) given a full genotype
        Literally, prior probability we will observe an unphased genotype that has the same number of alt calls
        as the real underlying genotype. i.e.:
            p_m  = 
                P( '.' | '0|0' )  = 
                P( '.' | '0|1' )  = 
                P( '.' | '1|0' )  = 
                P( '.' | '1|1' )
        """
        p_m = 2e-12

        #observation not caring where either of the 0 or 1's came from to give observed 'missing', 0/0', '0/1', '1/1'
        wx_to_y_probs = torch.Tensor([
            [ [p_m, p_0, p_1, p_2,],
              [p_m, p_1, p_0, p_1,] ],
            [ [p_m, p_1, p_0, p_1,],
              [p_m, p_2, p_1, p_0,] ],
        ])
        assert wx_to_y_probs.shape[-1] == observed_dim

        probs_y = pyro.sample(
            "probs_y",
            dist.Dirichlet(wx_to_y_probs).to_event(2), # type: ignore # "Beta" *is* a known member of module dist
        )


    def train_on(full_sequences, lengths):
        validate_full_sequences(full_sequences, lengths)
        num_sequences, max_length, _ = map(int, full_sequences.shape)

        with pyro.plate("sequences", num_sequences, batch_size, dim=-1) as batch:
            lengths = lengths[batch]
            w, x = torch.as_tensor(0).to(torch.long), torch.as_tensor(0).to(torch.long)
            for t in pyro.markov(range(max_length)): # type: ignore # "Dirichlet" *is* a known member of module dist
                with poutine.mask(mask=(t < lengths)):
                    w = pyro.sample(
                        "w_{}".format(t),
                        dist.Categorical(probs_w[w]), # type: ignore # "Categorical" *is* a known member of module dist
                        infer={"enumerate": "parallel"},
                        obs=full_sequences[batch, t, 0],#use obs_mask for these sites
                    )
                    x = pyro.sample(
                        "x_{}".format(t),
                        dist.Categorical(probs_x[x]), # type: ignore # "Categorical" *is* a known member of module dist
                        infer={"enumerate": "parallel"},
                        obs=full_sequences[batch, t, 1],#]use obs_mask for these sites
                    )
                    y = pyro.sample(
                        "y_{}".format(t),
                        dist.Categorical(probs_y[w, x]), # type: ignore # "Categorical" *is* a known member of module dist
                        obs=unphase(full_sequences[batch, t]),
                    )


    def validate(full_sequences, lengths):
        validate_full_sequences(full_sequences, lengths)
        num_sequences, max_length, _ = map(int, full_sequences.shape)

        with pyro.plate("sequences", num_sequences, batch_size, dim=-1) as batch:
            lengths = lengths[batch]
            w, x = torch.as_tensor(0).to(torch.long), torch.as_tensor(0).to(torch.long)
            for t in pyro.markov(range(max_length)): # type: ignore # "Dirichlet" *is* a known member of module dist
                with poutine.mask(mask=(t < lengths)):
                    w = pyro.sample(
                        "w_{}".format(t),
                        dist.Categorical(probs_w[w]), # type: ignore # "Categorical" *is* a known member of module dist
                        infer={"enumerate": "parallel"},
                        # obs=full_sequences[batch, t, 0],
                    )
                    x = pyro.sample(
                        "x_{}".format(t),
                        dist.Categorical(probs_x[x]), # type: ignore # "Categorical" *is* a known member of module dist
                        infer={"enumerate": "parallel"},
                        # obs=full_sequences[batch, t, 1],
                    )
                    y = pyro.sample(
                        "y_{}".format(t),
                        dist.Categorical(probs_y[w, x]), # type: ignore # "Categorical" *is* a known member of module dist
                        obs=unphase(full_sequences[batch, t]),
                    ) #potentially combine this into one function with obs_mask


    def phase(sparse_sequences, lengths):
        validate_unphased_sequences(sparse_sequences, lengths)
        num_sequences, max_length = map(int, sparse_sequences.shape)

        with pyro.plate("sequences", num_sequences, batch_size, dim=-1) as batch:
            # if batch is not None:
            #     w_seq = np.zeros(shape=(max_length, len(batch)))
            #     x_seq = np.zeros(shape=(max_length, len(batch)))
            #     y_seq = np.zeros(shape=(max_length, len(batch)))
            # else:
            #     w_seq = np.zeros(max_length)
            #     x_seq = np.zeros(max_length)
            #     y_seq = np.zeros(max_length)
            w_seq = []
            x_seq = []
            y_seq = []
            batches = []

            batches.append(batch)
            lengths = lengths[batch]
            w, x = torch.as_tensor(0).to(torch.long), torch.as_tensor(0).to(torch.long)
            for t in pyro.markov(range(max_length)): # type: ignore # "Dirichlet" *is* a known member of module dist
                with poutine.mask(mask=(t < lengths)):
                    w = pyro.sample(
                        "w_{}".format(t),
                        dist.Categorical(probs_w[w]), # type: ignore # "Categorical" *is* a known member of module dist
                        infer={"enumerate": "parallel"},
                    )
                    w_seq.append(w)
                    x = pyro.sample(
                        "x_{}".format(t),
                        dist.Categorical(probs_x[x]), # type: ignore # "Categorical" *is* a known member of module dist
                        infer={"enumerate": "parallel"},
                    )
                    x_seq.append(x)

                    obs_mask = sparse_sequences[batch, t].to(torch.bool)
                    # non-zero (1,2,3) mean present genotype data, so set to 1 to include in obs
                    obs_mask[obs_mask != 0.] = 1
                    # 0 means missing genotype data, so set to 0 to exclude in obs
                    obs_mask[obs_mask == 0.] = 0
                    y = pyro.sample(
                        "y_{}".format(t),
                        dist.Categorical(probs_y[w, x]), # type: ignore # "Categorical" *is* a known member of module dist
                        obs=sparse_sequences[batch, t],
                        obs_mask=obs_mask
                    ).to(torch.float32)
                    y_seq.append(y)

        return probs_y, probs_w, probs_x, w_seq, x_seq, y_seq, batches

##########################################################EDIT########################################################
    if mode == "train":
        return train_on(sequences, lengths)
    elif mode == "validate":
        return validate(sequences, lengths)
    elif mode == "phase":
        return phase(sequences, lengths)
    else:
        return None # ¯\_(ツ)_/¯



def leave_only_first_played_note(sequences, present_notes):
    sequences = sequences[..., present_notes]
    sequences = sequences[..., 0] # leave only one played note
    return sequences




def test_phasing_accuracy_of_batch_by_concordance(unphased_sequences, full_sequences, w_list, x_list, batch):
    batch_accuracy = np.zeros(len(batch)).astype(np.float32)
    batch_counttophase = np.zeros(len(batch)).astype(np.int64)

    
    for i, seq_i in enumerate(batch):
        count_total_tophase = 0
        count_phased_correctly = 0
        
        for site in range(len(full_sequences[0])):
            if np.around(unphased_sequences[seq_i][site].numpy()) == 2: # 2 corresponds to 0/1, the only genotype that needs phasing
                count_total_tophase += 1
                if tuple(full_sequences[seq_i][site].tolist()) == (w_list[i][site], x_list[i][site]):
                    count_phased_correctly += 1

        batch_accuracy[i] = count_phased_correctly/count_total_tophase if count_total_tophase else np.nan
        batch_counttophase[i] = count_total_tophase

    return batch_accuracy, batch_counttophase


######################################################NOT USED#########################################################
def test_imputation_accuracy_of_batch_by_concordance_of_non_ref(unphased_sequences, full_sequences, w_list, x_list, batch):
    batch_accuracy = np.zeros(len(batch)).astype(np.float32)
    batch_counttoimp = np.zeros(len(batch)).astype(np.int64)
    
    for i, seq_i in enumerate(batch):
        count_total_toimp = 0
        count_imped_correctly = 0
        
        for site in range(len(full_sequences[0])):
            genotype_tuple = tuple(full_sequences[seq_i][site].tolist())
            # test only sites where we need to impute and the true genotype has at least 1 alt
            if unphased_sequences[seq_i][site] == 0 and sum(genotype_tuple) > 0:
                count_total_toimp += 1
                if genotype_tuple == (w_list[i][site], x_list[i][site]):
                    count_imped_correctly += 1

        batch_accuracy[i] = count_imped_correctly/count_total_toimp if count_total_toimp else np.nan
        batch_counttoimp[i] = count_total_toimp

    return batch_accuracy, batch_counttoimp
#####################################################################################################################


def test_accuracy(test_fn, unphased_sequences, full_sequences, w_list, x_list, batches):
    accuracy = np.zeros(len(full_sequences)).astype(np.float32)
    count_sitestotest = np.zeros(len(full_sequences)).astype(np.int64)

    for batch in batches:
        batch_accuracy, batch_sitestotest = test_fn(unphased_sequences, full_sequences, w_list, x_list, batch)
        for i, seq_i in enumerate(batch):
            accuracy[seq_i]     = batch_accuracy[i]
            count_sitestotest[seq_i] = batch_sitestotest[i]
    
    return accuracy, count_sitestotest




def sample_MM(
    model,
    sequences,
    lengths,
    args,
    method: Literal['infer_discrete', 'MCMC_NUTS', 'MCMC_HMC',
                    'poutine_trace', 'predictive'] = 'infer_discrete',
    seed = 0,
    guide: Union[Callable, None] = None,
):
    logging.info(f"sampling using {method} method")

    the_length = lengths[0]
    w_list = torch.zeros([the_length, len(sequences)], dtype=torch.int32)
    x_list = torch.zeros([the_length, len(sequences)], dtype=torch.int32)
    y_list = torch.zeros([the_length, len(sequences)], dtype=torch.int32)
    batches = []

    if method == 'infer_discrete':
        probs_y, probs_w, probs_x, w_list, x_list, y_list, batches = pyro.infer.discrete.infer_discrete(  # type: ignore # "infer" *is* a known member of pyro
            model, temperature=0, first_available_dim=-3
        )(
            sequences, lengths, args=args, mode="phase"
        )

        print("probs_y: ", probs_y)
        print("probs_w: ", probs_w)
        print("probs_x: ", probs_x)

        w_list, x_list, y_list = map(torch.stack           , (w_list, x_list, y_list))
        w_list, x_list, y_list = map(lambda tens: tens.T   , (w_list, x_list, y_list)) #### looks fine for infer_discrete

    elif method.startswith('MCMC'):
        raise NotImplementedError()
        kernel = {"MCMC_NUTS": NUTS, "MCMC_HMC": HMC}[method](model)
        # mcmc = MCMC(kernel, num_samples=500)
        # mcmc.run(data)
        # samples = mcmc.get_samples()

        mcmc = MCMC(
            kernel,
            num_samples=1,
        )
        mcmc.run(seed, sequences, lengths, args, mode="phase")

        posterior_samples = mcmc.get_samples()
        # predictive = Predictive(model, posterior_samples=posterior_samples)
        # samples = predictive(rng_key, sequences, lengths, args, training=False)
        return None, None, None, None ###### MCMC shouldn't be used since the model needs continuous parameters for the gradients

    elif method == 'predictive':
        if guide is None: raise ValueError('"predictive" method for sampling a trained HMM requires specifying guide')
        predictive_sample = Predictive(model, guide=guide, num_samples=1)(sequences, lengths, args=args, mode="phase")

        for i in range(the_length):
            w_list[i] = predictive_sample[f'w_{i}']
            x_list[i] = predictive_sample[f'x_{i}']
            y_list[i] = predictive_sample[f'y_{i}']

        batches = [ torch.Tensor([i for i in range(len(sequences))]).to(torch.int32) ]
        w_list, x_list, y_list = w_list.T, x_list.T, y_list.T

    elif method == 'poutine_trace': ### this just runs the model forwards and should not be used for inference
        trace = poutine.trace(model).get_trace(sequences, lengths, args=args, mode="phase") # type: ignore # no, .trace is not None

        for i in range(the_length):
            w_list[i] = trace.nodes[f'w_{i}']['value']
            x_list[i] = trace.nodes[f'x_{i}']['value']
            y_list[i] = trace.nodes[f'y_{i}']['value']

        batches: List[torch.Tensor] = [ trace.nodes['sequences']['value'] ]
        w_list, x_list, y_list = w_list.T, x_list.T, y_list.T

    else: # ¯\_(ツ)_/¯
        raise ValueError('supply a valid method')

    logging.info(f"w_list shape: {w_list.shape}")
    logging.info(f"x_list shape: {x_list.shape}")
    logging.info(f"y_list shape: {y_list.shape}")
    logging.info(f"batches: {batches}")
    return w_list, x_list, y_list, batches


def perform_sanity_checks(
    w_list, x_list, y_list, batches, # generated sequences (model's output)
    test_sequences, # unphased unimputed sequences (model's input)
    valid_sequences, # corresponding correct full sequences
):
    logging.info("-" * 40)

    logging.info(
        "Sanity check I, comparing statistics from valid sequences and generated sequences:"
    )

    logging.info("proportion of alts in generated w_list and x_list:  {}\t{}".format(
        np.mean(w_list.detach().numpy()),
        np.mean(x_list.detach().numpy())
    ))
    logging.info("proportion of alts in corresponding full sequences: {}\t{}".format(
        np.mean(valid_sequences[:,:,0].detach().numpy()),
        np.mean(valid_sequences[:,:,1].detach().numpy())
    ))


    logging.info("")
    logging.info(
        "Sanity check II, checking that relation between x, w, and y holds: unphased(w_i, x_i) == y_i"
    )

    count_correct_wxy = np.zeros(len(test_sequences))
    seq_length = len(test_sequences[0])

    for sample_i in range(len(w_list)):
        w_seq = w_list[sample_i]
        x_seq = x_list[sample_i]
        y_seq = y_list[sample_i]

        wx_seq = pad_sequence([w_seq, x_seq])
        unphased_wx_seq = np.array(unphase(wx_seq))
        count_correct_wxy[sample_i] = np.count_nonzero(unphased_wx_seq == np.array(y_seq).astype(np.int32))

    logging.info("number of matches between sampled y and w,x per each test sample:")
    for sample_i in range(len(w_list)):
        logging.info(f"sample #{sample_i}: {int(count_correct_wxy[sample_i])}/{seq_length} ({100*count_correct_wxy[sample_i]/seq_length}%)")


def perform_accuracy_measurement(
    w_list, x_list, y_list, batches, # generated sequences (model's output)
    test_sequences, # unphased unimputed sequences (model's input)
    valid_sequences, # corresponding correct full sequences
):
    logging.info(
        "Measuring phasing accuracy by concordance"
    )

    accuracy, counttophase = test_accuracy(
        test_phasing_accuracy_of_batch_by_concordance,
        unphased_sequences=torch.squeeze(test_sequences),
        full_sequences=torch.squeeze(valid_sequences),
        w_list=w_list,
        x_list=x_list,
        batches=batches,
    )

    logging.info(f"number of sites to phase for all sequences: {counttophase}")
    logging.info(f"accuracy for all sequences: {accuracy}")

    total_tophase = counttophase.sum()
    average_accuracy = sum([
        accuracy[i]*(counttophase[i]/total_tophase) for i in range(len(accuracy)) if not np.isnan(accuracy[i])
    ])
    logging.info(f"total accuracy: {average_accuracy}") # accuracy measure 1

    logging.info("-" * 40)


    logging.info(
        "Measuring imputation accuracy by concordance (only for sites with at least 1 alt allele)"
    )

    unphased_sequences_arr = valid_sequences.detach().numpy()
    unphased_sequences_arr_summed_genotypes = np.sum(unphased_sequences_arr, axis=-1)
    unphased_sequences_arr_num_of_nonref_genotypes = unphased_sequences_arr_summed_genotypes[unphased_sequences_arr_summed_genotypes>0].shape[0]

    logging.info(
        "total non-ref sites: {} ({}%)".format(
            unphased_sequences_arr_num_of_nonref_genotypes,
            np.around(
                100 * unphased_sequences_arr_num_of_nonref_genotypes/
                    (unphased_sequences_arr.shape[0]*unphased_sequences_arr.shape[1])
            , decimals=4)
        ))

    accuracy, counttoimp = test_accuracy(
        test_imputation_accuracy_of_batch_by_concordance_of_non_ref,
        unphased_sequences=torch.squeeze(test_sequences),
        full_sequences=torch.squeeze(valid_sequences),
        w_list=w_list,
        x_list=x_list,
        batches=batches,
    )

    logging.info(f"number of non-ref sites to impute for all sequences: {counttoimp}")
    logging.info(f"accuracy for all sequences: {accuracy}") # accuracy measure 2

    total_toimp = counttoimp.sum()
    average_accuracy = sum([
        accuracy[i]*(counttoimp[i]/total_toimp) for i in range(len(accuracy)) if not np.isnan(accuracy[i])
    ])
    logging.info(f"total accuracy: {average_accuracy}") # accuracy measure 3




class subsequence(tuple):
    site: int
    sequence: int
    sequence_obj: Union[List, Tuple]


models = {
    name[len("model_") :]: model
    for name, model in globals().items()
    if name.startswith("model_")
}


def main(args):


    model = models[args.model]
    # model = og_model_1

    """for name in pyro.get_param_store().get_all_param_names():
        print(name, pyro.param(name).data.numpy())"""




    # _, fetch = load_dataset(JSB_CHORALES, split="train", shuffle=False)
    # lengths, sequences = fetch()
    # # kukedit:
    # sequences = sequences[0:13]
    # lengths = lengths[0:13]
    # if args.num_sequences:
    #     sequences = sequences[0 : args.num_sequences]
    #     lengths = lengths[0 : args.num_sequences]

    # # find all the notes that are present at least once in the training set
    # present_notes = (sequences == 1).sum(0).sum(0) > 0
    # # remove notes that are never played (we remove 37/88 notes with default args)
    # sequences = sequences[..., present_notes]

    # # kukedit:
    # sequences = sequences[:,:,0:2] # leave only 2 notes

    pyro.set_rng_seed(args.seed)


    logging.info("Loading data")

    data = read_genomic_datasets(
        test_samples = 40,
        train_samples = 320,
        sequence_length = 6000
    )

    # # for this setup, with learning rate 0.05, more than 40 steps doesn't help
    # data = read_genomic_datasets(
    #     test_samples = 8,
    #     train_samples = 256,
    #     sequence_length = 2000
    # )
    # data = read_genomic_datasets(
    #     test_samples = 8,
    #     train_samples = 256,
    #     sequence_length = 1000
    # )
    data_train = data["train"]
    sequences = data_train["sequences"]
    lengths = data_train["sequence_lengths"]
    # sequences = list(sequences)
    # sequences = torch.Tensor(sequences + sequences)
    # lengths = list(lengths)
    # lengths = torch.Tensor(lengths + lengths)

    # data = poly.load_data(poly.JSB_CHORALES)
    # sequences = data["train"]["sequences"]
    # lengths = data["train"]["sequence_lengths"]
    # sequences = sequences[0:13]
    # lengths = lengths[0:13]
    # present_notes = (sequences == 1).sum(0).sum(0) > 0
    # sequences = leave_only_first_played_note(sequences, present_notes)

    logging.info("-" * 40)

    logging.info("Training {} on {} sequences".format(model.__name__, len(sequences)))

    logging.info(f"sequences.shape: {sequences.shape}")
    unphased_sequences_arr = data['test']['sequences'].detach().numpy()
    unphased_sequences_arr_occurances_of_2 = unphased_sequences_arr[unphased_sequences_arr==2].shape[0]
    logging.info(
        f"Total sites to phase: {unphased_sequences_arr_occurances_of_2}"
    )
    logging.info(f"Learning rate: {args.learning_rate}")
    logging.info(f"Seed: {args.seed}")
    # logging.info(sequences)
    # logging.info('lengths: {}'.format(lengths))

    num_observations = float(lengths.sum())
    pyro.set_rng_seed(args.seed)
    pyro.clear_param_store()

    # We'll train using MAP Baum-Welch, i.e. MAP estimation while marginalizing
    # out the hidden state x. This is accomplished via an automatic guide that
    # learns point estimates of all of our conditional probability tables,
    # named probs_*.
    guide = AutoDelta(
        poutine.block(model, expose_fn=lambda msg: msg["name"].startswith("probs_"))
    )

    # # To help debug our tensor shapes, let's print the shape of each site's
    # # distribution, value, and log_prob tensor. Note this information is
    # # automatically printed on most errors inside SVI.
    # if args.print_shapes:
    #     first_available_dim = -3
    #     guide_trace = poutine.trace(guide).get_trace( # type: ignore
    #         sequences, lengths, args=args, batch_size=args.batch_size
    #     )
    #     model_trace = poutine.trace(
    #         poutine.replay(poutine.enum(model, first_available_dim), guide_trace) # type: ignore
    #     ).get_trace(sequences, lengths, args=args, batch_size=args.batch_size)
    #     logging.info(model_trace.format_shapes())

    # Enumeration requires a TraceEnum elbo and declaring the max_plate_nesting.
    # All of our models have two plates: "data" and "tones".
    optim = Adam({"lr": args.learning_rate})
    scheduler = pyro.optim.ExponentialLR({'optimizer': optim, 'optim_args': {'lr': 0.01}, 'gamma': 0.3})
    if args.tmc:
        if args.jit:
            raise NotImplementedError("jit support not yet added for TraceTMC_ELBO")
        elbo = TraceTMC_ELBO(max_plate_nesting=2)
        tmc_model = poutine.infer_config(
            model,
            lambda msg: {"num_samples": args.tmc_num_samples, "expand": False}
            if msg["infer"].get("enumerate", None) == "parallel"
            else {},
        )  # noqa: E501
        svi = SVI(tmc_model, guide, scheduler, elbo)
    else:
        Elbo = TraceEnum_ELBO
        elbo = Elbo(
            max_plate_nesting=2,
            strict_enumeration_warning=True, #strict_enumeration_warning=(model is not model_7),
            jit_options={"time_compilation": args.time_compilation},
        )
        svi = SVI(model, guide, optim, elbo)

    # We'll train on small minibatches.
    logging.info("Step\tLoss\t Time \t  Total time")
    t0 = time.time()
    #####there is a problem where since the batch size is just restricting the amount of data we use from sequences
    for epoch in range(5):
        for step in range(args.num_steps):
            tn = time.time()
            loss = svi.step(sequences, lengths, args=args, batch_size=args.batch_size, mode="train")
            tnp1 = time.time()
            logging.info("{: >5d}\t{}\t {} s \t {} s".format(
                step,
                round(loss/num_observations,5), # type: ignore
                round(tnp1-tn,3),
                round(tnp1-t0,3)
            ))

    ################################EDIT#################################################################
    print("model parameters")
    for name in pyro.get_param_store().get_all_param_names():
        print(name, pyro.param(name).data.numpy())

    print("torch parameters")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print (name," ", param.data)

    # We evaluate on the entire training dataset,
    # excluding the prior term so our results are comparable across models.
    train_loss = elbo.loss(model, guide, sequences, lengths, args, include_prior=False, mode="validate") #this runs from the guide
    logging.info("training loss = {}".format(train_loss / num_observations))

    # Finally we evaluate on the valid dataset.
    logging.info("-" * 40)
    logging.info(
        "Evaluating on {} valid sequences".format(len(data["valid"]["sequences"]))
    )
    valid_sequences = sequences = torch.squeeze(data["valid"]["sequences"])
    valid_lengths = lengths = torch.squeeze(data["valid"]["sequence_lengths"])
    # sequences = leave_only_first_played_note(sequences, present_notes)
    if args.truncate:
        lengths = lengths.clamp(max=args.truncate)
    num_observations = float(lengths.sum())

    # note that since we removed unseen notes above (to make the problem a bit easier and for
    # numerical stability) this valid loss may not be directly comparable to numbers
    # reported on this dataset elsewhere.
    valid_loss = elbo.loss(
        model, guide, sequences, lengths, args=args, include_prior=False, mode="validate"
    )
    logging.info("valid loss = {}".format(valid_loss / num_observations))

    # We expect models with higher capacity to perform better,
    # but eventually overfit to the training set.
    capacity = sum(
        value.reshape(-1).size(0) for value in pyro.get_param_store().values()
    )
    logging.info("{} capacity = {} parameters".format(model.__name__, capacity))


    # Phasing
    logging.info("-" * 40)
    logging.info(
        "Phasing {} test sequences".format(len(data["test"]["sequences"]))
    )

    sequences = torch.squeeze(data["test"]["sequences"])
    lengths = torch.squeeze(data["test"]["sequence_lengths"])


    for i in range(1,9):
        logging.info(("="*20) + f"Phasing test: predictive (#{i})" + ("="*20))
        w_list, x_list, y_list, batches = sample_MM(model, sequences, lengths, args, method='predictive', guide=guide)
        perform_sanity_checks       (w_list, x_list, y_list, batches, sequences, valid_sequences)
        perform_accuracy_measurement(w_list, x_list, y_list, batches, sequences, valid_sequences)

    for i in range(1,9):
        logging.info(("="*20) + f"Phasing test: infer_discrete (#{i})" + ("="*20))
        w_list, x_list, y_list, batches = sample_MM(model, sequences, lengths, args, method='infer_discrete')
        perform_sanity_checks       (w_list, x_list, y_list, batches, sequences, valid_sequences)
        perform_accuracy_measurement(w_list, x_list, y_list, batches, sequences, valid_sequences)

    for i in range(1,9):
        logging.info(("="*20) + f"Phasing test: poutine_trace (#{i})" + ("="*20))
        w_list, x_list, y_list, batches = sample_MM(model, sequences, lengths, args, method='poutine_trace', guide=guide)
        perform_sanity_checks       (w_list, x_list, y_list, batches, sequences, valid_sequences)
        perform_accuracy_measurement(w_list, x_list, y_list, batches, sequences, valid_sequences)



    return None




if __name__ == "__main__":
    #assert pyro.__version__.startswith("1.7.0")
    parser = argparse.ArgumentParser(
        description="MAP Baum-Welch learning Bach Chorales"
    )
    parser.add_argument(
        "-m",
        "--model",
        default="31",
        type=str,
        help="one of: {}".format(", ".join(sorted(models.keys()))),
    )
    parser.add_argument("-n", "--num-steps", default=50, type=int)
    parser.add_argument("-b", "--batch-size", default=8, type=int)
    # parser.add_argument("-d", "--hidden-dim", default=2, type=int)
    parser.add_argument("-nn", "--nn-dim", default=48, type=int)
    parser.add_argument("-nc", "--nn-channels", default=2, type=int)
    parser.add_argument("-lr", "--learning-rate", default=0.04, type=float) #make this smaller as a next attempt
    parser.add_argument("-t", "--truncate", type=int)
    parser.add_argument("-p", "--print-shapes", action="store_true")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--cuda", action="store_true")
    # parser.add_argument("--jit", action="store_true")
    parser.add_argument("--time-compilation", action="store_true")
    parser.add_argument(
        "--tmc",
        action="store_true",
        help="Use Tensor Monte Carlo instead of exact enumeration "
        "to estimate the marginal likelihood. You probably don't want to do this, "
        "except to see that TMC makes Monte Carlo gradient estimation feasible "
        "even with very large numbers of non-reparametrized variables.",
    )
    parser.add_argument("--tmc-num-samples", default=10, type=int)
    args = parser.parse_args()
    main(args)
