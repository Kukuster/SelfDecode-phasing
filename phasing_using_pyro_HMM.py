import argparse
import logging
import os
import random
import string
import time
from random import randint
from typing import Any, Callable, Iterable, List, Literal, Tuple, TypeVar, Union
from functools import partial
from collections import Counter
import sys


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
# from pyro.infer.autoguide import AutoDelta
import pyro.distributions as dist
from pyro.infer.autoguide.guides import AutoDelta
from pyro.ops.indexing import Vindex
from pyro.optim import Adam # type: ignore # "Adam" *is* a known import symbol
from pyro.util import ignore_jit_warnings

import torch

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

_sample_dist       = lambda adist: pyro.sample(rand_str(), adist)
_shape_dist        = lambda adist: pyro.sample(rand_str(), adist).shape
_sample_dist_infer = lambda adist: pyro.sample(rand_str(), adist, infer={"enumerate": "parallel"})
_shape_dist_infer  = lambda adist: pyro.sample(rand_str(), adist, infer={"enumerate": "parallel"}).shape

Cate = dist.Categorical # type: ignore # "Categorical" *is* a known member of module dist
Bern = dist.Bernoulli # type: ignore # "Bernoulli" *is* a known member of module dist
Diri = dist.Dirichlet # type: ignore # "Dirichlet" *is* a known member of module dist
Beta = dist.Beta # type: ignore # "Beta" *is* a known member of module dist

eye = torch.eye
sample = pyro.sample
T = torch.Tensor

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


def unphase_batch(batch_of_genotypes: torch.Tensor):
    assert len(batch_of_genotypes.shape) == 2, "batch_of_genotypes has to be a 2-d"
    batch_of_coded_unphased_genotypes = torch.empty(
        size=list(batch_of_genotypes.shape[:-1]) + [1],
        dtype=torch.float
    )

    if batch_of_genotypes.shape[-1] != 2:
        raise ValueError('the last dimension of a tensor should be 2: two calls at a site on a diploid genome')

    batch_of_genotypes_lists: List[List[int]] = batch_of_genotypes.tolist()
    for i in range(len(batch_of_genotypes_lists)):
        genotype: Tuple[Literal[0,1],Literal[0,1]] = tuple(batch_of_genotypes_lists[i]) # type: ignore
        batch_of_coded_unphased_genotypes[i] = unphasing_dict[genotype] # type: ignore
        assert len(genotype) == 2 and genotype[0] in {0,1} and genotype[1] in {0,1}, f"Got wrong genotype data: {genotype}"
    return batch_of_coded_unphased_genotypes.squeeze()

def unphase(genotype: torch.Tensor):
    assert len(genotype) == 2 and len(genotype.shape) == 1, "genotype has to be a 1-d, and has to have total of 2 elements"
    return torch.as_tensor(unphasing_dict[tuple(genotype.tolist())], dtype=torch.long) # type: ignore

# "Factorial HMM with two hidden states."
#
#    w[t-1] ----> w[t] ---> w[t+1]
#        \ x[t-1] --\-> x[t] --\-> x[t+1]
#         \  /       \  /       \  /
#          \/         \/         \/
#        y[t-1]      y[t]      y[t+1]
#
# IN PROGRESS
def model_3(sequences, lengths, args, batch_size=None, include_prior=True, training:bool=True):
    with ignore_jit_warnings():
        num_sequences, max_length, _ = map(int, sequences.shape)
        hidden_variables = 4 # 0 for missing genotype, 1, 2, and 3 for genotypes defined in unphasing_dict
        assert lengths.shape == (num_sequences,)
        assert lengths.max() <= max_length
    hidden_dim = int(args.hidden_dim ** 0.5) # split between w and x
    hidden_dim = 2 # ref and alt
    with poutine.mask(mask=include_prior):
        probs_w = pyro.sample(
            "probs_w", dist.Beta(0.9, 0.1).expand([2]).to_event(1) # type: ignore # "Beta" *is* a known member of module dist
        )
        probs_x = pyro.sample(
            "probs_x", dist.Beta(0.9, 0.1).expand([2]).to_event(1) # type: ignore # "Beta" *is* a known member of module dist
        )
        probs_y = pyro.sample(
            "probs_y",
            dist.Beta(0.1, 0.9).expand([hidden_dim, hidden_dim, hidden_variables]).to_event(3), # type: ignore # "Beta" *is* a known member of module dist
        )

    w_list = []
    x_list = []
    y_list = []

    def train_nobatch(sequences, lengths):
        # assert sequences should be of shape: (n, l, 2)
        for seq in pyro.plate("sequences", len(sequences), batch_size):
            # length = lengths[seq]
            length = lengths[seq]
            sequence = sequences[seq, :length]
            w, x = 0, 0
            for t in pyro.markov(range(lengths.max())): # type: ignore
                with poutine.mask(mask=(t < lengths).unsqueeze(-1)):
                    w = pyro.sample(
                        "w_{}_{}".format(seq, t),
                        dist.Bernoulli(probs_w[w]), # type: ignore
                        infer={"enumerate": "parallel"},
                        obs=sequences[seq, t, 0]
                    ).to(torch.long)
                    w_list.append(w)
                    x = pyro.sample(
                        "x_{}_{}".format(seq, t),
                        dist.Bernoulli(probs_x[x]), # type: ignore
                        infer={"enumerate": "parallel"},
                        obs=sequences[seq, t, 1]
                    ).to(torch.long)
                    x_list.append(x)
                    y = pyro.sample(
                        "y_{}_{}".format(seq, t),
                        dist.Categorical(probs_y[w, x]), # type: ignore
                        obs=unphase(sequences[seq, t]),
                    ).to(torch.long)
                    y_list.append(y)


    def train(sequences, lengths):
        # assert sequences should be of shape: (n, l, 2)
        with pyro.plate("sequences", num_sequences, batch_size, dim=-2) as batch:
            lengths = lengths[batch]
            w, x = 0, 0
            for t in pyro.markov(range(lengths.max())): # type: ignore
                with poutine.mask(mask=(t < lengths).unsqueeze(-1)):
                    w = pyro.sample(
                        "w_{}".format(t),
                        dist.Bernoulli(probs_w[w]), # type: ignore
                        infer={"enumerate": "parallel"},
                        obs=sequences[batch, t, 0]
                    ).to(torch.long)
                    w_list.append(w)
                    x = pyro.sample(
                        "x_{}".format(t),
                        dist.Bernoulli(probs_x[x]), # type: ignore
                        infer={"enumerate": "parallel"},
                        obs=sequences[batch, t, 1]
                    ).to(torch.long)
                    x_list.append(x)
                    y = pyro.sample(
                        "y_{}".format(t),
                        dist.Categorical(probs_y[w, x]), # type: ignore
                        obs=unphase_batch(sequences[batch, t]),
                    ).to(torch.long)
                    y_list.append(y)

    def phase(sequences, lengths):
        pass
        # # assert sequences should be of shape: (n, l, 1)
        # with pyro.plate("sequences", num_sequences, batch_size, dim=-2) as batch:
        #     lengths = lengths[batch]
        #     w, x = 0, 0
        #     for t in pyro.markov(range(lengths.max())):  # type: ignore
        #         with poutine.mask(mask=(t < lengths).unsqueeze(-1)):
        #             w = pyro.sample(
        #                 "w_{}".format(t),
        #                 dist.Categorical(probs_w[w]), # type: ignore
        #                 infer={"enumerate": "parallel"},
        #             ).to(torch.long)
        #             w_list.append(w)
        #             x = pyro.sample(
        #                 "x_{}".format(t),
        #                 dist.Categorical(probs_x[x]), # type: ignore
        #                 infer={"enumerate": "parallel"},
        #             ).to(torch.long)
        #             x_list.append(x)
        #             y = pyro.sample(
        #                 "y_{}".format(t),
        #                 dist.Bernoulli(probs_y[w, x]), # type: ignore
        #                 obs=sequences[batch, t],
        #             ).to(torch.long)
        #             y_list.append(y)

    if training:
        train(sequences, lengths)
    else:
        phase(sequences, lengths)

    return w_list, x_list, y_list



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
# kukedits:
# made work for a reduced shape of the data: for only 1 tone out of 88
def model_31(sequences, lengths, args, batch_size=None, include_prior=True, training:bool=True):

    # corresponds to number of possible indices for probs_w and probs_x
    #         and to number of possible values of w and x vars
    hidden_dim = 2   # for both w and x can be 0 and 1

    observed_dim = 4 # can be 0 (unknown), 1 (0/0), 2 (0/1), 3 (1/1)
    with poutine.mask(mask=include_prior):
        probs_w = pyro.sample(
            "probs_w", dist.Dirichlet(0.9 * torch.eye(hidden_dim) + 0.1).to_event(1) # type: ignore # "Dirichlet" *is* a known member of module dist
        )
        probs_x = pyro.sample(
            "probs_x", dist.Dirichlet(0.9 * torch.eye(hidden_dim) + 0.1).to_event(1) # type: ignore # "Dirichlet" *is* a known member of module dist
        )
        probs_y = pyro.sample(
            "probs_y",
            dist.Beta(0.1, 0.9).expand([hidden_dim, hidden_dim, observed_dim]).to_event(3), # type: ignore # "Beta" *is* a known member of module dist
        )

    w_list = []
    x_list = []
    y_list = []
    batches = []

    def train_on(full_sequences, lengths):
        assert len(sequences.shape) == 3
        with ignore_jit_warnings():
            num_sequences, max_length, num_calls = map(int, sequences.shape)
            assert num_calls == 2, "there should be 2 calls per each site"
            assert lengths.shape == (num_sequences,)
            assert lengths.max() <= max_length

        with pyro.plate("sequences", num_sequences, batch_size, dim=-1) as batch:
            batches.append(batch)
            lengths = lengths[batch]
            w, x = torch.as_tensor(0).to(torch.long), torch.as_tensor(0).to(torch.long)
            for t in pyro.markov(range(max_length if args.jit else lengths.max())): # type: ignore # "Dirichlet" *is* a known member of module dist
                with poutine.mask(mask=(t < lengths)):
                    w = pyro.sample(
                        "w_{}".format(t),
                        dist.Categorical(probs_w[w]), # type: ignore # "Categorical" *is* a known member of module dist
                        infer={"enumerate": "parallel"},
                        obs=full_sequences[batch, t, 0],
                    )
                    w_list.append(w)
                    x = pyro.sample(
                        "x_{}".format(t),
                        dist.Categorical(probs_x[x]), # type: ignore # "Categorical" *is* a known member of module dist
                        infer={"enumerate": "parallel"},
                        obs=full_sequences[batch, t, 1],
                    )
                    x_list.append(x)
                    y = pyro.sample(
                        "y_{}".format(t),
                        dist.Categorical(probs_y[w, x]), # type: ignore # "Categorical" *is* a known member of module dist
                        obs=unphase_batch(full_sequences[batch, t]),
                    )
                    y_list.append(y)

    def phase(sparse_sequences, lengths):
        assert len(sequences.shape) == 2
        with ignore_jit_warnings():
            num_sequences, max_length = map(int, sequences.shape)
            assert lengths.shape == (num_sequences,)
            assert lengths.max() <= max_length

        # with pyro.plate("sequences", num_sequences, batch_size, dim=-1) as batch:
        #     batches.append(batch)
        #     lengths = lengths[batch]
        #     w, x = torch.as_tensor(0).to(torch.long), torch.as_tensor(0).to(torch.long)
        #     for t in pyro.markov(range(max_length if args.jit else lengths.max())): # type: ignore # "Dirichlet" *is* a known member of module dist
        #         with poutine.mask(mask=(t < lengths)):
        #             w = pyro.sample(
        #                 "w_{}".format(t),
        #                 dist.Categorical(probs_w[w]), # type: ignore # "Categorical" *is* a known member of module dist
        #                 infer={"enumerate": "parallel"},
        #             )
        #             w_list.append(w)
        #             x = pyro.sample(
        #                 "x_{}".format(t),
        #                 dist.Categorical(probs_x[x]), # type: ignore # "Categorical" *is* a known member of module dist
        #                 infer={"enumerate": "parallel"},
        #             )
        #             x_list.append(x)
        #             y = pyro.sample(
        #                 "y_{}".format(t),
        #                 dist.Categorical(probs_y[w, x]), # type: ignore # "Categorical" *is* a known member of module dist
        #                 obs=sparse_sequences[batch, t],
        #             )
        #             y_list.append(y)

    if training:
        train_on(sequences, lengths)
    else:
        phase(sequences, lengths)

    return w_list, x_list, y_list, batches





def leave_only_first_played_note(sequences, present_notes):
    sequences = sequences[..., present_notes]
    sequences = sequences[..., 0] # leave only one played note
    return sequences



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

    print(model)

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
        test_samples = 16,
        train_samples = 64,
        sequence_length = 1000
    )
    data_train = data["train"]
    sequences = torch.squeeze(data_train["sequences"])
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

    logging.info(f'sequences.shape: {sequences.shape}')
    logging.info(sequences.shape)
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
        svi = SVI(tmc_model, guide, optim, elbo)
    else:
        Elbo = JitTraceEnum_ELBO if args.jit else TraceEnum_ELBO
        elbo = Elbo(
            max_plate_nesting=2,
            strict_enumeration_warning=True, #strict_enumeration_warning=(model is not model_7),
            jit_options={"time_compilation": args.time_compilation},
        )
        svi = SVI(model, guide, optim, elbo)

    # We'll train on small minibatches.
    logging.info("Step\tLoss")
    for step in range(args.num_steps):
        loss = svi.step(sequences, lengths, args=args, batch_size=args.batch_size)
        logging.info("{: >5d}\t{}".format(step, loss / num_observations)) # type: ignore

    if args.jit and args.time_compilation:
        logging.debug(
            "time to compile: {} s.".format(elbo._differentiable_loss.compile_time) # type: ignore
        )

    # We evaluate on the entire training dataset,
    # excluding the prior term so our results are comparable across models.
    train_loss = elbo.loss(model, guide, sequences, lengths, args, include_prior=False)
    logging.info("training loss = {}".format(train_loss / num_observations))

    # Finally we evaluate on the valid dataset.
    logging.info("-" * 40)
    logging.info(
        "Evaluating on {} valid sequences".format(len(data["valid"]["sequences"]))
    )
    sequences = torch.squeeze(data["train"]["sequences"])
    lengths = torch.squeeze(data["train"]["sequence_lengths"])
    # sequences = leave_only_first_played_note(sequences, present_notes)
    if args.truncate:
        lengths = lengths.clamp(max=args.truncate)
    num_observations = float(lengths.sum())

    # note that since we removed unseen notes above (to make the problem a bit easier and for
    # numerical stability) this valid loss may not be directly comparable to numbers
    # reported on this dataset elsewhere.
    valid_loss = elbo.loss(
        model, guide, sequences, lengths, args=args, include_prior=False
    )
    logging.info("valid loss = {}".format(valid_loss / num_observations))

    # We expect models with higher capacity to perform better,
    # but eventually overfit to the training set.
    capacity = sum(
        value.reshape(-1).size(0) for value in pyro.get_param_store().values()
    )
    logging.info("{} capacity = {} parameters".format(model.__name__, capacity))


    # sequences = data["test"]["sequences"]
    # lengths = data["test"]["sequence_lengths"]
    # xs = pyro.infer.discrete.infer_discrete(model, temperature=0, first_available_dim=-3) # type: ignore
    # print("xs", xs)
    # # print("xs.shape", xs.shape)
    # # xs = pyro.infer.discrete.infer_discrete(model_8, temperature=0, first_available_dim=-3)
    # trace = poutine.trace(xs).get_trace(sequences,lengths, args=args, batch_size=args.batch_size, training=False) # type: ignore
    # print(type(trace))
    # print(trace.nodes)
    # # print(trace.nodes.shape)




if __name__ == "__main__":
    assert pyro.__version__.startswith("1.7.0")
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
    parser.add_argument("-n", "--num-steps", default=10, type=int)
    parser.add_argument("-b", "--batch-size", default=8, type=int)
    parser.add_argument("-d", "--hidden-dim", default=2, type=int)
    parser.add_argument("-nn", "--nn-dim", default=48, type=int)
    parser.add_argument("-nc", "--nn-channels", default=2, type=int)
    parser.add_argument("-lr", "--learning-rate", default=0.05, type=float)
    parser.add_argument("-t", "--truncate", type=int)
    parser.add_argument("-p", "--print-shapes", action="store_true")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--jit", action="store_true")
    parser.add_argument("--time-compilation", action="store_true")
    parser.add_argument("-rp", "--raftery-parameterization", action="store_true")
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
