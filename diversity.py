import numpy as np

def diversity(name, props, args=None):
    if args is None:
        args = tuple()
    div_func_dict = {"richness": richness, "shannon": shannon, "inv_simpson": inv_simpson, "hill": hill}
    return div_func_dict[name](props, *args)

def richness(props):
    # count species that exist in sample
    return sum(props > 0)

def shannon(props):
    # first remove any sequences with zero share of repertoire
    props = props[props > 0]
    # calculate shannon diversity using formula
    return -sum(props * np.log(props))

def inv_simpson(props):
    # first remove any sequences with zero share of repertoire
    props = props[props > 0]
    # calculate Simpson diversity using formula
    return 1/(sum(props ** 2))

def hill(props, q_vals):
    # calculate Hill diversity profiles using list of q values
    # make sure q_vals is 1D array
    q_vals = np.array(q_vals).reshape(-1)
    # initialise empty list
    div = []
    # remove sequences with zero share of repertoire
    props = props[props > 0]
    for q in q_vals:
        if q == 1:
            # calculate perplexity separately due to log
            ind = np.exp(-np.sum(props * np.log(props)))
        else:
            # general formula works for remaining q values
            ind = np.sum(props ** q) ** (1 / (1 - q))
        div = np.append(div, ind)
    return np.squeeze(div)
