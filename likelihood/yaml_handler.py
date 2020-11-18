import yaml


def dict_update(mdict, pars, vals, aliases):
    """Updates values in `maps` dictionary entry."""
    dic = mdict["model"]

    for par, val in zip(pars, vals):
        for param in dic:
            if param == par:
                dic[param] = val

                # handle aliases
                if par in aliases:
                    for param in dic:
                        if param == aliases[par]:
                            dic[param] = val
                break

    mdict["model"] = dic
    return mdict


def update_params(fname_in, name, pars, vals):
    """
    Updates yaml parameter file with specified values.
    Outputs a new file appended by `_bf` containing
    the output of the minimizer.

    Parameters
    ----------
    fname_in : str
        Name of input yaml file.
    name : str
        Name of data vector.
    pars : str or list of strings
        Parameters to be overwritten.
    vals` : float or ``numpy.ndarray``
        Values of parameters.

    Returns
    -------
    None.

    """
    # Input handling
    if type(pars) is not list:
        pars = [pars,]
        vals = [vals,]

    vals = [float(val) for val in vals]
    assert len(pars) == len(vals)

    # Open file
    with open(fname_in) as f:
        doc = yaml.safe_load(f)

    # Grab aliases - format: {Y: X}, Y an alias for X
    aliases = {}
    for param in doc["params"]:
        if param.get("alias") is not None:
            aliases[param["alias"]] = param["name"]

    # Update map parameters
    for par, val in zip(pars, vals):
        for m in doc["maps"]:
            if m["name"] == name:
                m = dict_update(m, pars, vals, aliases)
            break

    # Write to file
    fname_out = "_bf.".join(fname_in.split("."))
    with open(fname_out, "w") as f:
        yaml.safe_dump(doc, f)


def update_nsteps(fname, nsteps):
    """Updates number of MCMC steps in yaml file."""
    with open(fname, "w") as f:
        doc = yaml.safe_load(f)
        doc["mcmc"]["n_steps"] = nsteps
        yaml.safe_dump(doc, f)


#### TEST CODE ####
# # Update global parameters
# for par, val in zip(pars, vals):
#     for param in doc["params"]:
#         if param["name"] == par:
#             param["value"] = val

#             if par in aliases:
#                 for param in doc["params"]:
#                     if param.get("alias") == par:
#                         param["value"] = val
#             break
