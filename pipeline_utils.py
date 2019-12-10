"""
Tidy-up the pipeline.
"""


from analysis.params import ParamRun


fname_params = "params_lensing.yml"
p = ParamRun(fname_params)


# Find which Cls we will be using
twopoints = p.get("data_vectors")[0]["twopoints"]


def unravel_maps(p):
    """Constructs dictionary of tracer types
    containing dictionary of tracer names, and all other attributes as values.
    """
    maps = {}
    for M in p.get("maps"):
        if not maps.get(M["type"]):
            maps[M["type"]] = {}

        maps[M["type"]][M["name"]] = M
    return maps



maps = unravel_maps(p)


def tracer_type(maps, tr):
    """Finds the types of the tracer names given in a list."""
    types = []



def which_cls(p):
    """Returns a list of twopoint cross-correlations pairs."""
    maps = unravel_maps(p)
    twopoints = []
    for tp in p.get("data_vectors")[0]["twopoints"]:
        tr = tp["tracers"]
        twopoints.append(tracer_type(maps, tr))

    return twopoints