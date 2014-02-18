from __future__ import division
from IPython.parallel import Client


def initialize_client(*args, **kwargs):
    ''' helper function to initialize the IPython Cluster client for multiprocessing '''
    # Make sure `ipcluster start -n 2 &` is executed in the same shell
    # When completed, kill with `ipcluster stop`
    if 'client' not in globals():
        global client, dv
        client = Client()
        dv = client[:]
        dv.block = True
        dv.execute("import pyximport; pyximport.install()")
        dv.execute("from lib.recommender.cython_recommender import *")
        dv.block = False

    for i, target in enumerate(dv.targets):
        dv.client[target]['eid'] = i
    dv['args'] = args
    dv['kwargs'] = kwargs


def calculate_f1_score(tp, fp, fn, verbose=False):
    '''
    Given tp, fp, & fn, calculate p & r to then calculate f1.

    Usage:

        tp, fp, fn = calculate_pn(...)
        print calculate_f1_score(tp, fp, fn) # prints `f1`
    '''
    p = tp / (tp + fp)
    r = tp / (tp + fn)
    if p == r == 0:
        f1 = 0
    else:
        f1 = 2*p*r/(p+r)

    if verbose:
        print "tp:", tp
        print "fp:", fp
                 with open(os.path.join(working_path, 'titles_lst_%d.pickle' % self.eid), 'rb') as picklefile:
                    titles_lst = pickle.load(picklefile)
                with open(os.path.join(working_path, 'tags_lst_%d.pickle' % self.eid), 'rb') as picklefile:
                    tags_lst = pickle.load(picklefile)   print "fn:", fn
        print "p:", p
        print "r:", r
        print "f1:", f1

    return f1

def title_tag_optimize(weights, threshold):
    '''
    WEIGHTS[0] = basic threshold
    WEIGHTS[1] = tag2tag threshold
    WEIGHTS[2] = title2tag threshold
    THRESHOLD = meta_recommender THRESHOLD
    '''

    args = [weights, threshold]
    kwargs = dict(n=10000, random_start=True)
    initialize_client(*args, **kwargs)
    cmd_str = "pn_part = custom_script(eid, *args, **kwargs)"
    dv.execute(cmd_str, block=True)

    tp = sum([item[0] for item in dv.pull('pn_part', block=True)])
    fp = sum([item[1] for item in dv['pn_part']])
    fn = sum([item[2] for item in dv['pn_part']])
    f1 = calculate_f1_score(tp, fp, fn)

    return f1


def main(job_id, params):
    weights = [float(weight) for weight in params['WEIGHTS']]
    threshold = float(params['THRESHOLD'])
    print "weights:", weights
    print "treshold:", threshold
    return title_tag_optimize(weights, threshold)