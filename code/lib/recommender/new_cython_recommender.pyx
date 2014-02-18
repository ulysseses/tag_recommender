from __future__ import division
from collections import Counter
import pickle, csv, time, copy
from itertools import izip, islice
import os.path
from multiprocessing import Manager, Process, Queue

from lib.iterators import row_stream


def basic_recommender(bytes title, usefulness):
    return Counter({token: usefulness.get(token, 0) for token in title.split()})

def tag2tag_recommender(tags, int n_recs=10):
    ''' Given a Counter {tags: scores}, generate and score associated tags. '''
    total_scores = Counter()
    
    # prune tokens with usefulness == 0
    tags -= Counter()
    
    cdef bytes tag, tag2
    cdef double use, co_count
    for tag,use in tags.iteritems():
        tag_scores = Counter({tag2: use*co_count/tag_occurrence[tag]
                              for tag2, co_count in tags_co_occurrence[tag].iteritems()})
        tag_scores = dict(tag_scores.most_common(n_recs)) # keep best n_recs
        combine_tags(total_scores, tag_scores)
    
    return total_scores

def combine_tags(c1, c2):
    """
        Probabilistic union of two tag sets.
        c1, c2: Counters with tags as key, probs as the values
        
        Example:

            c1 = Counter({'php': 0.2, 'xml': 0.14})
            c2 = Counter({'php': 0.1, 'jquery': 0.1})
            combine_tags(c1,c2)
    """
    cdef bytes t
    for t in c1:
        if t in c2:
            c1[t] = 1.0 - (1.0 - c1[t])*(1.0 - c2[t])
    
    for t in c2:
        if t not in c1:
            c1[t] = c2[t] # add missing items from c2
    
def title2tag_recommender(tokens, int n_recs=10):
    total_scores = Counter()
    cdef bytes token, tag
    cdef double co_count
    for token in tokens:
        tag_scores = Counter({tag: co_count / title_occurrence[token]
                              for tag, co_count in title_tag_co_occurrence[token].iteritems()})
        tag_scores = dict(tag_scores.most_common(n_recs))
        combine_tags(total_scores, tag_scores)
    return total_scores - Counter()

def normalize(rec, double weight):
    ''' Normalize each recommendation proportionally to the score of the leading tag. '''
    cdef double max_score
    cdef bytes tag
    if rec:
        max_score = rec.most_common(1)[0][1]
        for tag in rec:
            rec[tag] = weight * rec[tag] / max_score
    return rec

def meta_recommender(recommendations, weights):
    ''' Probabilistically combine `recommendations` with corresponding `weights`. '''
    # total = Counter()
    # cdef double weight
    # try:
    #     for rec, weight in izip(recommendations, weights):
    #         rec = normalize(rec, weight)
    #         combine_tags(total, rec) # accumulate tag scores
    # except:
    #     raise Exception("\n\nbasic: %s\n\ntag2tag: %s\n\ntitle2tag: %s\n\nweights: %s\n\n" % (recommendations[0], recommendations[1], recommendations[2], weights))
    # return total
    total = Counter()
    cdef double weight
    for rec, weight in izip(recommendations, weights):
        rec = normalize(rec, weight)
        combine_tags(total, rec) # accumulate tag scores
    return total

def select_tags(rec, double threshold=0.3):
    ''' Select at most 5 tags with score greather than `threshold`. '''
    cdef bytes tag
    cdef double score
    return [tag for tag, score in rec.most_common(5) if score > threshold]

def benchmark_time(f):
    def new_f(*args, **kwargs):
        t0 = time.time()
        f(*args, **kwargs)
        t1 = time.time()
        print "%s call time elapsed:" % f.__name__, t1 - t0, "seconds"
    return new_f    

def create_proxy_dict_of_dicts(proxy_dict, local_dict):
    ''' Copy a local_dict of dicts into proxy_dict. '''
    for k1, subdict in local_dict.iteritems():
        shared_subdict = manager.dict(subdict.items())
        proxy_dict[k1] = shared_subdict

@benchmark_time
def manage_dicts(work_path):
    ''' Initialize the shared global dictionaries and their `Manager` object. '''
    # load from previous state
    global working_path
    working_path = os.path.expanduser(work_path)

    # initialize global shared variables
    global usefulness, tag_occurrence, tags_co_occurrence, title_occurrence, title_tag_co_occurrence, manager
    manager = Manager()
    with open(os.path.join(working_path, 'usefulness.pickle'), 'rb') as picklefile:
        usefulness = manager.dict(pickle.load(picklefile).items())
    with open(os.path.join(working_path, 'tag_occurrence.pickle'), 'rb') as picklefile:
        tag_occurrence = manager.dict(pickle.load(picklefile).items())
    with open(os.path.join(working_path, 'tags_co_occurrence.pickle'), 'rb') as picklefile:
        tags_co_occurrence = manager.dict()
        create_proxy_dict_of_dicts(tags_co_occurrence, pickle.load(picklefile))
    with open(os.path.join(working_path, 'title_occurrence.pickle'), 'rb') as picklefile:
        title_occurrence = manager.dict(pickle.load(picklefile).items())
    with open(os.path.join(working_path, 'title_tag_co_occurrence.pickle'), 'rb') as picklefile:
        title_tag_co_occurrence = manager.dict()
        create_proxy_dict_of_dicts(title_tag_co_occurrence, pickle.load(picklefile))

def mp_boiler(*args, **kwargs):
    '''
    Multiprocessing Boiler Plate Python-2.7

    args:
        weights
        threshold
    kwargs:
        calculate_pn's kwargs:
            out_of_core_flag
            num_eng
            engine_start
            start
            n
        verbose
    '''
    manage_dicts("../working")
    # Set defaults for calculate_pn's kwargs, `out_of_core_flag`, and `verbose`
    pn_kwargs = {}
    pn_kwargs['out_of_core_flag'] = kwargs['out_of_core_flag'] \
        if type(kwargs.get('out_of_core_flag')) == bool else False
    pn_kwargs['num_eng'] = kwargs['num_eng'] if kwargs.get('num_eng') else 1
    pn_kwargs['engine_start'] = kwargs['engine_start'] if kwargs.get('engine_start') else 0
    pn_kwargs['start'] = kwargs['start'] if kwargs.get('start') else None
    pn_kwargs['n'] = kwargs['n'] if kwargs.get('n') else None

    if not kwargs.get('verbose'): kwargs['verbose'] = True

    # Boilerplate
    f1_queue = Queue()
    weights, threshold = args
    #print "DEBUG\targs:", args
    #print "DEBUG\tkwargs:", kwargs
    procs = [Process(target=calc_wrapper(eid, pn_kwargs['out_of_core_flag']),
                args=(f1_queue, weights, threshold,), kwargs=copy.deepcopy(pn_kwargs.items()),
                name='Process-%d' % eid)
                for eid in range(pn_kwargs['num_eng'])]
    for proc in procs: proc.start()
    for proc in procs: proc.join()

    # Gather the p's & n's from all processes
    f1s = [f1_queue.get() for eid in range(pn_kwargs['num_eng'])] # [(tp1, fp1, fn1), (tp2, fp2, fn2), ...]
    tp, fp, fn = 0, 0, 0
    for tp_part, fp_part, fn_part in f1s:
        tp += tp_part
        fp += fp_part
        fn += fn_part
    f1 = calculate_f1_score(tp, fp, fn, verbose=kwargs['verbose'])
    return f1

def calc_wrapper(eid, out_of_core_flag=False):
    ''' Not used as a decorator, `calc_wrapper` is used in the context of Multiprocessing. '''
    def inner(*args, **kwargs):
        if not out_of_core_flag:
            titles_lst_part_path = os.path.join(working_path, 'titles_lst_%d.pickle' % eid)
            tags_lst_part_path = os.path.join(working_path, 'tags_lst_%d.pickle' % eid)
            if os.path.exists(titles_lst_part_path) and os.path.exists(tags_lst_part_path):
                with open(titles_lst_part_path, 'rb') as picklefile:
                    kwargs['titles_lst'] = pickle.load(picklefile)
                with open(tags_lst_part_path, 'rb') as picklefile:
                    kwargs['tags_lst'] = pickle.load(picklefile)
            else:
                with open(os.path.join(working_path, 'titles_lst.pickle'), 'rb') as picklefile:
                    titles_lst = pickle.load(picklefile)
                with open(os.path.join(working_path, 'tags_lst.pickle'), 'rb') as picklefile:
                    tags_lst = pickle.load(picklefile)
                kwargs['titles_lst'] = titles_lst[eid*len(titles_lst)//2 : (eid+1)*len(titles_lst)//2]
                del titles_lst
                kwargs['tags_lst'] = tags_lst[eid*len(tags_lst)//2 : (eid+1)*len(tags_lst)//2]
                del tags_lst
            #print "DEBUG %d\tlen(kwargs['titles_lst']):" % eid, len(kwargs['titles_lst'])
            return calculate_pn(*args, **kwargs)
        else:
            kwargs['engine_start'] = eid
            return calculate_pn(*args, **kwargs)

    return inner

def calculate_pn(q, weights, threshold, out_of_core_flag=False, titles_lst=None, tags_lst=None, num_eng=1, engine_start=0,
                       start=None, n=None):
    ''' Given weights and threshold for the meta-recommender, calculate
        the positives and negatives (tp, fp, fn) of a (sub)-set of training examples '''

    cdef int tp, fp, fn
    tp, fp, fn = 0, 0, 0
    cdef double p, r, f1

    n = n if n else len(titles_lst)
    start = 0 if not start else start

    cdef bytes ID, title_str, body, tags_str, tag

    if not out_of_core_flag:
        #print "DEBUG\tlen(titles_lst):", len(titles_lst)
        #print "DEBUG\ttitles_lst[:5]:", titles_lst[:5]
        for title_str, tags_str in islice(izip(titles_lst, tags_lst), start, start+n):
            basic = basic_recommender(title_str, usefulness)
            tag2tag = tag2tag_recommender(basic, 10)
            title2tag = title2tag_recommender(basic, 10)
            #print "\nDEBUG\tbasic:", basic, '\n'
            #print "\nDEBUG\ttag2tag:", tag2tag, '\n'
            #print "\nDEBUG\ttitle2tag:", title2tag, '\n'
            recommendations = meta_recommender([basic, tag2tag, title2tag],
                                               weights)
            #print "\nDEBUG\trecommendations:", recommendations, '\n'
            selections = select_tags(recommendations, threshold=threshold)
            
            tags_set = set(tags_str.split())
            selections_set = set(selections)
            tp += len(selections_set & tags_set)
            fp += len(selections_set - tags_set)
            fn += len(tags_set - selections_set)
    else:
        for eid in xrange(engine_start, num_eng):
            for row in row_stream("../data/pruned_Train_%d.csv" % eid, start, start+n):
                ID, title_str, body, tags_str = row
                basic = basic_recommender(title_str, usefulness)
                tag2tag = tag2tag_recommender(basic, 10)
                title2tag = title2tag_recommender(basic, 10)
                recommendations = meta_recommender([basic - Counter(), tag2tag, title2tag],
                                                   weights)
                selections = select_tags(recommendations, threshold=threshold)
                
                tags_set = set(tags_str.split())
                selections_set = set(selections)
                tp += len(selections_set & tags_set)
                fp += len(selections_set - tags_set)
                fn += len(tags_set - selections_set)

    #return (tp, fp, fn)
    q.put((tp, fp, fn))

def calculate_f1_score(*args, verbose=False):
    '''
    Given tp, fp, & fn, calculate p & r to then calculate f1.

    Usage:

        tp, fp, fn = calculate_pn(...)
        print calculate_f1_score(tp, fp, fn) # prints `f1`
        ## OR ##
        print calculate_f1_score(calculate_pn(...)) # prints `f1`
    '''
    tp, fp, fn = args
    p = tp / (tp + fp)
    r = tp / (tp + fn)
    if p == r == 0:
        f1 = 0
    else:
        f1 = 2*p*r/(p+r)

    if verbose:
        print "tp:", tp
        print "fp:", fp
        print "fn:", fn
        print "p:", p
        print "r:", r
        print "f1:", f1

    return f1


if __name__ == '__main__':
    pass