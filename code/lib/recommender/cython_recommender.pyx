from __future__ import division
from collections import Counter
import pickle, csv, time
from itertools import izip, islice
import os.path
#import numpy
import random

from lib.iterators import row_stream


def basic_recommender(bytes title, usefulness):
    return Counter({token: usefulness.get(token, 0) for token in title.split()}) - Counter()

def tag2tag_recommender(tags, int n_recs=10):
    ''' Given a Counter {tags: scores}, generate and score associated tags. '''
    total_scores = Counter()
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
    
def title2tag_recommender(bytes title_str, int n_recs=10):
    total_scores = Counter()
    cdef bytes token, tag
    cdef double co_count
    for token in title_str.split():
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

class MultiCore:
    ''' MultiCore divides up titles and tags into `num_eng` copies. '''
    def __init__(self, eid, out_of_core=False, num_eng=2):
        self.eid = eid
        self.out_of_core = out_of_core
        self.num_eng = num_eng

    def __enter__(self):
        if not self.out_of_core:
            if 'titles_lst' not in globals():
                global titles_lst, tags_lst
                with open(os.path.join(working_path, 'titles_lst_%d.pickle' % self.eid), 'rb') as picklefile:
                    titles_lst = pickle.load(picklefile)
                with open(os.path.join(working_path, 'tags_lst_%d.pickle' % self.eid), 'rb') as picklefile:
                    tags_lst = pickle.load(picklefile)
        else:
            global num_eng, engine_start
            num_eng, engine_start = self.num_eng, self.eid

        return

    def __exit__(self, type, value, traceback):
        return

def load_dicts():
    ''' Initialize the global dictionaries '''
    global usefulness, tag_occurrence, tags_co_occurrence, title_occurrence, title_tag_co_occurrence
    # load from previous state
    global working_path
    if not os.path.exists("working.path"):
        working_path = os.path.expanduser("~/working/fb_recruit/working")
        if not os.path.exists(working_path):
            print "Default working path not found or is incorrect. Please create/change relative working path" \
                "in lib/recommender/cython_recommender.pyx"
            print "setting working_path to ~/working/fb_recruit/working ..."
            working_path = os.path.expanduser("~/working/fb_recruit/working")
    else:
        with open("working.path", 'r') as f:
            for line in f:
                li = line.strip()
                if not li.startswith("#"):
                    lir = li.rstrip()
                    if lir != '':
                        working_path = lir
                        path_found = True
        if not path_found or not os.path.exists(working_path):
            print "Working path not found in 'working.path'. Please try again next time."
            print "setting working_path to ~/working/fb_recruit/working ..."
            working_path = os.path.expanduser("~/working/fb_recruit/working")

    with open(os.path.join(working_path, 'usefulness.pickle'), 'rb') as picklefile:
        usefulness = pickle.load(picklefile)
    with open(os.path.join(working_path, 'tag_occurrence.pickle'), 'rb') as picklefile:
        tag_occurrence = pickle.load(picklefile)
    with open(os.path.join(working_path, 'tags_co_occurrence.pickle'), 'rb') as picklefile:
        tags_co_occurrence = pickle.load(picklefile)
    with open(os.path.join(working_path, 'title_occurrence.pickle'), 'rb') as picklefile:
        title_occurrence = pickle.load(picklefile)
    with open(os.path.join(working_path, 'title_tag_co_occurrence.pickle'), 'rb') as picklefile:
        title_tag_co_occurrence = pickle.load(picklefile)

def count_pn(weights, threshold, start=None, n=None, random_start=False, out_of_core=False,
                 num_eng=1, engine_start=0, verbose=False):
    ''' Given weights and threshold for the meta-recommender, calculate
        the positives and negatives (tp, fp, fn) of a (sub)-set of training examples '''

    cdef int tp, fp, fn
    tp, fp, fn = 0, 0, 0
    cdef double p, r, f1

    n = n if n else len(titles_lst)
    if random_start:
        start = int(random.random() * (len(titles_lst) - n))
    else:
        start = 0 if not start else start

    cdef bytes ID, title_str, body, tags_str, tag

    if not out_of_core:
        # if in the 1-core case, i.e. MultiCore has not been called:
        if 'titles_lst' or 'tags_lst' not in globals():
            global titles_lst, tags_lst
            with open(os.path.join(working_path, 'titles_lst.pickle'), 'rb') as picklefile:
                titles_lst = pickle.load(picklefile)
            with open(os.path.join(working_path, 'tags_lst.pickle'), 'rb') as picklefile:
                tags_lst = pickle.load(picklefile)

        for title_str, tags_str in islice(izip(titles_lst, tags_lst), start, start+n):
            basic = basic_recommender(title_str, usefulness)
            tag2tag = tag2tag_recommender(basic, 10)
            title2tag = title2tag_recommender(title_str, 10)
            recommendations = meta_recommender([basic, tag2tag, title2tag],
                                               weights)
            selections = select_tags(recommendations, threshold=threshold)
            
            tags_set = set(tags_str.split())
            selections_set = set(selections)
            tp += len(selections_set & tags_set)
            fp += len(selections_set - tags_set)
            fn += len(tags_set - selections_set)
    else:
        if 'num_eng' in globals() and 'engine_start' in globals():
            global num_eng, engine_start
        for eid in xrange(engine_start, num_eng):
            for row in row_stream("../data/pruned_Train_%d.csv" % eid, start=start, stop=start+n):
                ID, title_str, body, tags_str = row
                basic = basic_recommender(title_str, usefulness)
                tag2tag = tag2tag_recommender(basic, 10)
                title2tag = title2tag_recommender(title_str, 10)
                recommendations = meta_recommender([basic, tag2tag, title2tag],
                                                   weights)
                selections = select_tags(recommendations, threshold=threshold)
                
                tags_set = set(tags_str.split())
                selections_set = set(selections)
                tp += len(selections_set & tags_set)
                fp += len(selections_set - tags_set)
                fn += len(tags_set - selections_set)

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


    return (tp, fp, fn)

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
        print "fn:", fn
        print "p:", p
        print "r:", r
        print "f1:", f1

    return f1

def custom_script(*args, **kwargs):
    '''
    call: custom_script(eid, weights, threshold, out_of_core=False, num_eng=1,
              engine_start=0, start=None, n=None, verbose=True)

    positional args:
        args[0] = eid
        args[1] = weights
        args[2] = threshold
    default kwargs:
        kwargs['out_of_core'] = False
        kwargs['num_eng'] = 1
        kwargs['engine_start'] = 0
        kwargs['start'] = None
        kwargs['n'] = None
        kwargs['verbose'] = True
    '''
    pn_args = args[1:]
    pn_kwargs = {}
    pn_kwargs['out_of_core'] = kwargs['out_of_core'] \
        if type(kwargs.get('out_of_core')) == bool else False
    pn_kwargs['num_eng'] = kwargs['num_eng'] if kwargs.get('num_eng') else 1
    pn_kwargs['engine_start'] = kwargs['engine_start'] if kwargs.get('engine_start') else 0
    pn_kwargs['start'] = kwargs['start'] if kwargs.get('start') else None
    pn_kwargs['n'] = kwargs['n'] if kwargs.get('n') else None
    pn_kwargs['verbose'] = kwargs['verbose'] if type(kwargs.get('verbose')) == bool else True
    pn_kwargs['random_start'] = kwargs['random_start'] if kwargs.get('random_start') else False

    load_dicts()
    with MultiCore(args[0]):
        return count_pn(*pn_args, **pn_kwargs)


if __name__ == '__main__':
    pass