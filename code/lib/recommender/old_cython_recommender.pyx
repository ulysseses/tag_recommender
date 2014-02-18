from __future__ import division
from collections import Counter
import pickle, csv, time
from itertools import izip, islice
import os.path

from lib.iterators import row_stream

# # load from previous state
# with open("working.path", 'r') as f:
#     for line in f:
#         li = line.strip()
#         if not li.startswith("#"):
#             lir = li.rstrip()
#             if lir != '':
#                 working_path = lir
#                 path_found = True
# if not path_found or not os.path.exists(working_path):
#     raise Exception("Working path not found in 'working.path'. Please try again.")

# with open(os.path.join(working_path, 'usefulness.pickle'), 'rb') as picklefile:
#     usefulness = pickle.load(picklefile)
# with open(os.path.join(working_path, 'tag_occurrence.pickle'), 'rb') as picklefile:
#     tag_occurrence = pickle.load(picklefile)
# with open(os.path.join(working_path, 'tags_co_occurrence.pickle'), 'rb') as picklefile:
#     tags_co_occurrence = pickle.load(picklefile)
# with open(os.path.join(working_path, 'title_occurrence.pickle'), 'rb') as picklefile:
#     title_occurrence = pickle.load(picklefile)
# with open(os.path.join(working_path, 'title_tag_co_occurrence.pickle'), 'rb') as picklefile:
#     title_tag_co_occurrence = pickle.load(picklefile)

out_of_core_flag = False

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
        max_score = rec[max(rec)]
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
    def __init__(eid):
        self.eid = eid

    def __enter__(self):
        if not out_of_core_flag:
            global titles_lst, tags_lst
            with open(os.path.join(working_path, 'titles_lst_%d.pickle' % eid), 'rb') as picklefile:
                titles_lst = pickle.load(picklefile)
            with open(os.path.join(working_path, 'tags_lst_%d.pickle' % eid), 'rb') as picklefile:
                tags_lst = pickle.load(picklefile)
        else:
            kwargs['engine_start'] = self.eid

        return

    def __exit__(self, type, value, traceback):
        return


def calculate_f1_score(weights, threshold, num_eng=2, engine_start=0, verbose=False):
    ''' Given weights and threshold for the meta-recommender, calculate
        the F1 score on a (sub)-set of training examples '''

    # load from previous state
    with open("working.path", 'r') as f:
        for line in f:
            li = line.strip()
            if not li.startswith("#"):
                lir = li.rstrip()
                if lir != '':
                    working_path = lir
                    path_found = True
    if not path_found or not os.path.exists(working_path):
        raise Exception("Working path not found in 'working.path'. Please try again.")

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


    cdef int tp, fp, fn
    tp, fp, fn = 0, 0, 0
    cdef double p, r, f1

    #num_eng = 2
    n = 10000
    start = 0

    cdef double max_score
    cdef bytes tag

    cdef bytes ID, title_str, body, tags_str

    if not out_of_core_flag:
        if 'titles_lst' or 'tags_lst' not in globals():
            global titles_lst, tags_lst
            with open(os.path.join(working_path, 'titles_lst.pickle'), 'rb') as picklefile:
                titles_lst = pickle.load(picklefile)
            with open(os.path.join(working_path, 'tags_lst.pickle'), 'rb') as picklefile:
                tags_lst = pickle.load(picklefile)

        for title_str, tags_str in islice(izip(titles_lst, tags_lst), start, start+n):
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
    else:
        for eid in xrange(engine_start, num_eng):
            for row in row_stream("../data/pruned_Train_%d.csv" % eid, start=start, stop=start+n):
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



if __name__ == '__main__':
    t0 = time.time()
    calculate_f1_score([0.33, 0.39, 0.33], 0.4)
    t1 = time.time()
    print "time elapsed:", t1 - t0, "seconds"