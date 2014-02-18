from __future__ import division
from collections import Counter, defaultdict
from gensim.corpora.dictionary import Dictionary
from lib.iterators import row_stream
from itertools import izip

import networkx as nx
from itertools import combinations

common, usefulness = defaultdict(int), defaultdict(int)
total = Dictionary.load("../working/titledict.pickle")

num_eng = 4
for eid in xrange(num_eng):
    for row in row_stream("../data/pruned_Train_%d.csv" % eid):
        ID, title, body, tags = row
        title_tokens = title.split()
        tags = set(tags.split())
        for token in title_tokens:
            if token in tags:
                common[token] += 1
            
for (hash_id, count) in total.dfs.iteritems():
    token = total[hash_id]
    usefulness[token] = common[token] / count


''' Tag==>Tag recommender '''
G = nx.Graph()

num_eng = 4
for eid in xrange(num_eng):
    for row in row_stream("../data/pruned_Train_%d.csv" % eid):
        ID, title, body, tags = row
        title_tokens = title.split()
        tags = tags.split()
        
        # add tags
        for tag in tags:
            if tag:
                if not G.has_node(tag):
                    G.add_node(tag)
                    G.node[tag]['tag_count'] = 1
                else:
                    G.node[tag]['tag_count'] += 1
        
        # add edges
        for edge in combinations(tags, 2):
            ni, nj = edge
            if not G.has_edge(ni, nj):
                G.add_edge(ni, nj, weight=1)
            else:
                G.edge[ni][nj]['weight'] += 1

def tag2tag_recommender(tags, n_recs=10):
    ''' Given a Counter {tags: scores}, generate and score associated tags. '''
    
    total_scores = Counter()
    
    # prune tokens with usefulness == 0
    tags -= Counter()
    
    for tag in tags:
        tag_scores = Counter({nj: tags[tag]*G.edge[tag][nj]['weight']/G.node[tag]['tag_count']
                              for _, nj in G.edges(tag)})
        tag_scores = dict(tag_scores.most_common(n_recs)) # keep best n_recs
        #total_scores = combine_tags(total_scores, tag_scores)
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
    
    for t in c1:
        if t in c2:
            c1[t] = 1 - (1 - c1[t])*(1 - c2[t])
    if c2:
        for t in c2:
            if t not in c1:
                c1[t] = c2[t] # add missing items from c2
    

''' Title==>Tag recommender '''
Y = nx.Graph()

num_eng = 4
for eid in xrange(num_eng):
    for row in row_stream("../data/pruned_Train_%d.csv" % eid):
        ID, title, body, tags = row
        title = title.split()
        tags = tags.split()
        
        # add tags
        for token in title:
            if not Y.has_node(token):              # add title token if not done so already
                Y.add_node(token, count=1)
            elif not Y.node[token].get('count'):   # if title token is a tag recently added into the graph,
                Y.node[token]['count'] = 1         # then start counting it
                
            else:                                  # if title token already added, increase its count
                Y.node[token]['count'] += 1
        
        for tag in tags:
            if not Y.has_edge(token, tag):
                Y.add_edge(token, tag, weight=1)
            else:
                Y.edge[token][tag]['weight'] += 1


def title2tag_recommender(tokens, n_recs=10):
    total_scores = Counter()
    for token in tokens:
        if Y.has_node(token):
            tag_scores = Counter({nj: Y.edge[token][nj]['weight'] / Y.node[token]['count']
                              for _,nj in Y.edges(token)})
            tag_scores = dict(tag_scores.most_common(n_recs))
            #total_scores = combine_tags(total_scores, tag_scores)
            combine_tags(total_scores, tag_scores)
    return total_scores - Counter()


''' Submission '''
def normalize(rec, weight=1):
    ''' Normalize each recommendation proportionally to the score of the leading tag. '''
    if rec: 
        max_score = rec[max(rec)]
        for tag in rec:
            rec[tag] = weight * rec[tag] / max_score
    return rec

def meta_recommender(recommendations, weights):
    ''' Probabilistically combine `recommendations` with corresponding `weights` '''
    total = Counter()
    for rec, weight in izip(recommendations, weights):
        rec = normalize(rec, weight)
        combine_tags(total, rec) # accumulate tag scores
    return total

def select_tags(rec, threshold=0.3):
    ''' Select at most 5 tags with score greater than `threshold` '''
    return [tag for tag, score in rec.most_common(5) if score > threshold]
