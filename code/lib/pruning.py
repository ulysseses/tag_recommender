from __future__ import division
from gensim.corpora.dictionary import Dictionary
import pickle, csv
from lib.iterators import row_stream

def build_dictionary_from_splits(splits_template, column, n, save_pickle=None):
    ''' Build dictionary from splits. If `save_pickle` is provided, then save. '''
    unfiltered_dict = Dictionary()
    for eid in xrange(n):
        unfiltered_dict.add_documents(csv_isolator("../../data/proc_Train_%d.csv" % eid, column))
    print "Before filtering,", unfiltered_dict
    if save_pickle:
        print "\nsaving..."
        unfiltered_dict.save(save_pickle)
    
    return unfiltered_dict


def build_dictionaries_from_splits(splits_template, n, save_pickle_tup=None):
    ''' Builds all 3 dictionaries from splits. If provided, `save_pickle_tup` must
        be a 3-tuple of the picklefile names in the following order:
        
        (title, body, tags)
        
        If `save_pickle_tup[i]` is None, the corresponding dictionary will not be saved.
    '''
    utitledict, ubodydict, utagdict = Dictionary(), Dictionary(), Dictionary()
    for eid in xrange(n):
        for row in row_stream(splits_template % eid):
            ID, title, body, tags = row
            utitledict.doc2bow(title.split(), allow_update=True)
            ubodydict.doc2bow(body.split(), allow_update=True)
            utagdict.doc2bow(tags.split(), allow_update=True)
    
    assert ubodydict.num_docs == utitledict.num_docs == utagdict.num_docs
    print "Before filtering..."
    print "utitledict:", utitledict
    print "ubodydict:", ubodydict
    print "utagdict:", utagdict
    
    if save_pickle_tup:
        assert len(save_pickle_tup) == 3
        if save_pickle_tup[0]:
            print "saving utitledict..."
            utitledict.save(save_pickle_tup[0])
        if save_pickle_tup[1]:
            print "saving ubodydict..."
            ubodydict.save(save_pickle_tup[1])
        if save_pickle_tup[2]:
            print "saving utagdict..."
            utagdict.save(save_pickle_tup[2])
            
    return (utitledict, ubodydict, utagdict)


def filter_extremes_wrapper(gdict, no_below=1, no_above=1.0, keep_n=None, save_pickle=None):
    ''' Given unfiltered gensim-dict `gdict`, wrap filter_extremes '''
    if type(gdict) == str:
        gdict = Dictionary.load(gdict)
    print "Before filtering:", gdict
    gdict.filter_extremes(**kwargs)
    print "After filtering:", gdict
    
    if save_pickle:
        print "\nsaving..."
        gdict.save(save_pickle)
    
    return gdict


def prune_csv_file1(infilename, outfilename, column, gensim_dict):
    ''' Using the (one) provided gensim.corpora.dictionary.Dictionary, prune out
        tokens not found in the filtered dictionary. If filtered examples
        have no tokens, remove them from file. '''
    with open(outfilename, 'w') as f:
        wtr = csv.writer(f, delimiter=',')
        for row in row_stream(infilename):
            tokens = row[column].split()
            filtered_tokens = [token for token in tokens if token in gensim_dict.token2id]
            if not filtered_tokens: # if no tokens remain, remove
                continue
            row[column] = ' '.join(filtered_tokens) # Python generators are un-affected from this
            wtr.writerow(row)


def prune_csv_file2(infilename, outfilename, gdict_tup, col_tup):
    ''' `gdict_tup` must be a 2-tuple of gensim-dicts.
        `col_tup` are the matching selection columns.
        Prune a csv-file with 2 dictionaries simultaneously. '''
    col_a, col_b = col_tup
    gdict_a, gdict_b = gdict_tup
    with open(outfilename, 'w') as f:
        wtr = csv.writer(f, delimiter=',')
        for row in row_stream(infilename):
            tokens_a, tokens_b = row[col_a].split(), row[col_b].split()
            filtered_tokens_a = [token for token in tokens_a if token in gdict_a.token2id]
            filtered_tokens_b = [token for token in tokens_b if token in gdict_b.token2id]
            if not filtered_tokens_a or not filtered_tokens_b:
                continue
            row[col_a] = ' '.join(filtered_tokens_a)
            row[col_b] = ' '.join(filtered_tokens_b)
            wtr.writerow(row)


def prune_csv_file3(infilename, outfilename, gdict_tup, col_tup):
    ''' `gdict_tup` must be a 3-tup
        `col_tup` are the matching selection columns.
        Prune a csv-file with all 3 dictionaries simultaneously. '''
    col_a, col_b, col_c = col_tup
    gdict_a, gdict_b, gdict_c = gdict_tup
    with open(outfilename, 'w') as f:
        wtr = csv.writer(f, delimiter=',')
        for row in row_stream(infilename):
            tokens_a, tokens_b, tokens_c = row[col_a].split(), row[col_b].split(), row[col_c].split()
            filtered_tokens_a = [token for token in tokens_a if token in gdict_a.token2id]
            filtered_tokens_b = [token for token in tokens_b if token in gdict_b.token2id]
            filtered_tokens_c = [token for token in tokens_c if token in gdict_c.token2id]
            if not filtered_tokens_a or not filtered_tokens_b or not filtered_tokens_c:
                continue
            row[col_a] = ' '.join(filtered_tokens_a)
            row[col_b] = ' '.join(filtered_tokens_b)
            row[col_c] = ' '.join(filtered_tokens_c)
            wtr.writerow(row)


from numba import autojit
import numpy as np
from matplotlib import pyplot as plt
import operator

@autojit
def delta(a, b):
    ''' Record the delta changes from `a` into `b`. '''
    b[0] = a[0]
    for i in xrange(1, len(a)):
        b[i] = b[i-1] + a[i]

def plot_dict_hist(gdict):
    ''' Provided gensim-dict `gdict`, plot hist statistics '''
    if type(gdict) == str:
        gdict = Dictionary.load(gdict)
    sorted_dfs = sorted(gdict.dfs.iteritems(), key=operator.itemgetter(1), reverse=True)
    y = [tup[1] for tup in sorted_dfs]
    x = arange(0, len(y))
    
    plt.figure(figsize=(8,5));
    plt.loglog(x, y);
    plt.grid();
    plt.xlabel("Token rank");
    plt.ylabel("Document count");
    
    cdf = np.empty(len(y))
    delta(y, cdf)
    cdf /= np.max(cdf) # normalize
    
    x50 = x[cdf > 0.50][0]
    x80 = x[cdf > 0.80][0]
    x90 = x[cdf > 0.90][0]
    x95 = x[cdf > 0.95][0]
    
    plt.axvline(x50, color='c');
    plt.axvline(x80, color='g');
    plt.axvline(x90, color='r');
    plt.axvline(x95, color='k');
    
    print "50%\t", x50
    print "80%\t", x80
    print "90%\t", x90
    print "95%\t", x95

def analyze_top_dfs(tokendict, tagdict, cutoff_factor=1):
    ''' Provided gensim-dicts `tokendict` and `tagsdict`, show the top word frequencies. '''
    if type(tokendict) == str:
        tokendict = Dictionary.load(tokendict)
    if type(tagdict) == str:
        tagdict = Dictionary.load(tagdict)
    
    max_tag_df = max(tagdict.dfs.iteritems(), key=operator.itemgetter(1))
    sorted_dfs = sorted(tokendict.dfs.iteritems(), key=operator.itemgetter(1), reverse=True)
    print "count threshold: %-15s\t%d" % (tagdict[max_tag_df[0]], max_tag_df[1])
    print "----------------------------------------------"
    for tup in sorted_dfs[:100]:
        if tup[1] > max_tag_df[1] * cutoff_factor:
            print "%-15s\t%d" % (tokendict[tup[0]][:15], tup[1])
        else: break