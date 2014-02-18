from __future__ import division
from itertools import islice, combinations, izip
import csv, pickle, dill
from collections import defaultdict
from gensim.corpora.dictionary import Dictionary
import os.path

from iterators import row_stream, line_stream

# Determine when building dictionaries, tags_lst &/ titles_lst should be in-core or out-of-core
out_of_core_flag = False
# Determine whether or not to save dictionaries for external use
external_use_flag = True # LEAVE AS TRUE; FALSE IS FOR DEBUGGING

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


if not out_of_core_flag:
    with open(os.path.join(working_path, 'titles_lst.pickle'), 'rb') as picklefile:
        titles_lst = pickle.load(picklefile)
    with open(os.path.join(working_path, 'tags_lst.pickle'), 'rb') as picklefile:
        tags_lst = pickle.load(picklefile)

''' Build usefulness '''
common, usefulness = defaultdict(int), defaultdict(int)
titledict = Dictionary.load(os.path.join(working_path, 'titledict.pickle'))
cdef bytes ID, title_str, body, tags_str, token

if not out_of_core_flag:
    for title_str, tags_str in izip(titles_lst, tags_lst):
        tags = tags_str.split()
        for token in title_str.split():
            if token in tags:
                common[token] += 1
else:
    num_eng = 4
    for eid in xrange(num_eng):
        for row in row_stream("../../data/pruned_Train_%d.csv" % eid):
            ID, title_str, body, tags_str = row
            tags = tags_str.split()
            for token in title_str.split():
                if token in tags:
                    common[token] += 1

cdef int hash_id, count
for (hash_id, count) in titledict.dfs.iteritems():
    usefulness[titledict[hash_id]] = common[titledict[hash_id]] / count
del common, titledict

# Save usefulness
if external_use_flag:
    temp = {}
    for t_tag, t_use in usefulness.iteritems():
        temp[t_tag] = t_use
    with open("../../working/usefulness.pickle", 'wb') as picklefile:
        pickle.dump(temp, picklefile, -1)
    del temp, usefulness


''' Build tag_occurrence and tags_co_occurrence '''
tag_occurrence, tags_co_occurrence = defaultdict(int), defaultdict(lambda : defaultdict(int))
cdef bytes tag, tag2
if not out_of_core_flag:
    for tags_str in tags_lst:
        tags = tags_str.split()
        for tag in tags:
            tag_occurrence[tag] += 1
        for tag, tag2 in combinations(tags, 2):
            tags_co_occurrence[tag][tag2] += 1
            tags_co_occurrence[tag2][tag] += 1
else:
    for eid in xrange(num_eng):
        for tags_str in line_stream(os.path.abspath(working_path + '../data/' + 'pruned_tagsonly_%d.txt' % eid)):
            tags = tags_str.split()
            for tag in tags:
                tag_occurrence[tag] += 1
            for tag, tag2 in combinations(tags, 2):
                tags_co_occurrence[tag][tag2] += 1
                tags_co_occurrence[tag2][tag] += 1

# Save tag_occurrence and tags_co_occurrence
if external_use_flag:
    temp = {}
    for t_tag, t_count in tag_occurrence.iteritems():
        temp[t_tag] = t_count
    with open(os.path.join(working_path, 'tag_occurrence.pickle'), 'wb') as picklefile:
        pickle.dump(temp, picklefile, -1)
    del temp, tag_occurrence
    
    temp = {}
    for t_token, d in tags_co_occurrence.iteritems():
        td = temp[t_token] = {}
        for t_tag, t_count in d.iteritems():
            td[t_tag] = t_count
    with open(os.path.join(working_path, 'tags_co_occurrence.pickle'), 'wb') as picklefile:
        pickle.dump(temp, picklefile, -1)
    del temp, tags_co_occurrence


''' Build title_occurrence and title_tag_co_occurrence '''
title_occurrence, title_tag_co_occurrence = defaultdict(int), defaultdict(lambda : defaultdict(int))
if not out_of_core_flag:
    for title_str, tags_str in izip(titles_lst, tags_lst):
        for token in title_str.split():
            title_occurrence[token] += 1
            for tag in tags_str.split():
                title_tag_co_occurrence[token][tag] += 1
else:
    num_eng = 4
    for eid in xrange(num_eng):
        for row in row_stream(os.path.abspath(working_path + '../data/' + "../../data/pruned_Train_%d.csv" % eid)):
            ID, title_str, body, tags_str = row
            for token in title_str.split():
                title_occurrence[token] += 1
                for tag in tags_str.split():
                    title_tag_co_occurrence[token][tag] += 1

# Save title_occurrence and title_tag_co_occurrence
if external_use_flag:
    temp = {}
    for t_token, t_count in title_occurrence.iteritems():
        temp[t_token] = t_count
    with open(os.path.join(working_path, 'title_occurrence.pickle'), 'wb') as picklefile:
        pickle.dump(temp, picklefile, -1)
    del temp, title_occurrence
    
    temp = {}
    for t_token, d in title_tag_co_occurrence.iteritems():
        td = temp[t_token] = {}
        for t_tag, t_count in d.iteritems():
            td[t_tag] = t_count
    with open(os.path.join(working_path, 'title_tag_co_occurrence.pickle'), 'wb') as picklefile:
        pickle.dump(temp, picklefile, -1)
    del temp, title_tag_co_occurrence
