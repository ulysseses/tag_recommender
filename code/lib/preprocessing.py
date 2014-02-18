import nltk, csv, re, pickle
from nltk.tokenize import RegexpTokenizer
from bs4 import BeautifulSoup
from nltk.corpus.reader.wordnet import ADJ, NOUN, ADV, VERB
from iterators import row_stream


def remove_duplicate_rows(infilename, outfilename, key=1):
    ''' Remove duplicate rows on the `key`th column from the inputted csv file '''
    dupe_set = set()
    fin, fout = open(infilename), open(outfilename, 'w')
    with fin, fout:
        rdr, wtr = csv.reader(fin, delimiter=','), csv.writer(fout, delimiter=',')
        headers = rdr.next() # Ignore headers
        for row in rdr:
            if row[key] not in dupe_set:
                dupe_set.add(row[key])
                wtr.writerow(row)


# Treebank-POS tags stop-list
REJECT = set(["CC", "CD", "DT", "EX", "IN", "LS", "MD", "PDT", "POS",
              "PRP", "PRP$", "RB", "RP", "SYM", "TO", "UH", "WDT", "WP",
              "WP$", "WRB"])
              
ACCEPT = set(["FW", "JJ", "JJR", "JJS", "NN", "NNP", "NNPS", "NNS",
              "RBR", "RBS", "VB", "VBD", "VBG", "VBN", "VBP",
              "VBZ"])

# Manual stop-list
sw = set()
with open("../working/english.stop") as f:
    for l in f:
        sw.add(l.rstrip())

# NLTK Tokenizer, Tagger
tokenizer = RegexpTokenizer(r"\b\w[\w#+'-]*(?<!\.$)")
tagger = pickle.load(open("../working/treebank_brill_aubt.pickle"))
lemmatizer = nltk.stem.WordNetLemmatizer()

# NLTK Lemmatizer
wntag_dict = {'J':ADJ, 'V':VERB, 'N':NOUN, 'R':ADV, '-':NOUN}
def tb2wn(x):
    ''' Map Treebank POS tag -> WordNet Lemma tag '''
    return wntag_dict.get(x[0])


def process_title(title):
    ''' Extract the title string and wrap filter_tokens '''
    tokens = tokenizer.tokenize(title)
    return filter_tokens(tokens)

def process_body(body):
    ''' Extract the body string and wrap filter_tokens '''
    # Remove code and html-tags
    soup = BeautifulSoup(body)
    for tag in soup.find_all('p'):
        for child in tag.children:
            if child.name != 'a':
                if child.name == 'code':
                    for cc in child.children:
                        body_lst.append(cc.__repr__()[2:-1])
                else:
                    body_lst.append(child.__repr__()[2:-1])
    
    # Extract tokens from processed body
    tokens_lst = tokenizer.batch_tokenize(body_lst)
    tokens = []
    for tl in tokens_lst:
        tokens.extend(tl)
    return filter_tokens(tokens)

def filter_tokens(tokens):
    ''' Given a list of tokens, remove token if in REJECT or sw
        Return as concatenated string '''
    tkn_tag_tuples = [(token.lower(), tb2wn(tag)) for (token,tag) in tagger.tag(tokens)
                          if tag not in REJECT]
    lemmaed_tokens = [lemmatizer.lemmatize(*tup) for tup in tkn_tag_tuples
                          if (tup[1] is not None) and (tup[0] not in sw)]
    return ' '.join(lemmaed_tokens).decode('ascii', errors='ignore')

def filter_html_tags(body):
    ''' Use BeautifulSoup to remove html tags '''
    pass

def process_csv(infilename, outfilename, start=None, stop=None):
    with open(outfilename, 'w') as f:
        wtr = csv.writer(f, delimiter=',')
        for row in row_stream(infilename, start=start, stop=stop):
            ID, title, body, tags = row
            wtr.writerow([ ID, process_title(title.decode('ascii', errors='ignore')),
                           process_body(body.decode('ascii', errors='ignore')), tags ])