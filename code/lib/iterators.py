from itertools import islice
import csv

def row_stream(filename, start=None, stop=None):
    ''' A csv-row generator, starting on row `start` and ending on row `stop` '''
    with open(filename) as f:
        for row in islice(csv.reader(f, delimiter=',', quotechar='"'), start, stop):
            yield row

def line_stream(infilename, start=None, stop=None):
    with open(infilename) as f:
        for line in islice(f, start, stop):
            yield line

def split_file(infilename, outfiletemplate, n):
    ''' Split every n lines into a total of n outfiles. '''
    f_handlers = [open(outfiletemplate % eid, 'w') for eid in xrange(n)]
    with open(infilename) as f:
        for ln,line in enumerate(f):
            f_handlers[ln % n].write(line)

def split_csv(infilename, outfiletemplate, n):
    ''' Split every n lines into a total of n outfiles. '''
    f_handlers = [csv.writer(open(outfiletemplate % eid, 'w'), delimiter=',')
                      for eid in xrange(n)]
    with open(infilename) as f:
        for rn,row in enumerate(csv.reader(f, delimiter=',', quotechar='"')):
            f_handlers[rn % n].writerow(row)

def combine_files(infiletemplate, outfilename, n):
    ''' Given the value of n, combine n files into 1.
        Merges serially (file after file). '''
    with open(outfilename, 'w') as fout:
        for eid in xrange(n):
            fin = open(infiletemplate % eid)
            fout.write(fin.read())
            fin.close()

def combine_csvs(infiletemplate, outfilename, n):
    ''' Given the value of n, combine n csv files into 1.
        Merges serially (file after file). '''
    with open(outfilename, 'w') as fout:
        wtr = csv.writer(fout, delimiter=',')
        for eid in xrange(n):
            for row in row_stream(infiletemplate % eid):
                wtr.writerow(row)

def csv_isolator(filename, column):
    ''' Isolate a column in the csv-file and yield a list containing its contents. '''
    for row in row_stream(filename):
        yield row[column].split()

def extract_csv(infilename, outfilename, column):
    ''' Isolate a column in the csv-file and write it into another file. '''
    with open(outfilename, 'w') as f:
        for row in row_stream(infilename):
            f.write(row[column] + '\n')
            
def extract_from_csvs(infiletemplate, outfilename, n, column):
    ''' Isolate a column from multiple csv-files and write them
        serially into another file. '''
    with open(outfilename, 'w') as f:
        for eid in xrange(n):
            for row in row_stream(infiletemplate % eid):
                f.write(row[column] + '\n')

