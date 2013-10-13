import re
import sys
import util
import nltk
import csv

DOCBIAS_REGEX = r'^DOCHASBIAS\((\w+), (\w+)\) Truth=\[(\d.\d+)\]'
BIAS_REGEX = r'^HASBIAS\((\w+), (\w+)\) Truth=\[(\d.\d+)\]'

indoc = util.indoc()
docbias = util.docbias()
words = util.words()
sents = util.sentences()

def correct(docid, bias):
    return bias == docbias[docid]

def sents_in(doc):
    return [sent for sent in indoc if indoc[sent] == doc]

def inferred(regex, group):
    ret = {}
    for line in group.split('\n'):
        match = re.search(regex, line)
        if match:
            key =  match.group(1)
            val = match.group(2)
            truth = float(match.group(3))
            ret.setdefault(key, {})[val] = truth
    return ret

if __name__ == '__main__':
    fname = sys.argv[1]
    docbias_group = False
    bias_group = False
    biases = []
    doc_biases = []
    offending_word_dist = nltk.FreqDist()
    with open(fname) as pslfile:
        text = pslfile.read()
        # Break text up into sections for each EM iteration for all experiments
        groups = text.split("--- Atoms:")[1:]
        inferred_bias = {}
        inferred_docbias = {}
        for n, group in enumerate(groups):
            # Load inferred sentence bias values for training sentences.
            if n % 12 == 9:
                inferred_bias = inferred(BIAS_REGEX, group)
                outfile = open('inferredtrain.csv', 'wb')
                writer = csv.writer(outfile, delimiter=',', quotechar='"')
                for sent in inferred_bias:
                    for bias, truth in inferred_bias[sent].iteritems():
                        writer.writerow([sent, bias, truth, docbias[indoc[sent]], sents[sent]])
                outfile.close()
            # Load inferred sentence bias values for test sentences.
            if n % 12 == 10:
                inferred_bias = inferred(BIAS_REGEX, group)
                outfile = open('inferredtest.csv', 'wb')
                writer = csv.writer(outfile, delimiter=',', quotechar='"')
                for sent in inferred_bias:
                    for bias, truth in inferred_bias[sent].iteritems():
                        writer.writerow([sent, bias, truth, docbias[indoc[sent]], sents[sent]])
                outfile.close()
