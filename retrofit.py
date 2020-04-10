import argparse
import gzip
import math
import numpy
import re
import sys
import os

from copy import deepcopy

isNumber = re.compile(r'\d+.*')
def norm_word(word):
  if isNumber.search(word.lower()):
    return '---num---'
  elif re.sub(r'\W+', '', word) == '':
    return '---punc---'
  else:
    return word.lower()

''' Read all the word vectors and normalize them '''
def read_word_vecs(filename):
  wordVectors = {}
  if filename.endswith('.gz'): fileObject = gzip.open(filename, 'r')
  else: fileObject = open(filename, 'r')
  
  V, dim = fileObject.readline().split()
  for line in fileObject:
    line = line.strip().lower()
    word = line.split()[0]
    wordVectors[word] = numpy.zeros(len(line.split())-1, dtype=float)
    for index, vecVal in enumerate(line.split()[1:]):
      wordVectors[word][index] = float(vecVal)
    ''' normalize weight vector '''
    wordVectors[word] /= math.sqrt((wordVectors[word]**2).sum() + 1e-6)
    
  sys.stderr.write("Vectors read from: "+filename+" \n")
  return wordVectors, V, dim

''' Write word vectors to file '''
def print_word_vecs(wordVectors, output_dir, V, dim, src_lang="en", tgt_lang="es"):
  sys.stderr.write('\nWriting down the vectors in '+output_dir+'\n')
  #outFile_tgt = open(os.path.join(output_dir, "vectors-%s" % tgt_lang), 'w')
  V_src = 0
  V_tgt = 0
  for word in wordVectors.keys():
    if word.startswith(src_lang):
      V_src += 1
    elif word.startswith(tgt_lang):
      V_tgt += 1

  os.makedirs(output_dir, exist_ok=True)
  def output_vec(V, dim, lang):
    outFile = open(os.path.join(output_dir, "vectors-%s.txt" % lang), 'w')  
    outFile.write("%s %s\n" % (V, dim))
    for word, values in wordVectors.items():
      if word.startswith("%s:" % lang):
        outFile.write(word[3:]+' ')
        for val in wordVectors[word]:
          outFile.write('%.4f' %(val)+' ')
        outFile.write('\n')      
    outFile.close()
  output_vec(V_src, dim, src_lang)
  output_vec(V_tgt, dim, tgt_lang)
  
''' Read the PPDB word relations as a dictionary '''
def read_lexicon(filename, src_lang="en", tgt_lang="es"):
  from collections import defaultdict
  lexicon = defaultdict(list)
  for line in open(filename, 'r'):
    #words = line.lower().strip().split()
    src_word, tgt_word = line.lower().strip().split()
    src_word = src_lang +":"+ src_word # assume lexicons do not have lang. prefixes
    tgt_word = tgt_lang +":"+ tgt_word
    #lexicon[norm_word(words[0])] = [norm_word(word) for word in words[1:]]
    #lexicon[norm_word(src_word)].append(norm_word(tgt_word))
    lexicon[norm_word(tgt_word)].append(norm_word(src_word))
    lexicon[norm_word(src_word)].append(norm_word(tgt_word))
  return lexicon

''' Retrofit word vectors to a lexicon '''
def retrofit(wordVecs, lexicon, numIters):
  newWordVecs = deepcopy(wordVecs)
  wvVocab = set(newWordVecs.keys())
  loopVocab = wvVocab.intersection(set(lexicon.keys()))
  for it in range(numIters):
    #print(newWordVecs)
    # loop through every node also in ontology (else just use data estimate)
    for word in loopVocab:
      wordNeighbours = set(lexicon[word]).intersection(wvVocab)
      numNeighbours = len(wordNeighbours)
      #no neighbours, pass - use data estimate
      if numNeighbours == 0:
        continue
      # the weight of the data estimate if the number of neighbours
      newVec = numNeighbours * wordVecs[word]
      # loop over neighbours and add to new vector (currently with weight 1)
      for ppWord in wordNeighbours:
        newVec += newWordVecs[ppWord]
      newWordVecs[word] = newVec/(2*numNeighbours)
  return newWordVecs
  
if __name__=='__main__':

  parser = argparse.ArgumentParser()
  parser.add_argument("-i", "--input", type=str, default=None, help="Input word vecs")
  parser.add_argument("-l", "--lexicon", type=str, default=None, help="Lexicon file name")
  parser.add_argument("--src_lang", type=str, default=None, help="Embedding file name")
  parser.add_argument("--tgt_lang", type=str, default=None, help="Embedding file name")
  parser.add_argument("-o", "--output", type=str, help="Output directory")
  parser.add_argument("-n", "--numiter", type=int, default=10, help="Num iterations")
  args = parser.parse_args()

  wordVecs, V, dim = read_word_vecs(args.input)
  lexicon = read_lexicon(args.lexicon, args.src_lang, args.tgt_lang)
  numIter = int(args.numiter)
  #outFileName = args.output
  
  ''' Enrich the word vectors using ppdb and print the enriched vectors '''
  print_word_vecs(retrofit(wordVecs, lexicon, numIter), args.output, V, dim, src_lang=args.src_lang, tgt_lang=args.tgt_lang) 
