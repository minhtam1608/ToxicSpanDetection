#!/usr/bin/env python
import sys
import os
import os.path
from scipy.stats import sem
import numpy as np
from ast import literal_eval
import torch
from drive.MyDrive.toxic_spans.evaluation.fix_spans import _contiguous_ranges

common_target_lists = ['Gay', 'gays', 'gay', 'Trump', 'Black', 'black', 'Homosexual', 'Arabs',
                      'blacks', 'Christians', 'christians', 'CHRISTIANS', 'homosexual', 'Homosexual', 'Islamic',
                      'Jews', 'MUSLUMS', 'Muslims', 'Muslim', 'Mexicans', 'Mexican', 'white', 'asian', 'hindu', 'Islamist',
                      'buddhist', 'lesbian', 'WHITES','White', 'white', 'muslim', 'MUSLIMS', 'Muslims', 'women', 'moslums','homosexuals', 'Catholics', 'Hindus']

def get_f1(predictions, gold):
    """
    F1 (a.k.a. DICE) operating on two lists of offsets (e.g., character).
    >>> assert f1([0, 1, 4, 5], [0, 1, 6]) == 0.5714285714285714
    :param predictions: a list of predicted offsets
    :param gold: a list of offsets serving as the ground truth
    :return: a score between 0 and 1
    """
    if len(gold) == 0:
        return (1., 1., 1.) if len(predictions) == 0 else (0., 0., 0.)
    if len(predictions) == 0:
        return (0., 0., 0.)
    predictions_set = set(predictions)
    gold_set = set(gold)
    nom = 2 * len(predictions_set.intersection(gold_set))
    denom = len(predictions_set) + len(gold_set)
    precision = (nom/2)/len(predictions_set)
    recall = float(nom/2)/len(gold_set)

    return (float(nom)/float(denom), precision, recall)

# def subsIO2wordsIO(subIO, subpos_perword):
  
#   sub2word_mapping = {word_sub[0]: i for i, word_sub in enumerate(subpos_perword)}
#   word_toxic = np.zeros(len(subpos_perword))
#   sub_toxic = (subIO == 1).nonzero()[0] + 1
#   for sub in sub_toxic:
#     if sub in sub2word_mapping.keys():
#       word_toxic[sub2word_mapping[sub]] = 1
#   return word_toxic


def IO_spans(pred_doc):
  toxic = (pred_doc == 1).nonzero()[0]
  return _contiguous_ranges(toxic, str_type= False)

def IO_span_char(IO, mapping):
  # print(IO)
  span_char = []
  for token in IO:
    s_char = mapping[token[0]][0]
    e_char = mapping[token[1]][1] - 1
    span_char.append((s_char, e_char))

  return span_char

def list_spans2list_char(list_spans):
  list_char = []
  # print(list_spans)
  for span in list_spans:
    list_char.extend(list(range(span[0], span[1] + 1)))
  return list_char

import string 

def batch_IO_f1(pred_logits, word_mapping,texts,num_words,label_spans, return_pred = False,  mode = 'val', logits = True):
  if logits:
    pred_logits = pred_logits.permute(0, 2, 1).detach().cpu().numpy()
  else:
    pred_logits = pred_logits.detach().cpu().numpy()
  word_toxics = np.argmax(pred_logits, axis = -1)

  f1_scores = []
  pred_chars = []
  outs = []
  punc = string.punctuation + string.whitespace
  # for k,  (pred, sen) in enumerate(zip(word_toxics, batch)):
  for k,  (pred, mapping, text, num_word) in enumerate(zip(word_toxics, word_mapping, texts, num_words)):
    # wordIO = pred[: len(sen.word_mapping)]
    # word_span = IO_spans(wordIO)
    # pred_span_char = IO_span_char(word_span, sen.word_mapping)
    wordIO = pred[1: 1 + num_word]
    word_span = IO_spans(wordIO)
    pred_span_char = IO_span_char(word_span, mapping)


    fix_span_char = []
    for s in pred_span_char:
      
      word  = text[s[0]:s[1] + 1]  
      if (word in punc) & (len(word) == 1): continue
      if (word == ' ') or (word == '') or (word in common_target_lists) : continue 

      for i, char in enumerate(word):
        if char not in punc:
          break

      for j, char in enumerate(word[::-1]):
        if char not in punc:
          break

      new_word = word[i:-j] if j != 0 else word[i:]
      new_s = (s[0] + i, s[1] - j)
      assert new_word == text[new_s[0]:new_s[1] + 1]

      if new_word == '': continue
      
      fix_span_char.append(new_s)

    pred_char = list_spans2list_char(fix_span_char)
    pred_chars.append(fix_span_char)
    outs.append(pred_char)

    if mode == 'val':
      true_char = list_spans2list_char(label_spans[k])
      f1_scores.append(get_f1(pred_char, true_char))
  if (return_pred) or (mode == 'test'): return pred_chars, outs
  return f1_scores 




def evaluate(pred, gold):
    """
    Based on https://github.com/felipebravom/EmoInt/blob/master/codalab/scoring_program/evaluation.py
    :param pred: file with predictions
    :param gold: file with ground truth
    :return:
    """
    # read the predictions
    pred_lines = pred.readlines()
    # read the ground truth
    gold_lines = gold.readlines()

    # only when the same number of lines exists
    if (len(pred_lines) == len(gold_lines)):
        data_dic = {}
        for n, line in enumerate(gold_lines):
            parts = line.split('\t')
            if len(parts) == 2:
                data_dic[int(parts[0])] = [literal_eval(parts[1])]
            else:
                raise ValueError('Format problem for gold line %d.', n)

        for n, line in enumerate(pred_lines):
            parts = line.split('\t')
            if len(parts) == 2:
                if int(parts[0]) in data_dic:
                    try:
                        data_dic[int(parts[0])].append(literal_eval(parts[1]))
                    except ValueError:
                        # Invalid predictions are replaced by a default value
                        data_dic[int(parts[0])].append([])
                else:
                    raise ValueError('Invalid text id for pred line %d.', n)
            else:
                raise ValueError('Format problem for pred line %d.', n)

        # lists storing gold and prediction scores
        scores = []
        for id in data_dic:
            if len(data_dic[id]) == 2:
                gold_spans = data_dic[id][0]
                pred_spans = data_dic[id][1]
                scores.append(f1(pred_spans, gold_spans))
            else:
                sys.exit('Repeated id in test data.')

        return (np.mean(scores), sem(scores))

    else:
        sys.exit('Predictions and gold data have different number of lines.')


def main(argv):
    # https://github.com/Tivix/competition-examples/blob/master/compute_pi/program/evaluate.py
    # as per the metadata file, input and output directories are the arguments
    [input_dir, output_dir] = argv

    # unzipped submission data is always in the 'res' subdirectory
    # https://github.com/codalab/codalab-competitions/wiki/User_Building-a-Scoring-Program-for-a-Competition#directory-structure-for-submissions
    truth_path = os.path.join(input_dir, 'ref', 'spans-gold.txt')
    submission_path = os.path.join(input_dir, 'res', 'spans-pred.txt')
    if not os.path.exists(submission_path):
        sys.exit('Could not find submission file {0}'.format(submission_path))
    with open(submission_path) as pred, open(truth_path) as gold:
      scores = evaluate(pred, gold)

    # the scores for the leaderboard must be in a file named "scores.txt"
    # https://github.com/codalab/codalab-competitions/wiki/User_Building-a-Scoring-Program-for-a-Competition#directory-structure-for-submissions
    output_file = open(os.path.join(output_dir, 'scores.txt'), "w")
    output_file.write("spans_F1:{0}\n".format(scores[0]))
    output_file.close()


if __name__ == "__main__":
    main(sys.argv[1:])
