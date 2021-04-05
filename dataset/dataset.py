from transformers import AutoTokenizer
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import random
# from flair.data import Sentence


tokenizer = AutoTokenizer.from_pretrained("unitary/toxic-bert")
tokenizer.return_offset_mapping = True
tokenizer.add_special_space = True

from transformers import AutoTokenizer
tokenizer_roberta = AutoTokenizer.from_pretrained("unitary/unbiased-toxic-roberta", add_prefix_space = True)
tokenizer_roberta.return_offset_mapping = True
tokenizer_roberta.add_special_space = True

import spacy
nlp = spacy.load('en')
import copy



def get_word_position(sent):
  list_tokens = nlp(sent)
  list_words = [token.text for token in list_tokens]
  list_pos =[(token.idx, token.idx + len(token.text)) for token in list_tokens]
  return list_words, list_pos 


class Feature():
  def __init__(self, example, tokenizer, max_words = 180, max_subw_per_word = 8, max_subw = 200, confidence_thres = 0.3, mode = 'train',p = 1.):
    self.example = example
    self.tokenizer = tokenizer
    self.max_words = max_words
    self.max_subw_per_word = max_subw_per_word
    self.max_subw = max_subw
    self.confidence_thres = confidence_thres

    self.mode = mode
    # self.max_neg_rel = max_neg_rel
    # assert (tokenizer_type == 'roberta') or (tokenizer_type == 'bert')
    # self.tokenizer_type = tokenizer_type
    self.p = p
    self.list_words = example.list_words
    self.word_mapping = example.word_mapping
    if mode == 'train':
      self.word_IO = example.wordIO

    # self.word_mapping = word_mapping
    self._token_ids, self.num_subtokens, self.subpos_perword = self.get_input_ids()

  def get_input_ids(self):
    _token_ids = []
    sub_len = []
    for word in ([self.tokenizer.cls_token] + self.list_words + [self.tokenizer.sep_token]):
      subs = self.tokenizer.tokenize(word)
      if len(subs) == 0:
        subs = [tokenizer.pad_token]
        # print('hihi')
      sub_len.append(len(subs))
        # assert [self.tokenizer.convert_tokens_to_ids(sub) for sub in subs] == [0]
      token_id = [self.tokenizer.convert_tokens_to_ids(sub) for sub in subs]
      _token_ids.extend(token_id)

    i = 1
    _subpos_perword = []
    for j in sub_len[1:-1]:
      _subpos_perword.append((list(range(i, i + j))))
      i = i + j

    return _token_ids, len(_token_ids) - 2, _subpos_perword

  @property
  def input_ids(self):
    tokens = copy.deepcopy(self._token_ids)
    pad_len = self.max_subw - len(self._token_ids)
    if pad_len > 0:
      tokens.extend([tokenizer.pad_token_id]*pad_len)
    return tokens

  @property
  def attention_mask(self):
    mask = np.zeros(self.max_subw, dtype= np.int) 
    mask[: self.num_subtokens + 2] = 1
    return mask

  @property
  def token_type_ids(self):
    return np.zeros(self.max_subw, dtype= np.int)

  @property
  def num_words(self):
    return len(self.list_words)


    
  def get_span_label_IO(self, mapping):
    _len = len(mapping)
    new_span = np.zeros(len(mapping))
    
    def find_pos(pos_char):
      pos_sub = 0
      for i, subpos in enumerate(mapping):
        s_subpos, e_subpos = subpos
        if (pos_char >= s_subpos) and (pos_char <= e_subpos):
          pos_sub =  i
          break
      return pos_sub

    for span in self.example['valid_spans']:
      start_char, end_char= span[0], span[1] + 1
      start_sub, end_sub = find_pos(start_char), find_pos(end_char)
      new_span[start_sub: end_sub + 1] = 1
      # if (end_sub - start_sub) >= self.max_label_word:
      #   p  = np.random.rand(1)[0]
      #   if p > self.p:
      #     new_span[start_sub: end_sub + 1] = 0
 
    return new_span

  # @property
  # def subpos_perword(self):
  #   sub_len = []
  #   for word in self.list_words:
  #     subs = self.tokenizer.tokenize(word)
  #     _len = 1 if len(subs) == 0 else len(subs)
  #     sub_len.append(_len)
  #   # sub_len = [0] + [len(self.tokenizer.tokenize(word)) for word in self.list_words]
  #   i = 1
  #   _subpos_perword = []
  #   for j in sub_len:
  #     _subpos_perword.append((list(range(i, i + j))))
  #     i = i + j
  #   return _subpos_perword
      
  @property
  def word_bpe_matrices(self):
    out = np.zeros((self.max_words, self.max_subw_per_word))
    for i, subpos in enumerate(self.subpos_perword):
      _len = len(subpos)
      out[i+1, :_len] = subpos if _len > 1 else subpos[0]
    out[self.num_words+1][0] = self.num_subtokens+1
    return out

  @property
  def word_masks(self):
    word_mask = (self.word_bpe_matrices!= 0).astype(np.int32)

    ## account for the first token <s>
    word_mask[0][0] = 1
    word_mask[self.num_words + 1][0] = 1
    return word_mask

  # @property
  # def word_IO(self):
  #   return self.get_span_label_IO(self.word_mapping)

  @property
  def label(self):
    assert self.mode == 'train'
    seq_tag = np.ones((self.max_words), dtype = np.float32)*-100
    word_IO = copy.deepcopy(self.word_IO)
    word_IO[(word_IO > self.confidence_thres) & (word_IO < (1 - self.confidence_thres))] = -100
    seq_tag[0: self.num_words + 2] = word_IO

    # neg = list((seq_tag == 0).nonzero()[0])
    # # neg_sample = random.sample(neg, min(len(neg), self.max_neg_rel))
    # # ignore = [i for i in neg if i not in neg_sample]
    # for _neg in neg:

    #   p  = np.random.rand(1)[0]
    #   seq_tag[_neg] = 0  if p < self.p else -100
    # seq_tag[0] = 0
    # seq_tag[self.num_words + 1] = 0

    return seq_tag

def custom_collate_fn(batch):
  max_num_sub = max([feat.num_subtokens for feat in batch]) + 5
  for feat in batch:
    feat.max_subw = max_num_sub

  max_num_words = max([feat.num_words for feat in batch]) + 5
  max_subw_per_word = max([len(sub_pos) for feat in batch for sub_pos in feat.subpos_perword]) + 1
  for feat in batch:
    feat.max_words = max_num_words
    feat.max_subw_per_word = max_subw_per_word
  return batch

def process(batch):
  # return batch
  input_ids = torch.LongTensor(np.stack([feat.input_ids for feat in batch], axis = 0))
  attention_masks = torch.FloatTensor(np.stack([feat.attention_mask for feat in batch], axis = 0))
  token_type_ids = torch.LongTensor(np.stack([feat.token_type_ids for feat in batch], axis = 0))

  mode = batch[0].mode
  assert (mode == 'train') or (mode == 'test') or (mode == 'val')

  word_bpe_matrices = torch.LongTensor(np.stack([feat.word_bpe_matrices for feat in batch], axis = 0))
  word_masks = torch.LongTensor(np.stack([feat.word_masks for feat in batch], axis = 0))
  if batch[0].mode == 'train':
    labels = torch.FloatTensor(np.stack([feat.label for feat in batch], axis = 0))

    return input_ids, attention_masks, token_type_ids, word_bpe_matrices, word_masks, labels
  
  elif (batch[0].mode == 'test') or (batch[0].mode == 'val') :
    return input_ids, attention_masks, token_type_ids, word_bpe_matrices, word_masks

class ToxicFlairDataset():
  def __init__(self, df, confidence_thres = 0.3, mode = 'train'):
    self.df = df
    self.confidence_thres = confidence_thres
    self.mode = mode

  def __getitem__(self, idx):
    example = self.df.loc[idx]
    sentence = Sentence(example.list_words)

    sentence.list_words = []
    sentence.word_mapping = []
    ids = []
    remove = ['\u200b', '️']
    for i, word in enumerate(example.list_words):
      if len(word.split()) == 0 or (word in remove):
        ids.append(i)
      else:
        sentence.list_words.append(word)
        
    # sentence.list_words = extample.list_words
    sentence.text = example.text
    if self.mode == 'train':
      sentence.valid_spans = example.valid_spans
      label = []
      for i, (wordIO, mapping) in enumerate(zip(example.wordIO, example.word_mapping)):
        if i not in ids:
          label.append(wordIO)
          sentence.word_mapping.append(mapping)
      label = np.array(label, dtype  =np.float)
      label[(label > self.confidence_thres) & (label < (1 - self.confidence_thres))] = -100
      sentence.label = label
      assert(len(sentence.label) == len(sentence.tokens)), '{}'.format(idx)
     
    return sentence

  def __len__(self):
    return len(self.df)



class ToxicDataset():
  def __init__(self, df, tokenizer,  max_words = 150, max_subw_per_word = 10, max_subw = 180, mode = 'train', confidence_thres = 0.3) :
    self.df= df
    self.tokenizer = tokenizer
    self.max_words = max_words
    self.max_subw_per_word = max_subw_per_word
    self.max_subw = max_subw
    self.mode = mode
    self.confidence_thres = confidence_thres
    # self.max_neg_rel = max_neg_rel
    # self.tokenizer_type=tokenizer_type
    # word_list_and_pos = [get_word_position(text) for text in df.text.values]
    # self.list_words = [item[0] for item in word_list_and_pos]
    # self.word_mappings = [item[1] for item in word_list_and_pos]


  def __getitem__(self, idx):
    example = self.df.iloc[idx]
    feature = Feature(example, tokenizer= self.tokenizer, 
                      max_words = self.max_words,
                      max_subw_per_word = self.max_subw_per_word,
                      max_subw = self.max_subw,
                      mode = self.mode,
                      confidence_thres = self.confidence_thres)

    return feature

  def __len__(self):
    return len(self.df)

### hard labeling data
# submision -> continuous-range -> valid_spans -> word_io
# unlabeled_ds = ToxicDataset(unlabeled_df, tokenizer_roberta, mode= 'test')
# unlabeled_dl = DataLoader(unlabeled_ds, batch_size= bs*2 ,collate_fn= custom_collate_fn)
# submits = learner.predict_spans(unlabeled_dl)


# unlabeled_df.valid_spans = [_contiguous_ranges(span, str_type = False) for span in submits]
# unlabeled_ds = ToxicDataset(unlabeled_df, tokenizer_roberta, mode= 'test')

# unlabeled_df.wordIO = [feat.wordIO for feat in unlabeled_ds]
# unlabeled_ds = ToxicDataset(unlabeled_df, tokenizer_roberta, mode= 'test')
# unlabeled_dl = DataLoader(unlabeled_ds, batch_size= bs*2 ,collate_fn= custom_collate_fn)

### HATE EXPLAIN DATASET
# with open('/content/drive/MyDrive/toxic_spans/data/dataset.json', 'r') as f:
#   data = json.load(f)

# list_labels = [[an['label'] for an in data[key]['annotators']] for key in data.keys()]
# all_labels = set([l for label in list_labels for l in label])

# label_ids = []
# from collections import Counter
# for i, labels in enumerate(list_labels):
#   count_lab = Counter(labels)
#   common = count_lab.most_common()[0]
#   if (common[1] > 1) & (common[0] != 'normal'):
#     label_ids.append((i, common[0]))

# post_ids = np.array(list(data.keys()))
# toxic_post_ids = post_ids[[lab[0] for lab in label_ids]]
# toxic_post_ids = list(toxic_post_ids)
# toxic_post_ids.remove('24439295_gab')
# def all_vote(rationale):
#   return rationale.prod(0)

# rationales = [all_vote(np.array(data[id]['rationales'])) for id in toxic_post_ids]
# add_train_data = pd.DataFrame({'wordIO': [ra.astype(np.float) for ra in rationales]})
# texts = [data[id]['post_tokens'] for id in toxic_post_ids]
# add_train_data['list_words'] = texts


### ENSEMPLE AND SILVER LABELED DATA
# new_unlabeled = pd.read_csv('/content/drive/MyDrive/toxic_spans/data/new_unlabeled_data.csv')
# words, poss = [], []
# for sent in tqdm_notebook(new_unlabeled.text.values):
#   list_words, list_pos = get_word_position(sent)
#   words.append(list_words)
#   poss.append(list_pos)

# new_unlabeled['list_words'] = words
# new_unlabeled['word_mapping'] = poss

# with open('/content/drive/MyDrive/toxic_spans/data/new_unlabeled_df.plk', 'wb') as f:
#   pickle.dump(new_unlabeled, f)

# new_unlabeled_ds = ToxicDataset(new_unlabeled, tokenizer_roberta, mode= 'test')
# new_unlabeled_dl = DataLoader(new_unlabeled_ds, batch_size= 8 ,collate_fn= custom_collate_fn)

# dir = '/content/drive/MyDrive/toxic_spans/models/'
# model_name = ['model5.1.pth', 'model5.2.pth', 'model5.3.pth', 'model7.1.pth', 'model7.2.pth']
# all_probs = []
# for model in model_name:
#   learner.model.load_state_dict(torch.load(dir + model))
#   _, _, probs = learner.predicted_spans(new_unlabeled_dl)
#   all_probs.append(probs)

# ensemble_prob = []
# for prob1, prob2, prob3, prob4, prob5 in zip(*all_probs):
#   ensemble_prob.append(0.1*prob1 + 0.1*prob2 + 0.2*prob3 + 0.4*prob4 + 0.2*prob5)

# predict_lists = []
# for prob, batch in zip(ensemble_prob, new_unlabeled_dl):
#   for pred, feat in zip(prob[:, :, 1].detach().cpu().numpy(), batch):
#     predict_lists.append(pred[0: feat.num_words + 2])

# with open('/content/drive/MyDrive/new_list_prob.pkl', 'wb') as f:
#   pickle.dump(predict_lists, f)

# new_unlabeled['wordIO'] = predict_lists

# with open('/content/drive/MyDrive/toxic_spans/data/new_unlabeled_pred_data.plk', 'wb') as f:
#   pickle.dump(new_unlabeled, f)
