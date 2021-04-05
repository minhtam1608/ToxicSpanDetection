import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

def collect_word_subword_embeddings(sub_word_embeddings, word_bpe_matrices, word_masks):      
  batch, max_words, max_subw_per_word = word_bpe_matrices.shape
  feature_dim = sub_word_embeddings.shape[-1]

  word_bpe_matrices = word_bpe_matrices.view(batch, -1)
  word_sub_embeddings = torch.gather(sub_word_embeddings, 1, word_bpe_matrices[..., None].expand(*word_bpe_matrices.shape, feature_dim))
  word_sub_embeddings = word_sub_embeddings.view(batch, max_words, max_subw_per_word, feature_dim)
  return word_sub_embeddings * word_masks[..., None] 

class WordEmbeddings(nn.Module):
  def __init__(self, max_subw_per_word, feature_dim, method = 'sum', norm = True):
    super().__init__()

    assert method in ['sum', 'mean', 'max' , 'weighted_sum'], 'invalid method to aggregate subword embeddings'
    self.method = method
    self.norm = norm

    if method == 'weighted_sum':
      self.linear = nn.Linear(max_subw_per_word, 1)
    if norm:
      self.layernorm = nn.LayerNorm((feature_dim))

  def forward(self, word_sub_embeddings):
    if self.method == 'sum': 
      word_sub_embeddings= word_sub_embeddings.sum(dim = 2)
    elif self.method == 'mean': 
      word_sub_embeddings= word_sub_embeddings.mean(dim = 2)
    elif self.method == 'max': 
      word_sub_embeddings=  word_sub_embeddings.max(dim = 2)
    else:
      word_sub_embeddings = self.linear(word_sub_embeddings.permute(0, 1, 3, 2)).squeeze(-1)

    if self.norm:
      return self.layernorm(word_sub_embeddings)
    else: return word_sub_embeddings

class CustomHead(nn.Module):
  def __init__(self,feature_dim,hidden_size,  max_subw_per_word = None, method_word = 'sum', norm = False, activation = nn.ReLU6, dropout = True):
    super().__init__()


    #Word embedding 
    self.word_embeddings = WordEmbeddings(max_subw_per_word, feature_dim, method = method_word, norm = norm)

    self.linear_project = nn.Sequential( nn.Dropout(0.3),
                          nn.Linear(feature_dim, hidden_size), 
                          nn.Tanh())

    self.dropout = nn.Dropout(0.25)
    self.linear = nn.Linear(hidden_size, 2, bias = True)

  def forward(self, word_sub_embeddings):
    word_embeddings = self.word_embeddings(word_sub_embeddings)

    # NER prediction
    middle = self.linear_project(word_embeddings)
    outputs = self.linear(self.dropout(middle))

    return outputs.permute(0, 2, 1)

class ToxicModel(nn.Module):
  def __init__(self, backbone, custom_head, num_hidden_states = 1):
    super().__init__()
    self.backbone = backbone
    self.custom_head = custom_head
    self.num_hidden_states = num_hidden_states
    
  def forward(self, input_ids, attention_masks, token_type_ids, word_bpe_matrices, word_masks):
    outputs = self.backbone(input_ids, attention_masks, token_type_ids)
    hidden_states = outputs[2][1:]
    sub_word_embedings = torch.cat(hidden_states[-self.num_hidden_states:], dim= -1)
    word_sub_embeddings = collect_word_subword_embeddings(sub_word_embedings, word_bpe_matrices, word_masks)
    return self.custom_head(word_sub_embeddings)

class Loss(nn.Module):
  def __init__(self, ignore_id = -100, smooth_eps = 0.1):
    super().__init__()
    # self.weight_loss = weight_loss
    self.ignore_id = ignore_id
    # self.k = k
    self.smooth_eps = smooth_eps
    self.ner_loss = nn.CrossEntropyLoss(ignore_index= -100, reduction = 'mean')

  def forward(self, pred, gold):
    ner_loss = self.smooth_loss(pred, gold)
    # ner_loss = self.ner_loss(pred_ner, true_ner)
    # mask_ner = true_ner.ne(-100)
    return ner_loss

  def smooth_loss(self, pred, gold):
    # max_length = max([len(sentence.tokens) for sentence in sentences])

    # labels = np.ones((len(sentences), max_length),  dtype=np.float)*-100

    # for i, sen in enumerate(sentences):
    #   # print(i, len(labels[i][:len(sen.label)]), len(sen.label), max_length)
    #   labels[i][:len(sen.label)] = sen.label
    # gold = torch.FloatTensor(labels).cuda()
    # pred = pred.permute(0, 2, 1)
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''
    n_class = pred.shape[1]
    gold_copy = gold.detach().clone()
    # print(gold_copy)
    gold_copy[gold_copy == -100.] = 0

    zero_gold = 1 - gold_copy
    prob_gold = torch.cat((zero_gold.unsqueeze(-1), gold_copy.unsqueeze(-1)), dim = -1).permute(0, 2, 1)

    # one_hot = F.one_hot(gold_copy, num_classes= n_class)

    # one_hot = one_hot * (1 - self.smooth_eps) + (1 - one_hot) * self.smooth_eps / (n_class - 1)
    # if len(one_hot.shape) == 3: one_hot = one_hot.permute(0, 2, 1)
    # if len(one_hot.shape) == 4: one_hot = one_hot.permute(0, 3, 1, 2)
    log_prb = F.log_softmax(pred, dim= 1)

    # non_pad_mask = gold != self.ignore_id
    # non_pad_mask[]
    non_pad_mask = gold.ne(self.ignore_id)
    assert non_pad_mask.sum() > 0
    loss = -(prob_gold * log_prb).sum(dim=1)

    assert loss.shape == gold.shape
    loss = loss.masked_select(non_pad_mask).mean()  # average later

    return loss

# class Loss(nn.Module):
#   def __init__(self, num_tags = 2):
#     super().__init__()
#     self.crf = CRF(num_tags, batch_first= True).cuda()
  
#   def forward(self, emissions, sentences):
#     max_length = max([len(sentence.tokens) for sentence in sentences])

#     labels = np.zeros((len(sentences), max_length),  dtype=np.float)
#     masks = torch.zeros(len(sentences), max_length, device= 'cuda', dtype=torch.uint8)
#     for i, sen in enumerate(sentences):
#       for tok, map, t in zip(sen.tokens, sen.word_mapping, sen.list_words):
        
#         print(tok, map, t)
#       # print(sen.tokens)
#       # print(sen.word_mapping)
#       # print(sen.text)
#       print(len(sen.label))
#       labels[i][:len(sen.label)] = sen.label
#       masks[i][:len(sen.label)] = 1
#     labels = torch.FloatTensor(labels).cuda()

#     return -self.crf(emissions, labels, mask = masks), labels
