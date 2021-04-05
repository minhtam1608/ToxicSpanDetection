from ..evaluation.semeval2021 import batch_IO_f1
from .dataset import process
from tqdm import tqdm_notebook
import numpy as np
import torch
# from flair.training_utils import store_embeddings


class EvalStorage():
  def __init__(self):
    self.f1_scores = []
    # self.epoch_loss = []

  def reset(self):
    self.f1_scores = []
    # self.epoch_loss = []

  def update(self, f1_score):

    self.f1_scores.extend(f1_score)
    # self.epoch_loss.append(loss.detach().cpu().item())

  def cal_metrics(self):
    f1 = [score[0] for score in self.f1_scores]
    precision = [score[1] for score in self.f1_scores]
    recall = [score[2] for score in self.f1_scores]
    self.average_f1 = sum(f1) / len(f1)
    self.precision = sum(precision) /( len(precision) + 1e-12)
    self.recall = sum(recall) / (len(recall) + 1e-12)
    # self.average_loss = sum(self.epoch_loss) / len(self.epoch_loss)

  def print_performance(self, i_epoch, mode  ='train'):
    print(f"epoch:{i_epoch+1}, {mode} \n")
    print(f"f1:{np.round(self.average_f1, 5)}, precision:{np.round(self.precision, 5)}, recall:{np.round(self.recall, 5)}")
  
  def get_result(self):
      return self.average_f1

class Learner():
  def __init__(self, model, train_dl, dev_dl, test_dl, loss_fn, optimize_method, embedding_storage_mode = 'gpu'):
    self.model = model
    self.train_dl = train_dl
    self.dev_dl = dev_dl
    self.test_dl = test_dl
    self.loss_fn = loss_fn
    self.optimize_method = optimize_method
    self.embedding_storage_mode = embedding_storage_mode
    self.backbone_blocks = [model.backbone.embeddings] + [model.backbone.encoder.layer[i] for i in range(model.backbone.config.num_hidden_layers)]
    self.model.cuda()
    self.train_global_step = 0
    self.dev_global_step = 0

    self.train_storage = EvalStorage()
    self.val_storage = EvalStorage()
    self.test_storage = EvalStorage()

  def even_mults(self, start, stop, n, gamma = None):
      "Build log-stepped array from `start` to `stop` in `n` steps."
      mult = gamma if start is None else stop/start
      start = stop/mult if start is None else start
      return np.array([start*(mult**(i/(n-1))) for i in range(n)])

  def discriminative_lr(self, lr_start, lr_stop, gamma = 10):
    rest_of_head = self.model.custom_head
    blocks = self.backbone_blocks + [rest_of_head]
    lrs = self.even_mults(start = lr_start, stop = lr_stop, n = len(blocks), gamma = gamma)
    assert len(blocks) == len(lrs)
    lr_dict = [{'params':block.parameters(), 'lr':lr} for block, lr in zip(blocks, lrs)]
    # if self.use_ner_label_weight:
    #   lr_dict.append({'params': self.model.custom_head.ner_linear.parameters(), 'lr': lrs[0]})
    return self.optimize_method(lr_dict , lr = lr_stop), lrs

  def freeze(self):
    """freeze model's backbone"""
    for param in self.model.embeddings.parameters():
      param.requires_grad = False

  def unfreeze(self, num_block = None):
    for param in model.backbone.parameters():
      param.requires_grad = True
    num_block = len(self.backbone_blocks) if num_block is None else num_block
    for block in self.backbone_blocks[-num_block:]:
      for param in block.parameters():
        param.requires_grad = True 
    # for param in self.model.embeddings.parameters():
    #   param.requires_grad = True

  def is_freeze(self):
    '''check if any block of backbone is freezed'''
    freezed = []
    for i, block in enumerate(self.backbone_blocks):
      for param in block.parameters():
        if param.requires_grad == False:
          freezed.append(i)
        break
    return freezed

  def train(self, epochs, lr_max, name, lr_min= None, gamma = None, schedule = False, gradual_unfreezing = False, dis_lr = False, final_div_factor = 1e2, div_factor  =25):


    param_optimizer = list(self.model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

    optimizer= self.optimize_method(optimizer_grouped_parameters, lr = lr_max, weight_decay  = 0.001)
    if schedule:
      max_lr = list(lrs) if dis_lr else lr_max
      scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=max_lr,div_factor = div_factor, steps_per_epoch=len(self.train_dl), epochs=epochs, final_div_factor=final_div_factor)
    else:
      scheduler = None
    
    best_f1 = -1

    for i in range(epochs):
      if gradual_unfreezing:
        self.unfreeze(num_block = i + 1)
      self.train_one_epoch(optimizer, i, scheduler)
      f1 = self.eval_one_epoch(i)

      if f1 > best_f1:
          print('Save model ....')
          best_f1 = f1
          torch.save(self.model.state_dict(), '/content/drive/MyDrive/toxic_spans/models/' + name)

      torch.save(self.model.state_dict(), f'/content/model{i}{f1}.pth')

  def one_iter(self, batch):
    outputs = [t.cuda() for t in process(batch)]
    mode = batch[0].mode
    if mode == 'train':
      input_ids, attention_masks, token_type_ids, word_bpe_matrices, word_masks, labels = outputs
    if mode == 'val':
      input_ids, attention_masks, token_type_ids, word_bpe_matrices, word_masks  =outputs
    pred_logits = self.model(input_ids, attention_masks, token_type_ids, word_bpe_matrices, word_masks)
    # emissions = self.model(batch)
    # print(pred_logits.shape)
    # loss = self.loss_fn(emissions, batch)
    if mode  == 'train':
      loss = self.loss_fn(pred_logits, labels)
      return pred_logits, loss
    if mode == 'val': return pred_logits

  def cal_f1_batch(self,pred_logits, batch, return_pred = False, logits = True, mode = 'val'):
    mode = batch[0].mode
    label_spans = [feat.example['valid_spans'] for feat in batch] if mode == 'val' else None
    num_words = [feat.num_words for feat in batch]
    mappings = [feat.word_mapping for feat in batch]
    # subpos_perwords = [feat.subpos_perword for feat in batch]
    texts = [feat.example['text'] for feat in batch]
    return batch_IO_f1(pred_logits, word_mapping = mappings ,texts = texts ,num_words=num_words, label_spans = label_spans, return_pred = return_pred, mode = mode, logits = logits)

  def train_one_iter(self, batch, optimizer, scheduler = None):
    self.train_global_step += 1
    pred_logits, loss = self.one_iter(batch)

    loss.backward()
    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.)
    optimizer.step()
    if scheduler is not None:
        scheduler.step()
    optimizer.zero_grad()
    # store_embeddings(batch, self.embedding_storage_mode)


    # return self.cal_f1_batch(pred_logits, batch), loss
    
  def eval_one_iter(self, batch):
    self.dev_global_step += 1
    pred_logits = self.one_iter(batch)
    return self.cal_f1_batch(pred_logits, batch, mode = 'val')

  def eval_one_epoch(self, i_epoch):
    self.model.eval()
    with torch.no_grad():
      self.val_storage.reset()
      for batch in tqdm_notebook(self.dev_dl): 
        f1_score = self.eval_one_iter(batch)
        self.val_storage.update(f1_score)
        # store_embeddings(batch, storage_mode=self.embedding_storage_mode)
      self.val_storage.cal_metrics()
      self.val_storage.print_performance(i_epoch, mode = 'val')

      for batch in tqdm_notebook(self.test_dl): 
        f1_score = self.eval_one_iter(batch)
        self.test_storage.update(f1_score)
        # store_embeddings(batch, storage_mode=self.embedding_storage_mode)
      self.test_storage.cal_metrics()
      self.test_storage.print_performance(i_epoch, mode = 'test')

      f1 = self.test_storage.get_result()

      return f1

  def train_one_epoch(self, optimizer, i_epoch, scheduler = None):
    self.model.train()
    # self.train_storage.reset()
    for i, batch in tqdm_notebook(enumerate(self.train_dl)):
    # for batch in tqdm_notebook(self.train_dl):
      self.train_one_iter(batch, optimizer, scheduler)
      if (i %800 == 0) and (i>0) :
        self.eval_one_epoch(0)
        self.model.train()
      # self.train_storage.update(*self.train_one_iter(batch, optimizer, scheduler))
    # self.train_storage.cal_metrics()
    # self.train_storage.print_performance(i_epoch, mode = 'train')


  def predicted_spans(self, dl):
    all_spans = []
    submits = []
    all_prob = []
    self.model.eval()
    with torch.no_grad():
      for batch in tqdm_notebook(dl): 
        # pred_logits = self.model(batch)
        outputs = [t.cuda() for t in process(batch)]
        input_ids, attention_masks, token_type_ids, word_bpe_matrices, word_masks = outputs
        pred_logits = self.model(input_ids, attention_masks, token_type_ids, word_bpe_matrices, word_masks)
        pred_chars, submit = self.cal_f1_batch(pred_logits, batch, return_pred= True, mode = 'test')
        submits.extend(submit)
        texts = [feat.example.text for feat in batch]
        for spans, text in zip(pred_chars, texts):
          for s in spans:
            all_spans.append(text[s[0]:s[1] + 1])
        all_prob.append(torch.softmax(pred_logits.permute(0, 2, 1),dim=-1).detach().cpu())
    return all_spans, submits, all_prob

def labeling_data(model, dl):
  predict_lists = []
  model.eval()
  with torch.no_grad():
    for batch in tqdm_notebook(dl): 
      outputs = [t.cuda() for t in process(batch)]
      input_ids, attention_masks, token_type_ids, word_bpe_matrices, word_masks = outputs
      pred_logits = learner.model(input_ids, attention_masks, token_type_ids, word_bpe_matrices, word_masks)
      pred_logits = torch.softmax(pred_logits.permute(0, 2, 1), axis = -1)[:, :, 1]
      for pred, feat in zip(pred_logits, batch):
        # print(pred[1 : feat.num_words + 1], feat.num_words)
        predict_lists.append(pred[0: feat.num_words + 2])
  
  return predict_lists
    
