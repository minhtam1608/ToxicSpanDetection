## Self-Training for Toxic Span Detection
### Introduction
The SemEval 2021 task 5: Toxic Spans Detection identifies toxic spans in text, providing a  valuable,  automatic tool for moderating online contents. In this task, we utilize extra data with a  slightly customized  Self-training,  a  semi-supervised learning technique,  for sequence-tagging problems.

### Method
We  start  with a  baseline,  a  simple sequence tagger utilizing  ToxicRoBERTa as the backbone. In the Self-training branch, the teacher- the best-scored baseline, generates soft pseudo labels for the student to learn. On the other hand, the Feature-based Learning model concatenates the input vector with different embedding methods i.e.  Flair, FastText, and BPE, then trains the Named Entity Recognition task. Predicted character offsets (for each sentence) of two models are combined using Intersection Union (Ensemble Section) to obtain the final prediction. 
This GitHub repository is the implementation of the backbone and self-training branch.

## Dataset
The dataset can be found at : https://github.com/ipavlopoulos/toxic_spans
The original dataset contains 7939 annotated samples for training and 2000 unlabeled samples for testing. We use a small trial dataset, given by the task organizer, which consists of 690 labeled samples, as our development set. We train our models on the training set, use the development set to find the best hyper-parameters, and finally make our submission on the private test set.

## Results
