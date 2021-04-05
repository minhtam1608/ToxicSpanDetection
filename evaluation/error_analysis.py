common_target_lists = ['Gay', 'gays', 'gay', 'Trump', 'Black', 'black', 'Homosexual',
                      'blacks', 'Christians', 'christians', 'CHRISTIANS', 'homosexual', 'Homosexual', 'Islamic',
                      'Jews', 'MUSLUMS', 'Muslims', 'Muslim', 'Mexicans', 'Mexican', 'white', 'asian', 'hindu', 'Islamist',
                      'buddhist', 'lesbian', 'WHITES','White', 'white', 'muslim', 'MUSLIMS', 'Muslims', 'women', 'moslums','homosexuals', 'Catholics', 'Hindus']
def list_spans2list_char(list_spans):
  list_char = []
  for span in list_spans:
    list_char.extend(list(range(span[0], span[1] + 1)))
  return list_char

submits = []
with open('/content/drive/MyDrive/toxic_spans/spans-pred.txt', 'r') as f:
  for line in f:
    submits.append(line.strip().split('\t')[-1])

spans_pos = [_contiguous_ranges(sub) for sub in submits]
spans = [[feat.example.text[s[0]:s[1]+1] for s in span] for span, feat in zip(spans_pos, test_ds)]

for span, span_pos in zip(spans, spans_pos):
  for s, s_p in zip(span, span_pos):
    if s in common_target_lists:
      print(s, span, span_pos)
      span.remove(s)
      span_pos.remove(s_p)

new_submits = [list_spans2list_char(span) for span in spans_pos]
with open('/content/drive/MyDrive/toxic_spans/data/or_test_expand.plk', 'rb') as f:
  test_df = pickle.load(f)
import ast
f1 = [get_f1(pred, ast.literal_eval(true))[0] for pred, true in zip(new_submits, test_df.valid_spans.values)]
sum(f1)/len(f1)

import ast
with open('/content/7024.txt','r') as f:
    lines = f.readlines()
    va_pred = lines
va_pred = [ast.literal_eval(pred.split('\t')[-1]) for pred in va_pred]
final_pred = [set.intersection(set(tam_p), set(va_p)) for tam_p, va_p in zip(new_submits, va_pred)]
f1 = [get_f1(pred, ast.literal_eval(true))[0] for pred, true in zip(final_pred, test_df.valid_spans.values)]
sum(f1)/len(f1)