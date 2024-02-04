# Maze-BERT

This directory adapts the original [Maze](https://github.com/vboyce/Maze) code to more easily support additional languages and modern neural net architectures. Concretely, we replace the original RNN language model with a Transformer from the huggingface repository; this makes it easy to load in a model from another language (or a multilingual model) by changing the model name in the params file. 

## Installation
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

For additional details, you can also consult the original installation and usage instructions at [vboyce.github.io/Maze/install.html](https://vboyce.github.io/Maze/install.html), though note that this fork has diverged enough that those instructions may no longer be very relevant.

## Usage
```bash
source venv/bin/activate
cd maze_automate
# English
./distract.py test_input.txt output_file.txt -p params_en_bert.txt
# Korean
./distract.py test_input.txt output_file.txt -p params_ko_bert.txt
```

In `params_ko_bert.txt`, the `model_path` parameter specifies which huggingface model will be used to sample word probabilities. `dictionary_class` specifies a class in `wordfreq_distractor.py` which will be used to sample word frequencies. 

Note that you will probably need to adjust the `min_delta` and `min_abs` surprisal thresholds, as the average suprisal may differ across languages and models. In particular, BERT models tend to have lower surprisal than the RNNs used by the original code, so the thresholds used by this fork may be much smaller than those in the original code.

Depending on the model used, you may see a notification like this when running `distract.py`:

```
BertForMaskedLM LOAD REPORT from: kykim/bert-kor-base
Key                          | Status     |  | 
-----------------------------+------------+--+-
bert.pooler.dense.bias       | UNEXPECTED |  | 
bert.pooler.dense.weight     | UNEXPECTED |  | 
cls.seq_relationship.weight  | UNEXPECTED |  | 
bert.embeddings.position_ids | UNEXPECTED |  | 
cls.seq_relationship.bias    | UNEXPECTED |  | 

Notes:
- UNEXPECTED:	can be ignored when loading from different task/architecture; not ok if you expect identical arch.
```

This can be safely ignored. This message is just informing you that your chosen model has additional capabilities which are not being used by the distractor generation task.

## Adapting to New Languages
To adapt this code to a new language:
1. Edit the params file to change `model_path` to a transformer model which supports the desired language;
2. Add a class `wordfreq_<language_name>_dict` inside `wordfreq_distractor.py`. This should be as simple as copying one of the existing classes, changing the two-character language tag that gets passed to `wordfreq.get_frequency_dict` (e.g. 'ko', 'en', 'fr'), and possibly changing the regex used to filter out-of-vocabulary items (you can just use `if re.match("^.*$", word)` to allow everything); and finally,
3. Change `dictionary_class` in the params file to point to the class you just added.

