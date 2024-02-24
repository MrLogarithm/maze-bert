# Maze-BERT

This directory adapts the original [Maze](https://github.com/vboyce/Maze) code to more easily support additional languages and modern neural net architectures. Concretely, we replace the original RNN language model with a Transformer from the huggingface repository; this makes it easy to load in a model from another language (or a multilingual model) by changing the model name in the params file. 

## Installation
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cd maze_automate
./set_up.py --gulordava
```
For additional details, you can also consult the original installation and usage instructions at [vboyce.github.io/Maze/install.html](https://vboyce.github.io/Maze/install.html).

## Usage
```bash
source venv/bin/activate
cd maze_automate
# English
./distract.py test_input.txt output_file.txt -p params_en_bert.txt
# Korean
./distract.py test_input.txt output_file.txt -p params_ko_bert.txt
```

In `params_ko.txt`, the `model_path` parameter specifies which huggingface model will be used to sample word probabilities. `dictionary_class` specifies a class in `wordfreq_distractor.py` which will be used to sample word frequencies. To adapt this code to a new language, you must simply:
1. Change `model_path` to a model which supports the language in question;
2. Add a class `wordfreq_<language_name>_dict` to `wordfreq_distractor.py`. This should be as simple as copying one of the existing classes, changing the two-character language tag (e.g. 'ko', 'en', 'fr'), and possibly changing the regex used to filer out OOV items; and
3. Change `dictionary_class` to point to the class you just added.

Note that you will probably need to adjust the `min_delta` and `min_abs` surprisal thresholds, as the average suprisal may differ across languages and models. In particular, BERT models tend to have lower surprisal than the RNNs used by the original code, so the thresholds used by this fork will tend to be much smaller than those in the original code.

