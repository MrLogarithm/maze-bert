import logging
import importlib
from set_params import set_params
from limit_repeats import Repeatcounter
from input import read_input
from output import save_ibex, save_delim
import os.path

from transformers import BertTokenizerFast, BertModel
from transformers import pipeline

def run_stuff(infile, outfile, parameters="params.txt", outformat="delim"):
    """Takes an input file, and an output file location
    Does the whole distractor thing (according to specified parameters)
    Writes in outformat"""
    if outformat not in ["delim", "ibex"]:
        logging.error("outfile format not understood")
        raise ValueError
    params = set_params(parameters)
    sents = read_input(infile)
    dict_class = getattr(importlib.import_module(params.get("dictionary_loc", "wordfreq_distractor")),
                         params.get("dictionary_class", "wordfreq_English_dict"))
    d = dict_class(params)
    tokenizer = BertTokenizerFast.from_pretrained(params['model_path'])
    unmasker = pipeline('fill-mask', model=params['model_path'])
    threshold_func = getattr(importlib.import_module(params.get("threshold_loc", "wordfreq_distractor")),
                             params.get("threshold_name", "get_thresholds"))
    repeats=Repeatcounter(params.get("max_repeat", 0))
    print(repeats.max)
    print(repeats.limit)
    for ss in sents.values():
        ss.do_model(unmasker)
        ss.do_surprisals()
        ss.make_labels()
        ss.do_distractors(d, threshold_func, params, repeats)
        print(repeats.distractors)
        print(repeats.banned)
        ss.clean_up()
    if outformat == "delim":
        save_delim(outfile, sents)
    else:
        save_ibex(outfile, sents)
