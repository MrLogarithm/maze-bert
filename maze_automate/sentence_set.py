import logging
from utils import copy_punct, strip_punct
from limit_repeats import Repeatcounter

import numpy as np

def no_duplicates(my_list):
    """True if list has no duplicates, else false"""
    return len(my_list) == len(set(my_list))


class Sentence:
    """a sentence to get distractors for
    has a list of words (in the sentence)
    a list of labels for matching with other sentences
    an id = item number
    and tag, which we do nothing with, but store anyhow
    """

    def __init__(self, words, labels, id, tag):
        if no_duplicates(labels):
            self.words = words  # list of words in sentence
            self.word_sentence = " ".join(self.words)  # sentence itself
            self.labels = labels  # list of labels in sentence
            self.label_sentence = " ".join([str(lab) for lab in self.labels])  # labels as sentence
            self.id = id  # item number
            self.tag = tag  # group/type
            self.distractors = ["x-x-x"]  # we probably shouldn't hard code this in this way, but whatevs
            self.distractor_sentence = ""
            self.probs = {}  # using a dictionary so we can start at 1 and not 0
            self.surprisal = {}
        else:
            logging.error("duplicate labels on sentence %s", " ".join(words))
            raise ValueError("duplicate labels")

    def do_model(self, model, top_k=30522):
        """Run the model and record surprisals at each position"""
        for i in range(len(self.words)):
            masked = ' '.join(self.words[:i] + ['[MASK]'] + self.words[i+1:]).lower()
            masked = masked.replace("[mask]", "[MASK]")
            output = model(masked, top_k=top_k)
            output = {candidate["token_str"]: candidate['score'] for candidate in output}
            self.probs[self.labels[i]] = output

    def do_surprisal(self):
        """Get surprisals of words in sentence"""
        for i in range(1, len(self.labels)):  # zeroeth position doesn't count
            lab = self.labels[i]
            word = self.words[i].strip(".,!?:;()")
            try:
                target_score = self.probs[lab][word]
                self.surprisal[lab] = -1 * np.log2(target_score)
            except:
                self.surprisal[lab] = 0


class Label:
    """A set of words etc, associated with a label, within a sentence set"""

    def __init__(self, id, lab):
        self.id = id  # item number
        self.lab = lab
        self.words = []
        self.probs = []
        self.surprisals = []
        self.surprisal_targets = []

    def add_sentence(self, word, probs, surprisal):
        """Given a position that belongs in the label, add it's attributes to our lists"""
        self.words.append(word)
        self.probs.append(probs)
        self.surprisals.append(surprisal)

    def choose_distractor(self, dictionary, threshold_func, params, banned):
        """Given a parameters specified in params and stuff
        Find a distractor not on banned (banned=already used in same sentence set)
        That hopefully meets threshold"""
        for surprisal in self.surprisals:  # calculate desired surprisal thresholds
            self.surprisal_targets.append(max(params["min_abs"], surprisal + params["min_delta"]))
        # get us some distractor candidates
        min_length, max_length, min_freq, max_freq = threshold_func(self.words)
        distractor_opts = dictionary.get_potential_distractors(min_length, max_length, min_freq, max_freq, params)
        avoid=[]
        for word in self.words: #it's real awkward if the distractor is the same as the real word, so let's not do that
            avoid.append(strip_punct(word).lower())
        # initialize
        best_word = "x-x-x"
        best_min_surp = 0
        Zs = dict()
        for dist in distractor_opts:
            if dist not in banned and dist not in avoid:  # if we've already used it in this sentence set, don't bother
                good = True
                min_surp = 1e5
                for i in range(len(self.probs)):  # check distractor candidate against each sentence's probs
                    if dist not in self.probs[i]:
                        dist_surp = 1e5
                    else:
                        dist_surp = -1 * np.log2(self.probs[i][dist])

                    if dist_surp < self.surprisal_targets[i]:
                        good = False  # it doesn't meet the target
                        min_surp = min(min_surp, dist_surp)  # but we should keep track of the lowest anyway
                if good:  # stayed above all surprisal thresholds
                    self.distractor = dist  # we're done, yay!
                    return self.distractor
                if min_surp > best_min_surp:  # best so far
                    best_min_surp = min_surp
                    best_word = dist
        logging.warning("Could not find a word to meet threshold for item %s, label %s, returning %s with %f min surp instead",
            self.id, self.lab, best_word, best_min_surp)
        self.distractor = best_word
        return self.distractor


class Sentence_Set:
    """A set of sentence objects, with the same id"""

    def __init__(self, id):
        self.id = id
        self.sentences = []
        self.label_ids = set()
        self.first_labels = set()
        self.labels = {}  # dictionary of label:label object

    def add(self, sentence):
        """Adds a sentence item to the sentence_set"""
        if sentence.id == self.id:
            self.sentences.append(sentence)
            first_label = sentence.labels[0]
            self.first_labels = self.first_labels.union(set([first_label]))
            self.label_ids = self.label_ids.union(sentence.labels[1:])
            if self.first_labels & self.label_ids != set():
                logging.error("Labels of first words cannot match labels of later words in the same set in item %s", self.id)
                raise ValueError()
        else:
            logging.error("ID doesn't match")
            raise ValueError()

    def do_model(self, model):
        """Applies model to sentences"""
        for sentence in self.sentences:
            sentence.do_model(model)

    def do_surprisals(self):
        """Gets surprisals for the real words"""
        for sentence in self.sentences:
            sentence.do_surprisal()

    def make_labels(self):
        """Regroups the stuff in the sentence items into by-label groups"""
        for lab in self.label_ids: #init label objects
            self.labels[lab] = Label(self.id, lab)
        for sentence in self.sentences: #dump stuff into the label objects
            for i in range(1, len(sentence.labels)):
                lab = sentence.labels[i]
                self.labels[lab].add_sentence(sentence.words[i], sentence.probs[lab], sentence.surprisal[lab])

    def do_distractors(self, d, threshold_func, params, repeats):
        """Get distractors using specified stuff"""
        banned = repeats.banned[:] #don't allow duplicate distractors within the set
        for label in self.labels.values(): #get distractors for each label
            dist = label.choose_distractor(d, threshold_func, params, banned)
            banned.append(dist)
            repeats.increment(dist)
        for sentence in self.sentences: #give the sentences the distractors
            for i in range(1, len(sentence.labels)):
                lab = sentence.labels[i]
                # we match distractors to their real words on punctuation
                distractor = copy_punct(sentence.words[i], self.labels[lab].distractor)
                sentence.distractors.append(distractor)
            sentence.distractor_sentence = " ".join(sentence.distractors) #and in sentence_format

    def clean_up(self):
        """Removes memory intensive things like label items and prob distributions"""
        self.labels = {}
        for sentence in self.sentences:
            sentence.probs = []
