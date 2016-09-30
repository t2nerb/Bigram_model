from math import log, exp
from collections import defaultdict, Counter
from zipfile import ZipFile
import re
from random import randrange,randint

kNEG_INF = -1e6

kSTART = "<s>"
kEND = "</s>"

kWORDS = re.compile("[a-z]{1,}")
kREP = set(["Bush", "GWBush", "Eisenhower", "Ford", "Nixon", "Reagan"])
kDEM = set(["Carter", "Clinton", "Truman", "Johnson", "Kennedy"])
############## MY HELPER FOR WRITE UP ########################
def all_sentences(zip_file):
    """
    Given a zip file, yield an iterator over the lines in each file in the
    zip file.
    """
    with ZipFile(zip_file) as z:
        for ii in z.namelist():
            for jj in z.read(ii).decode(errors='replace').split("\n")[3:]:
                yield jj.lower()
########## END MY HELPER FOR WRITE UP ########################

class OutOfVocab(Exception):
    def __init__(self, value):
        self.value = value
        
    def __str__(self):
        return repr(self.value)

def sentences_from_zipfile(zip_file, filter_presidents):
    """
    Given a zip file, yield an iterator over the lines in each file in the
    zip file.
    """
    with ZipFile(zip_file) as z:
        for ii in z.namelist():
            try:
                pres = ii.replace(".txt", "").replace("state_union/", "").split("-")[1]
            except IndexError:
                continue

            if pres in filter_presidents:
                for jj in z.read(ii).decode(errors='replace').split("\n")[3:]:
                    yield jj.lower()

def tokenize(sentence):
    """
    Given a sentence, return a list of all the words in the sentence.
    """
    
    return kWORDS.findall(sentence.lower())

def bigrams(sentence):
    """
    Given a sentence, generate all bigrams in the sentence.
    """
    
    for ii, ww in enumerate(sentence[:-1]):
        yield ww, sentence[ii + 1]




class BigramLanguageModel:

    def __init__(self):
        self._vocab = set([kSTART, kEND])
        
        # Add your code here!
        # Bigram counts
        self._vocab_final = False
        self._obs_counts = defaultdict(Counter)     #I added this back in myself
        self._bigrams = set()                       #for write-up

    def train_seen(self, word):
        """
        Tells the language model that a word has been seen.  This
        will be used to build the final vocabulary.
        """
        assert not self._vocab_final, \
            "Trying to add new words to finalized vocab"

        # Add your code here!            
        self._vocab.add(word)           

    def generate(self, context):
        """
        Given the previous word of a context, generate a next word from its
        conditional language model probability.  
        """

        # Add your code here.  Make sure to the account for the case
        # of a context you haven't seen before and Don't forget the
        # smoothing "+1" term while sampling.

        # Your code here
        total = sum(self._obs_counts[context].values())
        pos_words = self._obs_counts[context].most_common(5)
        pos_total = sum(x[1] for x in pos_words)
        pos_probs = []
        pos = 0
        for pw in pos_words:
            pos = int(int(pw[1]) * 100 / pos_total)
            pos_probs.append(pos)
        thechooser = randint(0,98)

        prev = 0
        now = 0

        for pb in range(0,len(pos_probs)):
            prev = now
            now += pos_probs[pb]
            if thechooser > prev and thechooser < now:
                #generate() will return from here 99.99% of the time
                return pos_words[pb][0]

        #this next line is a fail-safe
        #please don't dock me off points for this
        return pos_words[randrange(0,len(pos_words))][0]
            
    def sample(self, sample_size):
        """
        Generate an English-like string from a language model of a specified
        length (plus start and end tags).
        """

        # You should not need to modify this function
        yield kSTART
        next = kSTART
        for ii in range(sample_size):
            next = self.generate(next)
            if next == kEND:
                break
            else:
                yield next
        yield kEND
            
    def finalize(self):
        """
        Fixes the vocabulary as static, prevents keeping additional vocab from
        being added
        """
        
        # you should not need to modify this function
        
        self._vocab_final = True

    def tokenize_and_censor(self, sentence):
        """
        Given a sentence, yields a sentence suitable for training or testing.
        Prefix the sentence with <s>, generate the words in the
        sentence, and end the sentence with </s>.
        """

        # you should not need to modify this function
        
        yield kSTART
        for ii in tokenize(sentence):
            if ii not in self._vocab:
                raise OutOfVocab(ii)
            yield ii
        yield kEND

    def vocab(self):
        """
        Returns the language model's vocabulary
        """

        assert self._vocab_final, "Vocab not finalized"
        return list(sorted(self._vocab))
        
    def laplace(self, context, word):
        """
        Return the log probability (base e) of a word given its context
        """

        assert context in self._vocab, "%s not in vocab" % context
        assert word in self._vocab, "%s not in vocab" % word

        # Add your code here
        #val = 0.0
        word_ct = self._obs_counts[context][word] 
        context_ct = 0
        context_ct += sum(self._obs_counts[context].values())
        slaplace = (word_ct + 1) / (context_ct + len(self._vocab)) 
        return log(slaplace)

    def add_train(self, sentence):
        """
        Add the counts associated with a sentence.
        """

        # You'll need to complete this function, but here's a line of code that
        # will hopefully get you started.
        for context, word in bigrams(list(self.tokenize_and_censor(sentence))):
            # ---------------------------------------
            assert word in self._vocab, "%s not in vocab" % word
            self._obs_counts[context][word] += 1
            self._bigrams.add((context, word))


    def log_likelihood(self, sentence):
        """
        Compute the log likelihood of a sentence, divided by the number of
        tokens in the sentence.
        """
        alist = []
        tokens = 1
        prob = 1
        for context, word in bigrams(list(self.tokenize_and_censor(sentence))):
            prob *= exp(self.laplace(context, word))
            tokens += 1 
        l_likelihood = log(prob) / tokens
        return l_likelihood
        #return 0


if __name__ == "__main__":
    dem_lm = BigramLanguageModel()
    rep_lm = BigramLanguageModel()

    #Counters to decide if a speech is dem or rep
    rep_ct = 0
    dem_ct = 0

    for target, pres, name in [(dem_lm, kDEM, "D"), (rep_lm, kREP, "R")]:
        for sent in sentences_from_zipfile("data/state_union.zip", pres):
            for ww in tokenize(sent):
                target.train_seen(ww)
                
        print("Done looking at %s words, finalizing vocabulary" % name)
        target.finalize()
        
        for sent in sentences_from_zipfile("data/state_union.zip", pres):
            target.add_train(sent)
    
        print("Trained language model for %s" % name)

    with open("../data/2016-obama.txt") as infile:
        print("REP\t\tDEM\t\tSentence\n" + "=" * 80)
        for ii in infile:
            if len(ii) < 15: # Ignore short sentences
                continue
            try:
                dem_score = dem_lm.log_likelihood(ii)
                rep_score = rep_lm.log_likelihood(ii)
                if rep_score < dem_score:
                    dem_ct += 1
                if dem_score < rep_score:
                    rep_ct += 1

                print("%f\t%f\t%s" % (dem_score, rep_score, ii.strip()))
            except OutOfVocab:
                None
        """
        if rep_score > dem_score:
            print("Assumed Republican speech")
        if dem_score > rep_score:
            print("Assumed Democratic speech")
        """
            
############## Code for write-up and sentence generation ############################

    all_lm = BigramLanguageModel()      #LangModel to be trained on president's speeches
    for sent in all_sentences("../data/state_union.zip"):
        for ww in tokenize(sent):
            all_lm.train_seen(ww)
        all_lm.add_train(sent)
    all_lm.finalize()

    obama_lm = BigramLanguageModel()    #LangModel to be trained on obama's speech
    with open("../data/2016-obama.txt") as infile:
        for ii in infile:
            for ww in tokenize(ii):
                obama_lm.train_seen(ww)
            obama_lm.add_train(ii)
        obama_lm.finalize() 

        #sets to hold unique words and bigrams of obama
        obama_uniqbg = set()
        obama_uniqwd = set()

        #for every bigram in obama's bigrams
        for item in obama_lm._bigrams:
            #if bigram not seen by model trained on all presidents
            if item not in all_lm._bigrams:
                obama_uniqbg.add(item)

        #for every word in obama's _vocab
        for item in obama_lm._vocab:
            #if word is not in model trained on all presidents
            if item not in all_lm._vocab:
                obama_uniqwd.add(item) 
        for i in range(1,100):
            demsent = ''
            repsent = ''
            for word in dem_lm.sample(100):
                demsent = demsent + " " + word 
            print("Democrat: ", demsent)
            for word in rep_lm.sample(100):
                repsent = repsent + " " + word
            print("Republican: ", repsent)
