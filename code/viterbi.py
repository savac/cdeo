import copy
import numpy as np

class hmmClass(object):
    def __init__(self, tProb, eProb, tw, ew):
        self.tProb=tProb
        self.eProb=eProb
        self.tw=tw
        self.ew=ew

class Viterbi:
    trell = []
    def __init__(self, hmm, words, labels):
        '''Inputs:
        labels - a list of lists of labels with each sublist corresponding to a node
        '''
        self.trell = []
        for iw in range(len(words)):
            word = words[iw]
            temp = {}
            for label in labels[iw]:
                temp[label] = [0,None]
            self.trell.append([word,copy.deepcopy(temp)])
        self.fill_in(hmm)

    def fill_in(self,hmm):
        for i in range(len(self.trell)):
            for token in self.trell[i][1]:
                word = self.trell[i][0]
                if i == 0:
                    self.trell[i][1][token][0] = np.dot(hmm.eProb[(word, token)], hmm.ew)
                else:
                    max = None
                    guess = None
                    c = None
                    prev_word = self.trell[i-1][0]
                    for k in self.trell[i-1][1]:
                        c = self.trell[i-1][1][k][0] + np.dot(hmm.tProb[((prev_word, k), (word, token))], hmm.tw)
                        if max == None or c > max:
                            max = c
                            guess = k
                    max += np.dot(hmm.eProb[(word, token)], hmm.ew)
                    self.trell[i][1][token][0] = max
                    self.trell[i][1][token][1] = guess

    def return_max(self):
        tokens = []
        token = None
        for i in range(len(self.trell)-1,-1,-1):
            if token == None:
                max = None
                guess = None
                for k in self.trell[i][1]:
                    if max == None or self.trell[i][1][k][0] > max:
                        max = self.trell[i][1][k][0]
                        token = self.trell[i][1][k][1]
                        guess = k
                tokens.append(guess)
            else:
                tokens.append(token)
                token = self.trell[i][1][token][1]
        tokens.reverse()
        return tokens
