# -*- coding: utf-8 -*-

import sys
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(1)
#log = open("BILSTM+CRF.log", "w")
#####################################################################
# Helper functions to make the code more readable.


def to_scalar(var):
    # returns a python float
    return var.view(-1).data.tolist()[0]


def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return to_scalar(idx)


def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    tensor = torch.LongTensor(idxs)
    return autograd.Variable(tensor)


# Compute log sum exp in a numerically stable way for the forward algorithm
def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + \
        torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))

#####################################################################
# Create model


class BiLSTM_CRF(nn.Module):

    def __init__(self, dropout_p, vocab_size, lemma_size, pretrain_embeddings, tag_to_ix, word_embedding_dim, pretrain_embedding_dim, lemma_embedding_dim, input_dim, hidden_dim):
        super(BiLSTM_CRF, self).__init__()
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.lemma_size = lemma_size

        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)

        self.word_embeds = nn.Embedding(vocab_size, word_embedding_dim)
        self.word_embeds.weight.requires_grad
        self.lemma_embeds = nn.Embedding(lemma_size, lemma_embedding_dim)
        self.pretrain_embeds = nn.Embedding(pretrain_embeddings.size(0), pretrain_embeddings.size(1))
        self.pretrain_embeds.weight = nn.Parameter(pretrain_embeddings, False)
        self.dropout = nn.Dropout(dropout_p)

        self.emb2input = nn.Linear(word_embedding_dim + lemma_embedding_dim + pretrain_embedding_dim, input_dim)
        self.tanh = nn.Tanh()

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=1, bidirectional=True)

        # Maps the output of the LSTM into tag space.
        self.hidden2tag = nn.Linear(hidden_dim*2, self.tagset_size)

        # Matrix of transition parameters.  Entry i,j is the score of
        # transitioning *to* i *from* j.
        self.transitions = nn.Parameter(torch.randn(self.tagset_size, self.tagset_size))

        # These two statements enforce the constraint that we never transfer
        # to the start tag and we never transfer from the stop tag 
        self.transitions.data[tag_to_ix[START_TAG], :] = -10000
        self.transitions.data[:, tag_to_ix[STOP_TAG]] = -10000

        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (autograd.Variable(torch.randn(2, 1, self.hidden_dim)),
                autograd.Variable(torch.randn(2, 1, self.hidden_dim)))

    def _forward_alg(self, feats):
        # Do the forward algorithm to compute the partition function
        init_alphas = torch.Tensor(1, self.tagset_size).fill_(-10000.)
        # START_TAG has all of the score.
        init_alphas[0][self.tag_to_ix[START_TAG]] = 0.

        # Wrap in a variable so that we will get automatic backprop
        forward_var = autograd.Variable(init_alphas)

        # Iterate through the sentence
        for feat in feats:
            alphas_t = []  # The forward variables at this timestep
            for next_tag in range(self.tagset_size):
                # broadcast the emission score: it is the same regardless of
                # the previous tag
                emit_score = feat[next_tag].view(
                    1, -1).expand(1, self.tagset_size)
                # the ith entry of trans_score is the score of transitioning to
                # next_tag from i
                trans_score = self.transitions[next_tag].view(1, -1)
                # The ith entry of next_tag_var is the value for the
                # edge (i -> next_tag) before we do log-sum-exp
                next_tag_var = forward_var + trans_score + emit_score
                # The forward variable for this tag is log-sum-exp of all the
                # scores.
                alphas_t.append(log_sum_exp(next_tag_var))
            forward_var = torch.cat(alphas_t).view(1, -1)
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        alpha = log_sum_exp(terminal_var)
        return alpha

    def _get_lstm_features(self, sentence):
        self.hidden = self.init_hidden()
        combined_word_embeds = []
        for item in sentence[0]:
            tmp_embeds = self.word_embeds(item).view(len(item), -1)
            combined_word_embeds.append(torch.sum(tmp_embeds,0))
        word_embeds = torch.cat(combined_word_embeds)

        combined_pretrain_embeds = []
        for item in sentence[1]:
            tmp_embeds = self.pretrain_embeds(item).view(len(item), -1)
            combined_pretrain_embeds.append(torch.sum(tmp_embeds,0))
        pretrain_embeds = torch.cat(combined_pretrain_embeds)

        combined_lemma_embeds = []
        for item in sentence[2]:
            tmp_embeds = self.lemma_embeds(item).view(len(item), -1)
            combined_lemma_embeds.append(torch.sum(tmp_embeds,0))
        lemma_embeds = torch.cat(combined_lemma_embeds)

        
        embeds = self.tanh(self.emb2input(torch.cat((word_embeds, pretrain_embeds, lemma_embeds), 1))).view(len(sentence[0]), 1, -1)
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        lstm_out = lstm_out.view(len(sentence[0]), self.hidden_dim*2)
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats

    def _score_sentence(self, feats, tags):
        # Gives the score of a provided tag sequence
        score = autograd.Variable(torch.Tensor([0]))
        tags = torch.cat([torch.LongTensor([self.tag_to_ix[START_TAG]]), tags])
        for i, feat in enumerate(feats):
            score = score + \
                self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
        score = score + self.transitions[self.tag_to_ix[STOP_TAG], tags[-1]]
        return score

    def _viterbi_decode(self, feats):
        backpointers = []

        # Initialize the viterbi variables in log space
        init_vvars = torch.Tensor(1, self.tagset_size).fill_(-10000.)
        init_vvars[0][self.tag_to_ix[START_TAG]] = 0

        # forward_var at step i holds the viterbi variables for step i-1
        forward_var = autograd.Variable(init_vvars)
        for feat in feats:
            bptrs_t = []  # holds the backpointers for this step
            viterbivars_t = []  # holds the viterbi variables for this step

            for next_tag in range(self.tagset_size):
                # next_tag_var[i] holds the viterbi variable for tag i at the
                # previous step, plus the score of transitioning
                # from tag i to next_tag.
                # We don't include the emission scores here because the max
                # does not depend on them (we add them in below)
                next_tag_var = forward_var + self.transitions[next_tag]
                best_tag_id = argmax(next_tag_var)
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id])
            # Now add in the emission scores, and assign forward_var to the set
            # of viterbi variables we just computed
            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
            backpointers.append(bptrs_t)

        # Transition to STOP_TAG
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        best_tag_id = argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]

        # Follow the back pointers to decode the best path.
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        # Pop off the start tag (we dont want to return that to the caller)
        start = best_path.pop()
        assert start == self.tag_to_ix[START_TAG]  # Sanity check
        best_path.reverse()
        return path_score, best_path

    def neg_log_likelihood(self, sentence, tags):
        feats = self._get_lstm_features(sentence)
        forward_score = self._forward_alg(feats)
        gold_score = self._score_sentence(feats, tags)
        return forward_score - gold_score

    def forward(self, sentence):  # dont confuse this with _forward_alg above.
        # Get the emission scores from the BiLSTM
        lstm_feats = self._get_lstm_features(sentence)
        # Find the best path, given the features.
        score, tag_seq = self._viterbi_decode(lstm_feats)
        return score, tag_seq

def evaluate(instances, model):
#    log.write("evaluation start...\n")
    print "evaluation start..."
    right = 0.
    total = 0.
    idx = 0
    for instance in instances:
        idx += 1
        if idx % 100 == 0:
#            log.write(str(idx)+"\n")
            print idx
        packed_instance = []
        packed_instance.append([autograd.Variable(torch.LongTensor(x)) for x in instance[0]])
        packed_instance.append([autograd.Variable(torch.LongTensor(x)) for x in instance[1]])
        packed_instance.append([autograd.Variable(torch.LongTensor(x)) for x in instance[2]])

        _, tag_seq = model(packed_instance)
        assert len(tag_seq) == len(instance[-1])

        for i in range(len(tag_seq)):
            if tag_seq[i] == instance[-1][i]:
                right += 1
        total += len(tag_seq)
    return right/total


#####################################################################
# Run training
trn_filename = "gmb.tags"
#trn_filename = "gmb.tags.part"
dev_filename = "pmb.tags"
#dev_filename = "pmb.tags.part"
tst_filename = "pmb.tags"
#tst_filename = "pmb.tags.part"
pretrain_filename = "sskip.100.vectors"
#pretrain_filename = "sskip.100.vectors.part"
START_TAG = "<START>"
STOP_TAG = "<STOP>"
UNK = "<UNK>"
WORD_EMBEDDING_DIM = 64
PRETRAIN_EMBEDDING_DIM = 100
LEMMA_EMBEDDING_DIM = 32
INPUT_DIM = 50
HIDDEN_DIM = 100


from utils import readfile
from utils import readpretrain
from utils import data2instance
from utils import data2instance_orig
#########################################################
# Load training data
trn_data = readfile(trn_filename)
word_to_ix = {UNK:0}
lemma_to_ix = {UNK:0}
tag_to_ix = {}
for sentence, _, lemmas, tags in trn_data:
    for words in sentence:
        words = words.split("~")
        for word in words:
            if word not in word_to_ix:
                word_to_ix[word] = len(word_to_ix)
    for tag in tags:
        if tag not in tag_to_ix:
            tag_to_ix[tag] = len(tag_to_ix)
    for lemma in lemmas:
        lemma = lemma.split("~")
        for lem in lemma:
            if lem not in lemma_to_ix:
                lemma_to_ix[lem] = len(lemma_to_ix)


##########################################################
# Load pretrain
pretrain_to_ix = {UNK:0}
pretrain_embeddings = [ [0. for i in range(100)]]
pretrain_data = readpretrain(pretrain_filename)
for one in pretrain_data:
    pretrain_to_ix[one[0]] = len(pretrain_to_ix)
    pretrain_embeddings.append([float(a) for a in one[1:]])
#log.write("pretrain dict size: " + str(len(pretrain_to_ix))+"\n")
print "pretrain dict size: " + str(len(pretrain_to_ix))
###########################################################
# Load dev data
dev_data = readfile(dev_filename)
for sentence, _, lemmas, tags in dev_data:
    for tag in tags:
        if tag not in tag_to_ix:
            tag_to_ix[tag] = len(tag_to_ix)
###########################################################
# Load tst data
tst_data = readfile(tst_filename)
for sentence, _, lemmas, tags in tst_data:
    for tag in tags:
        if tag not in tag_to_ix:
            tag_to_ix[tag] = len(tag_to_ix)

tag_to_ix[START_TAG] = len(tag_to_ix)
tag_to_ix[STOP_TAG] = len(tag_to_ix)

#log.write("word dict size: " + str(len(word_to_ix))+"\n")
#log.write("lemma dict size: " + str(len(lemma_to_ix))+"\n")
#log.write("tag dict size: "+ str(len(tag_to_ix))+"\n")

print "word dict size: " + str(len(word_to_ix))
print "lemma dict size: " + str(len(lemma_to_ix))
print "tag dict size: "+ str(len(tag_to_ix))

model = BiLSTM_CRF(0.3, len(word_to_ix), len(lemma_to_ix), torch.FloatTensor(pretrain_embeddings), tag_to_ix, WORD_EMBEDDING_DIM, PRETRAIN_EMBEDDING_DIM, LEMMA_EMBEDDING_DIM, INPUT_DIM, HIDDEN_DIM)
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001, weight_decay=1e-4)

###########################################################
# prepare training instance
trn_instances = data2instance_orig(trn_data, [(word_to_ix,0), (pretrain_to_ix,0), (lemma_to_ix,0), (tag_to_ix,-1)])
#log.write("trn size: " + str(len(trn_instances))+"\n")
print "trn size: " + str(len(trn_instances))
###########################################################
# prepare development instance
dev_instances = data2instance_orig(dev_data, [(word_to_ix,0), (pretrain_to_ix,0), (lemma_to_ix,0), (tag_to_ix,-1)])
#log.write("dev size: " + str(len(dev_instances))+"\n")
print "dev size: " + str(len(dev_instances))
###########################################################
# prepare test instance
tst_instances = data2instance_orig(tst_data, [(word_to_ix,0), (pretrain_to_ix,0), (lemma_to_ix,0), (tag_to_ix,-1)])
#log.write("tst size: " + str(len(tst_instances))+"\n")
print "tst size: " + str(len(tst_instances))
# Check predictions before training
#log.write("DEV accuracy= " + str(evaluate(dev_instances, model))+"\n")
print "DEV accuracy= " + str(evaluate(dev_instances, model))
#log.write("TST accuracy= " + str(evaluate(tst_instances, model))+"\n")
#log.flush()

total_loss = 0
inst_idx = -1
timestep = 0

trn_see = 1000
eval_see = 10
#print(model.pretrain_embeds(autograd.Variable(torch.LongTensor([7]))))
#print(model.word_embeds(autograd.Variable(torch.LongTensor([1]))))
while True:
    timestep += 1
    inst_idx += 1
    if inst_idx == len(trn_instances):
        inst_idx = 0

    model.zero_grad()

    packed_instance = []
    packed_instance.append([autograd.Variable(torch.LongTensor(x)) for x in trn_instances[inst_idx][0]])
    packed_instance.append([autograd.Variable(torch.LongTensor(x)) for x in trn_instances[inst_idx][1]])
    packed_instance.append([autograd.Variable(torch.LongTensor(x)) for x in trn_instances[inst_idx][2]])
    packed_instance.append(torch.LongTensor(trn_instances[inst_idx][3]))

    #print(packed_instance)
    #print(model.pretrain_embeds(autograd.Variable(torch.LongTensor([7]))))
    #print(model.word_embeds(autograd.Variable(torch.LongTensor([1]))))

    neg_log_likelihood = model.neg_log_likelihood(packed_instance, packed_instance[-1])
    total_loss += neg_log_likelihood
    neg_log_likelihood.backward()
    optimizer.step()

    #print(model.pretrain_embeds(autograd.Variable(torch.LongTensor([7]))))
    #print(model.word_embeds(autograd.Variable(torch.LongTensor([1]))))
    #exit(1)
    if timestep % trn_see == 0:
        #log.write("epoch: "+str(timestep*1.0/len(trn_instances)) + " loss: " + str(to_scalar(total_loss)/trn_see)+"\n")
        print "epoch: "+str(timestep*1.0/len(trn_instances)) + " loss: " + str(to_scalar(total_loss)/trn_see)
        total_loss = 0
        #print(model.word_embeds(autograd.Variable(torch.LongTensor([1]))))
        #print(model.pretrain_embeds(autograd.Variable(torch.LongTensor([7]))))
        #log.flush()

    if timestep % (trn_see*eval_see) == 0:
        #log.write("DEV accuracy= " + str(evaluate(dev_instances, model))+"\n")
        print "DEV accuracy= " + str(evaluate(dev_instances, model))
#        log.write("TST accuracy= " + str(evaluate(tst_instances, model))+"\n")
        torch.save(model.state_dict(), "BILSTM+CRF_models/model."+str(timestep/trn_see/eval_see)+"\n")
        #log.flush()
#log.close()
