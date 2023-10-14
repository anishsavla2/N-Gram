import re
import math
import string
import pandas as pd
import operator
from collections import deque

# used for unseen words in training vocabularies
UNK = "<UNK>"
# sentence start and end
SENTENCE_START = "<START>"
SENTENCE_END = "<STOP>"

#
# def read_sentences_from_file(file_path):
#     with open(file_path, "r") as f:
#         return [re.split("\s+", line.rstrip('\n')) for line in f]

wordDictionary = {}
def read_file(file_path):
    sentenceList = []
    with open(file_path, "r") as f:
        for line in f:
            words = line.rstrip('\n').split(" ")
            sentenceList.append(words)

        for sent in sentenceList:
            word_with_brackets = ""
            for s in sent:
                if s in wordDictionary.keys():
                    wordDictionary[s] += 1
                else:
                    wordDictionary[s] = 1
        return sentenceList


def replace_Unknown_Word(fileName):

        fin = open(fileName,"rt")
        fout = open(fileName+"_"+"out.txt", "wt")
        special_char_list = ["(", ")", "?", "[", "*", "!", ","]
        for line in fin:
            newLine = line
            for word in line.split():
                if word in wordDictionary.keys() and wordDictionary.get(word) < 3:
                    if word == ".":
                        newLine = re.sub(r"\\{}".format(word), '<UNK>', newLine)
                    elif word in special_char_list:
                        newLine = re.sub(r'\w[{}]'.format(word), '<UNK>', newLine)
                    else:
                        newLine = re.sub(r"\b{}\b".format(word), '<UNK>', newLine)
            fout.write(newLine)




        fin.close()
        fout.close()

class UnigramLanguageModel:
    def __init__(self, sentences , defaultLambdaSet, smoothing_value ,smoothing=False):
        print("I am Here")
        #print(sentences)
        self.defaultLambdaSet = defaultLambdaSet
        self.smoothing_Value = smoothing_value
        self.unigram_frequencies = dict()
        self.corpus_length = 0
        #print(sentences)
        for sentence in sentences:
            # #print(sentence)
            # if sentence[0] == SENTENCE_START:
            #     pass
            # else:
            #     sentence.insert(0, SENTENCE_START)
            #     sentence.append(SENTENCE_END)
            #print(sentence)
            for word in sentence:
                self.unigram_frequencies[word] = self.unigram_frequencies.get(word, 0) + 1
                if word != SENTENCE_START and word != SENTENCE_END:
                    self.corpus_length += 1
        # subtract 2 because unigram_frequencies dictionary contains values for SENTENCE_START and SENTENCE_END
        self.unique_words = len(self.unigram_frequencies) - 2
        self.smoothing = smoothing

    def calculate_unigram_probability(self, word):

        word_probability_numerator = self.unigram_frequencies.get(word, 0)
        word_probability_denominator = self.corpus_length
        if self.smoothing == 0:
             word_probability_numerator += self.smoothing_Value

                # add one more to total number of seen unique words for UNK - unseen events
             #word_probability_denominator += self.unique_words + 1
        else:
            word_probability_numerator += self.smoothing_Value
            #print("word_probability_numerator : ", word_probability_numerator)
                # add one more to total number of seen unique words for UNK - unseen events
            word_probability_denominator += self.unique_words + 1
            #print("word_probability_denominator : ",word_probability_denominator)
        #print("Probab : ",float(word_probability_numerator) / float(word_probability_denominator))
        return float(word_probability_numerator) / float(word_probability_denominator)

    def calculate_sentence_probability(self, sentence, normalize_probability=True):

        sentence_probability_log_sum = 0
        for word in sentence:

            if word != SENTENCE_START and word != SENTENCE_END:
                if not self.defaultLambdaSet:
                    word_probability = self.calculate_unigram_probability(word)


                else:
                    word_probability = self.defaultLambdaSet[0] * self.calculate_unigram_probability(word)
                sentence_probability_log_sum += math.log(word_probability, 2)
                #print("sentence_probability_log_sum : ",sentence_probability_log_sum)
        #print("Unigram sentence probab : ",math.pow(2, sentence_probability_log_sum))
        return math.pow(2, sentence_probability_log_sum) if normalize_probability else sentence_probability_log_sum

    def sorted_vocabulary(self):
        full_vocab = list(self.unigram_frequencies.keys())
        #full_vocab.remove(SENTENCE_START)
        # full_vocab.remove(SENTENCE_END)
        full_vocab.sort()
        # full_vocab.append(UNK)
        # full_vocab.append(SENTENCE_START)
        # full_vocab.append(SENTENCE_END)
        return full_vocab


class BigramLanguageModel(UnigramLanguageModel):
    def __init__(self, sentences, defaultLambdaSet, smoothing_value, smoothing=False):
        UnigramLanguageModel.__init__(self, sentences,defaultLambdaSet, smoothing_value, smoothing)
        self.defaultLambdaSet = defaultLambdaSet
        self.smoothing_Value = smoothing_value
        print("I am here in Bigram")
        self.bigram_frequencies = dict()
        self.unique_bigrams = set()
        for sentence in sentences:
            # print(sentence)
            # print("Hello")
            # if sentence[0] == SENTENCE_START:
            #     pass
            # else:
            #     sentence.insert(0, SENTENCE_START)
            #     sentence.append(SENTENCE_END)
            previous_word = None
            for word in sentence:

                if previous_word != None:
                    self.bigram_frequencies[(previous_word, word)] = self.bigram_frequencies.get((previous_word, word),
                                                                                                 0) + 1
                    if previous_word != SENTENCE_START and word != SENTENCE_END:
                        self.unique_bigrams.add((previous_word, word))
                previous_word = word
        # we subtracted two for the Unigram model as the unigram_frequencies dictionary
        # contains values for SENTENCE_START and SENTENCE_END but these need to be included in Bigram
        self.unique__bigram_words = len(self.unigram_frequencies)

        #print("Unigram Freq are : ", self.unigram_frequencies)

    def calculate_bigram_probabilty(self, previous_word, word):
        bigram_word_probability_numerator = self.bigram_frequencies.get((previous_word, word), 0)
        bigram_word_probability_denominator = self.unigram_frequencies.get(previous_word, 0)
        # print("Prev word: ", previous_word)
        # print("Current word: ", word)
        #print("bigram_word_probability_numerator : ",bigram_word_probability_numerator)
        #print("bigram_word_probability_denominator : ",bigram_word_probability_denominator)
        #print("Unigram Freq : ", self.unigram_frequencies.get('<START>'))
        if self.smoothing:
            if self.smoothing_value != 0:
                bigram_word_probability_numerator += self.smoothing_value
                bigram_word_probability_denominator += self.unique__bigram_words
        #bigram_word_probability_denominator += self.unique__bigram_words
        #print("Bigram freq : ", self.bigram_frequencies.get(('to', 'be')))
        return 0.0 if bigram_word_probability_numerator == 0 or bigram_word_probability_denominator == 0 else float(
            bigram_word_probability_numerator) / float(bigram_word_probability_denominator)

    def calculate_bigram_sentence_probability(self, sentence, normalize_probability=True):
        bigram_sentence_probability_log_sum = 0
        previous_word = None
        #print(sentence)
        for word in sentence:

            if previous_word != None:
                if not self.defaultLambdaSet:
                    bigram_word_probability =  self.calculate_bigram_probabilty(previous_word, word)
                    #print ("if bigram_word_probability {}".format(bigram_word_probability))
                else:
                    bigram_word_probability = self.defaultLambdaSet[1] * self.calculate_bigram_probabilty(previous_word, word)
                    #print ("else bigram_word_probability {}".format(bigram_word_probability))

                #print("Word probability is : ", bigram_word_probability)
                bigram_sentence_probability_log_sum += math.log(bigram_word_probability, 2)

            previous_word = word
        #print("Bigram sentence probab : ", math.pow(2,bigram_sentence_probability_log_sum))
        #print("Bigram freq : ", self.bigram_frequencies.get(('<START>', 'Having')))
        return math.pow(2,bigram_sentence_probability_log_sum) if normalize_probability else bigram_sentence_probability_log_sum


class TrigramLanguageModel(BigramLanguageModel):
    def __init__(self, sentences, defaultLambdaSet, smoothing_value, smoothing=False):
        for sentence in sentences:
            if sentence[0] == SENTENCE_START:
                pass
            else:
                sentence.insert(0, SENTENCE_START)
                sentence.append(SENTENCE_END)
        BigramLanguageModel.__init__(self, sentences, defaultLambdaSet, smoothing_value, smoothing)
        self.defaultLambdaSet = defaultLambdaSet
        self.smoothing_value = smoothing_value
        #print("I am here")
        self.trigram_frequencies = dict()
        self.unique_trigrams = set()

        for sentence in sentences:
            previous_word = None
            previous_second_word = None
            for word in sentence:
                if previous_word != None and previous_second_word != None:
                    if (previous_second_word,previous_word, word) in self.trigram_frequencies.keys():
                        self.trigram_frequencies[(previous_second_word, previous_word, word)]+=1
                        # if previous_second_word == "to" and previous_word == "be" and word == "announced":
                            #print ("i am in if")

                    else:
                        self.trigram_frequencies[(previous_second_word, previous_word, word)] = 1


                    #self.trigram_frequencies[(previous_second_word,previous_word, word)] = self.trigram_frequencies.get(( v, 0) + 1
                # if previous_second_word == "to" and previous_word == "be" and word == "announced" :
                #     #print(sentence)
                #     print("frequency : {}".format(self.trigram_frequencies[(previous_second_word,previous_word, word)]))
                #if previous_second_word == None:
                    #previous_second_word = word
                #if previous_second_word != None and previous_word != None:
                previous_second_word = previous_word
                previous_word = word
                #else:
                    #previous_word = word
            self.unique_trigram_words = len(self.trigram_frequencies)
        #print("Trigram Frequencies : ", self.trigram_frequencies.get(('to', 'be', 'announced')))
    def calculate_trigram_probabilty(self, previous_second_word, previous_word, word):

        #print ("{} {} {}".format(previous_second_word, previous_word, word))
        trigram_word_probability_numerator = self.trigram_frequencies.get((previous_second_word, previous_word, word), 0)
        trigram_word_probability_denominator = self.bigram_frequencies.get((previous_second_word, previous_word), 0)
        #print(self.bigram_frequencies.get(('to', 'be')))
        #print("trigram_word_probability_numerator : ",trigram_word_probability_numerator)
        #print("trigram_word_probability_denominator : ",trigram_word_probability_denominator)

        if self.smoothing == True:
            if self.smoothing_value != 0:

                trigram_word_probability_numerator += self.smoothing_value
                trigram_word_probability_denominator += self.unique_trigram_words


        # if previous_second_word == "to" and previous_word == "be" and word == "announced":
        #     print("trigram_word_probability_numerator : ",trigram_word_probability_numerator)
        #     print("trigram_word_probability_denominator : ", trigram_word_probability_denominator)


        return 0.0 if trigram_word_probability_numerator == 0 or trigram_word_probability_denominator == 0 else float(
            trigram_word_probability_numerator) / float(trigram_word_probability_denominator)

    def calculate_trigram_sentence_probability(self, sentence, normalize_probability=True):
        #print ("sentence is {}".format(sentence))
        trigram_sentence_probability_log_sum = 0
        previous_word = None
        previous_second_word = None

        for word in sentence:
            #print(word)
            if previous_word != None and previous_second_word != None:
                if not self.defaultLambdaSet:

                    trigram_word_probability = self.calculate_trigram_probabilty(previous_second_word, previous_word, word)
                    #print("trigram_word_probability : ",trigram_word_probability)
                else:
                    trigram_word_probability = self.defaultLambdaSet[2] * self.calculate_trigram_probabilty(previous_second_word, previous_word,
                                                                                 word)
                trigram_sentence_probability_log_sum += math.log(trigram_word_probability, 2)
                #print("trigram_sentence_probability_log_sum : ", trigram_sentence_probability_log_sum)

            if previous_second_word == None:
                previous_second_word = word
            elif previous_second_word != None and previous_word != None:
                previous_second_word = previous_word
                previous_word = word
            else:
                previous_word = word
        #print("trigram_sentence_probability_log_sum : ", trigram_sentence_probability_log_sum)
        return math.pow(2,
                        trigram_sentence_probability_log_sum) if normalize_probability else trigram_sentence_probability_log_sum


def calculate_number_of_unigrams(sentences):
    unigram_count = 0

    for sentence in sentences:
        unigram_count += len(sentence)
    return unigram_count

def calculate_number_of_bigrams(sentences):
    bigram_count = 0
    for sentence in sentences:
        # remove one for number of bigrams in sentence
        bigram_count += len(sentence) - 1
    return bigram_count

def calculate_number_of_trigrams(sentences):
    trigram_count = 0
    for sentence in sentences:
        # remove one for number of bigrams in sentence
        trigram_count += len(sentence) - 2
    return trigram_count



def calculate_unigram_perplexity(model, sentences):

    unigram_count = calculate_number_of_unigrams(sentences)

    sentence_probability_log_sum = 0
    for sentence in sentences:
        try:

            sentence_probablity = model.calculate_sentence_probability(sentence)
            individual_sentence_probability = math.log(sentence_probablity,2)
            sentence_probability_log_sum -= individual_sentence_probability
            if math.isinf(sentence_probability_log_sum):
                return 0
        except Exception as e:
            sentence_probability_log_sum -= 0
    return math.pow(2, sentence_probability_log_sum / unigram_count)

def calculate_bigram_perplexity(model, sentences):
    number_of_bigrams = calculate_number_of_bigrams(sentences)
    bigram_sentence_probability_log_sum = 0
    for sentence in sentences:
        try:
            bigram_sentence_probability_log_sum -= math.log(model.calculate_bigram_sentence_probability(sentence),2)

        except:
            bigram_sentence_probability_log_sum -= 0
    return math.pow(2, bigram_sentence_probability_log_sum / number_of_bigrams)

def calculate_trigram_perplexity(model, sentences):
    number_of_trigrams = calculate_number_of_trigrams(sentences)
    trigram_sentence_probability_log_sum = 0
    for sentence in sentences:
        try:
            trigram_sentence_probability_log_sum -= math.log(model.calculate_trigram_sentence_probability(sentence), 2)
        except:
            print("In Exception")

            trigram_sentence_probability_log_sum -= 0
    return math.pow(2, trigram_sentence_probability_log_sum / number_of_trigrams)

# print unigram and bigram probs
def print_unigram_probs(sorted_vocab_keys, model):
    for vocab_key in sorted_vocab_keys:
        if vocab_key != SENTENCE_START and vocab_key != SENTENCE_END:
            print("{}: {}".format(vocab_key if vocab_key != UNK else "UNK",
                                       model.calculate_unigram_probability(vocab_key)), end=" ")
    print("")

def print_bigram_probs(sorted_vocab_keys, model):
    print("\t\t", end="")
    # for vocab_key in sorted_vocab_keys:
    #     if vocab_key != SENTENCE_START:
    #         print(vocab_key if vocab_key != UNK else "UNK", end="\t\t")
    # print("")
    for vocab_key in sorted_vocab_keys:

        if vocab_key != SENTENCE_END:
            #print(vocab_key if vocab_key != UNK else "UNK", end="\t\t")
            for vocab_key_second in sorted_vocab_keys:

                if vocab_key_second != SENTENCE_START:
                    #print("{0:.5f}".format(model.calculate_bigram_probabilty(vocab_key, vocab_key_second)), end="\t\t")
                    print(model.calculate_bigram_probabilty(vocab_key, vocab_key_second))#, end="\t\t")

            print("")
    print("")


if __name__ == '__main__':
    trainingFile = '1b_benchmark.train.tokens'
    datasetTraining = read_file(trainingFile)
    defaultLambdaSet = []
    replace_Unknown_Word(trainingFile)
    datasetNewTraining = read_file('1b_benchmark.train.tokens_out.txt')
    dataset_model_training = TrigramLanguageModel(datasetNewTraining, defaultLambdaSet, smoothing_value=0)
    #dataset_model_smoothed_training = TrigramLanguageModel(datasetNewTraining, defaultLambdaSet, smoothing_value=1, smoothing=True)

    devFile = '1b_benchmark.dev.tokens'
    datasetDev = read_file(devFile)
    replace_Unknown_Word(devFile)
    #datasetNewDev = read_file('1b_benchmark.dev.tokens_out.txt')
    #dataset_model_dev = TrigramLanguageModel(datasetNewDev, defaultLambdaSet, smoothing_value=0)
    #dataset_model_smoothed_dev = TrigramLanguageModel(datasetNewDev, defaultLambdaSet, smoothing_value=1, smoothing=True)
    #sorted_vocab = data.sorted_vocabulary()

    testFile = '1b_benchmark.test.tokens'
    datasetTest = read_file(testFile)
    replace_Unknown_Word(testFile)
    datasetNewTest = read_file('1b_benchmark.test.tokens_out.txt')
    #dataset_model_test = TrigramLanguageModel(datasetNewTest, defaultLambdaSet, smoothing_value=0)
    #dataset_model_smoothed_test = TrigramLanguageModel(datasetNewTest, defaultLambdaSet, smoothing_value=1, smoothing=True)
    #sorted_vocab = dataset_model_smoothed_test.sorted_vocabulary()

    # Question 2 - Training Data Unsmoothed Perplexity
    # print("== Training Data Unsmoothed PERPLEXITY == ")
    # print("Training Data Perplexity for - unigram : ", calculate_unigram_perplexity(dataset_model_training, datasetNewTraining))
    # print("Training Data Perplexity for - bigram: ", calculate_bigram_perplexity(dataset_model_training, datasetNewTraining))
    # print("Training Data Perplexity for - trigram: ", calculate_trigram_perplexity(dataset_model_training,datasetNewTraining))

    #QUESTION 3 - 1 - Calculating Perplexity for alpha = 1 on training data
    #dataset_training_model_smoothed = TrigramLanguageModel(datasetNewTraining, defaultLambdaSet,smoothing_value=1, smoothing=True)
    print("======== Perplexity for Training Data with Smoothing value = 1 ==========")
    print("unigram smoothed Perplexity : ", calculate_unigram_perplexity(dataset_model_training, datasetNewTraining))
    print("bigram smoothed Perplexity : ", calculate_bigram_perplexity(dataset_model_training, datasetNewTraining))
    print("trigram smoothed Perplexity : ", calculate_trigram_perplexity(dataset_model_training,datasetNewTraining))


    # # Question 2 - Dev Data Unsmoothed Perplexity
    # print("== Dev Data Unsmoothed PERPLEXITY == ")
    # print("Dev Data Perplexity for - unigram : ",
    #       calculate_unigram_perplexity(dataset_model_dev, datasetNewDev))
    # print("Dev Data Perplexity for - bigram: ",
    #       calculate_bigram_perplexity(dataset_model_dev, datasetNewDev))
    # print("Dev Data Perplexity for - trigram: ",
    #       calculate_trigram_perplexity(dataset_model_dev, datasetNewDev))
    #
    # # Question 2 - Test Data Unsmoothed Perplexity
    # print("== Dev Data Unsmoothed PERPLEXITY == ")
    # print("Test Data Perplexity for - unigram : ",
    #       calculate_unigram_perplexity(dataset_model_test, datasetNewTest))
    # print("Test Data Perplexity for - bigram: ",
    #       calculate_bigram_perplexity(dataset_model_test, datasetNewTest))
    # print("Test Data Perplexity for - trigram: ",
    #       calculate_trigram_perplexity(dataset_model_test, datasetNewTest))
    #
    # #QUESTION 3 - 1 - Calculating Perplexity for alpha = 1 on training data
    # dataset_training_model_smoothed = TrigramLanguageModel(datasetNewTraining, defaultLambdaSet,
    #                                                        smoothing_value=1, smoothing=True)
    # print("======== Perplexity for Training Data with Smoothing value = 1 ==========")
    # print("unigram smoothed Perplexity : ", calculate_unigram_perplexity(dataset_training_model_smoothed, datasetNewTraining))
    # print("bigram smoothed Perplexity : ", calculate_bigram_perplexity(dataset_training_model_smoothed, datasetNewTraining))
    # print("trigram smoothed Perplexity : ", calculate_trigram_perplexity(dataset_training_model_smoothed,datasetNewTraining))
    #
    # #QUESTION 3 - 2 - Calculating Perplexity for different smoothing value on Training Data
    # dataset_training_model_smoothed = TrigramLanguageModel(datasetNewTraining, defaultLambdaSet,
    #                                               smoothing_value=2, smoothing=True)
    # print("\n======== Perplexity for Training Data with Smoothing value = 2 ==========")
    # print("unigram smoothed Perplexity : ", calculate_unigram_perplexity(dataset_training_model_smoothed, datasetNewTraining))
    # print("bigram smoothed Perplexity : ", calculate_bigram_perplexity(dataset_training_model_smoothed, datasetNewTraining))
    # print("trigram smoothed Perplexity : ", calculate_trigram_perplexity(dataset_training_model_smoothed,datasetNewTraining))
    # #
    # dataset_training_model_smoothed = TrigramLanguageModel(datasetNewTraining, defaultLambdaSet, smoothing_value=3, smoothing=True)
    # print("\n======== Perplexity for Training Data with Smoothing value = 3 ==========")
    # print("unigram smoothed Perplexity : ", calculate_unigram_perplexity(dataset_training_model_smoothed, datasetNewTraining))
    # print("bigram smoothed Perplexity : ", calculate_bigram_perplexity(dataset_training_model_smoothed, datasetNewTraining))
    # print("trigram smoothed Perplexity : ", calculate_trigram_perplexity(dataset_training_model_smoothed, datasetNewTraining))
    # #
    # dataset_training_model_smoothed = TrigramLanguageModel(datasetNewTraining, defaultLambdaSet, smoothing_value=10, smoothing=True)
    # print("\n======== Perplexity for Training Data with Smoothing value = 10 ==========")
    # print("unigram smoothed Perplexity : ", calculate_unigram_perplexity(dataset_training_model_smoothed, datasetNewTraining))
    # print("bigram smoothed Perplexity : ", calculate_bigram_perplexity(dataset_training_model_smoothed, datasetNewTraining))
    # print("trigram smoothed Perplexity : ", calculate_trigram_perplexity(dataset_training_model_smoothed, datasetNewTraining))
    #
    # # QUESTION 3 - 2 - Calculating Perplexity for different smoothing value on Dev Data
    # dataset_model_smoothed_dev = TrigramLanguageModel(datasetNewDev, defaultLambdaSet, smoothing_value=2, smoothing=True)
    # print("\n======== Perplexity for Dev Data with Smoothing value = 2 ==========")
    # print("unigram smoothed Perplexity : ",
    #       calculate_unigram_perplexity(dataset_model_smoothed_dev, datasetNewDev))
    # print("bigram smoothed Perplexity : ",
    #       calculate_bigram_perplexity(dataset_model_smoothed_dev, datasetNewDev))
    # print("trigram smoothed Perplexity : ",
    #       calculate_trigram_perplexity(dataset_model_smoothed_dev, datasetNewDev))
    # #
    # dataset_model_smoothed_dev = TrigramLanguageModel(datasetNewDev, defaultLambdaSet, smoothing_value=3, smoothing=True)
    #
    # print("\n======== Perplexity for Dev Data with Smoothing value = 3 ==========")
    # print("unigram smoothed Perplexity : ",
    #       calculate_unigram_perplexity(dataset_model_smoothed_dev, datasetNewDev))
    # print("bigram smoothed Perplexity : ",
    #       calculate_bigram_perplexity(dataset_model_smoothed_dev, datasetNewDev))
    # print("trigram smoothed Perplexity : ",
    #       calculate_trigram_perplexity(dataset_model_smoothed_dev, datasetNewDev))
    # #
    # dataset_model_smoothed_dev = TrigramLanguageModel(datasetNewDev, defaultLambdaSet, smoothing_value=10, smoothing=True)
    # print("\n======== Perplexity for Dev Data with Smoothing value = 10 ==========")
    # print("unigram smoothed Perplexity : ",
    #       calculate_unigram_perplexity(dataset_model_smoothed_dev, datasetNewDev))
    # print("bigram smoothed Perplexity : ",
    #       calculate_bigram_perplexity(dataset_model_smoothed_dev, datasetNewDev))
    # print("trigram smoothed Perplexity : ",
    #       calculate_trigram_perplexity(dataset_model_smoothed_dev, datasetNewDev))
    #
    # # Question 3- 3 Calculating Test data perplexity with alpha = 10 - which gave best dev data  smoothing perplexity
    #
    # dataset_model_smoothed_test = TrigramLanguageModel(datasetNewTest, defaultLambdaSet, smoothing_value=10,
    #                                                   smoothing=True)
    # print("\n======== Perplexity Test  Data with Smoothing value = 10 ==========")
    # print("unigram smoothed Perplexity : ",
    #       calculate_unigram_perplexity(dataset_model_smoothed_test, datasetNewTest))
    # print("bigram smoothed Perplexity : ",
    #       calculate_bigram_perplexity(dataset_model_smoothed_test, datasetNewTest))
    # print("trigram smoothed Perplexity : ",
    #       calculate_trigram_perplexity(dataset_model_smoothed_test, datasetNewTest))
    #
    # #
    # # =================================== QUESTION 4 ===================================
    #
    # # Reporting Training Data Perplexity for λ1 = 0.1, λ2 = 0.3, λ3 = 0.6. and 5 additional Sets
    # defaultLambdaSet = [0.1,0.3,0.6]
    # print("\n===== Perplexity for Interpolation for training data with values 0.1, 0.3, 0.6=======")
    # dataset_train_model_smoothed = TrigramLanguageModel(datasetNewTraining, defaultLambdaSet, smoothing_value=1, smoothing=True)
    # print("unigram smoothed perplexity with Interpolation: ", calculate_unigram_perplexity(dataset_train_model_smoothed, datasetNewTraining))
    # print("bigram smoothed perplexity with Interpolation: ", calculate_bigram_perplexity(dataset_train_model_smoothed, datasetNewTraining))
    # print("trigram smoothed perplexity with Interpolation:  ", calculate_trigram_perplexity(dataset_train_model_smoothed, datasetNewTraining))
    # #
    # defaultLambdaSet = [0.1, 0.4, 0.5]
    # print("\n===== Perplexity for Interpolation for training data with values 0.1, 0.4, 0.5=======")
    # dataset_train_model_smoothed = TrigramLanguageModel(datasetNewTraining, defaultLambdaSet, smoothing_value=1, smoothing=True)
    # print("unigram smoothed perplexity with Interpolation: ",
    #       calculate_unigram_perplexity(dataset_train_model_smoothed, datasetNewTraining))
    # print("bigram smoothed perplexity with Interpolation: ",
    #       calculate_bigram_perplexity(dataset_train_model_smoothed, datasetNewTraining))
    # print("trigram smoothed perplexity with Interpolation:  ",
    #       calculate_trigram_perplexity(dataset_train_model_smoothed, datasetNewTraining))
    # #
    # defaultLambdaSet = [0.2, 0.2, 0.6]
    # print("\n===== Perplexity for Interpolation for training data with values 0.2, 0.2, 0.6=======")
    # dataset_train_model_smoothed = TrigramLanguageModel(datasetNewTraining, defaultLambdaSet, smoothing_value=1, smoothing=True)
    # print("unigram smoothed perplexity with Interpolation: ",
    #       calculate_unigram_perplexity(dataset_train_model_smoothed, datasetNewTraining))
    # print("bigram smoothed perplexity with Interpolation: ",
    #       calculate_bigram_perplexity(dataset_train_model_smoothed, datasetNewTraining))
    # print("trigram smoothed perplexity with Interpolation:  ",
    #       calculate_trigram_perplexity(dataset_train_model_smoothed, datasetNewTraining))
    # #
    # defaultLambdaSet = [0.2, 0.3, 0.5]
    # print("\n===== Perplexity for Interpolation for training data with values 0.2, 0.3, 0.5=======")
    # dataset_train_model_smoothed = TrigramLanguageModel(datasetNewTraining, defaultLambdaSet, smoothing_value=1, smoothing=True)
    # print("unigram smoothed perplexity with Interpolation: ",
    #       calculate_unigram_perplexity(dataset_train_model_smoothed, datasetNewTraining))
    # print("bigram smoothed perplexity with Interpolation: ",
    #       calculate_bigram_perplexity(dataset_train_model_smoothed, datasetNewTraining))
    # print("trigram smoothed perplexity with Interpolation:  ",
    #       calculate_trigram_perplexity(dataset_train_model_smoothed, datasetNewTraining))
    # #
    # defaultLambdaSet = [0.1, 0.2, 0.7]
    # print("\n===== Perplexity for Interpolation for training data with values 0.1, 0.2, 0.7=======")
    # dataset_train_model_smoothed = TrigramLanguageModel(datasetNewTraining, defaultLambdaSet, smoothing_value=1, smoothing=True)
    # print("unigram smoothed perplexity with Interpolation: ",
    #       calculate_unigram_perplexity(dataset_train_model_smoothed, datasetNewTraining))
    # print("bigram smoothed perplexity with Interpolation: ",
    #       calculate_bigram_perplexity(dataset_train_model_smoothed, datasetNewTraining))
    # print("trigram smoothed perplexity with Interpolation:  ",
    #       calculate_trigram_perplexity(dataset_train_model_smoothed, datasetNewTraining))
    #
    # # Reporting Dev Data Perplexity for λ1 = 0.1, λ2 = 0.3, λ3 = 0.6. and 5 additional Sets
    # defaultLambdaSet = [0.1, 0.3, 0.6]
    # print("\n===== Perplexity for Interpolation for Dev data with values 0.1, 0.3, 0.6=======")
    # dataset_dev_model_smoothed = TrigramLanguageModel(datasetNewDev, defaultLambdaSet, smoothing_value=1,
    #                                                     smoothing=True)
    # print("unigram smoothed perplexity with Interpolation: ",
    #       calculate_unigram_perplexity(dataset_dev_model_smoothed, datasetNewDev))
    # print("bigram smoothed perplexity with Interpolation: ",
    #       calculate_bigram_perplexity(dataset_dev_model_smoothed, datasetNewDev))
    # print("trigram smoothed perplexity with Interpolation:  ",
    #       calculate_trigram_perplexity(dataset_dev_model_smoothed, datasetNewDev))
    # #
    # defaultLambdaSet = [0.1, 0.4, 0.5]
    # print("\n===== Perplexity for Interpolation for Dev data with values 0.1, 0.4, 0.5=======")
    # dataset_dev_model_smoothed = TrigramLanguageModel(datasetNewDev, defaultLambdaSet, smoothing_value=1,
    #                                                   smoothing=True)
    # print("unigram smoothed perplexity with Interpolation: ",
    #       calculate_unigram_perplexity(dataset_dev_model_smoothed, datasetNewDev))
    # print("bigram smoothed perplexity with Interpolation: ",
    #       calculate_bigram_perplexity(dataset_dev_model_smoothed, datasetNewDev))
    # print("trigram smoothed perplexity with Interpolation:  ",
    #       calculate_trigram_perplexity(dataset_dev_model_smoothed, datasetNewDev))
    # #
    # defaultLambdaSet = [0.2, 0.2, 0.6]
    # print("\n===== Perplexity for Interpolation for Dev data with values 0.2, 0.2, 0.6=======")
    # dataset_dev_model_smoothed = TrigramLanguageModel(datasetNewDev, defaultLambdaSet, smoothing_value=1,
    #                                                   smoothing=True)
    # print("unigram smoothed perplexity with Interpolation: ",
    #       calculate_unigram_perplexity(dataset_dev_model_smoothed, datasetNewDev))
    # print("bigram smoothed perplexity with Interpolation: ",
    #       calculate_bigram_perplexity(dataset_dev_model_smoothed, datasetNewDev))
    # print("trigram smoothed perplexity with Interpolation:  ",
    #       calculate_trigram_perplexity(dataset_dev_model_smoothed, datasetNewDev))
    # #
    # defaultLambdaSet = [0.2, 0.3, 0.5]
    # print("\n===== Perplexity for Interpolation for Dev data with values 0.2, 0.3, 0.5=======")
    # dataset_dev_model_smoothed = TrigramLanguageModel(datasetNewDev, defaultLambdaSet, smoothing_value=1,
    #                                                   smoothing=True)
    # print("unigram smoothed perplexity with Interpolation: ",
    #       calculate_unigram_perplexity(dataset_dev_model_smoothed, datasetNewDev))
    # print("bigram smoothed perplexity with Interpolation: ",
    #       calculate_bigram_perplexity(dataset_dev_model_smoothed, datasetNewDev))
    # print("trigram smoothed perplexity with Interpolation:  ",
    #       calculate_trigram_perplexity(dataset_dev_model_smoothed, datasetNewDev))
    # #
    # defaultLambdaSet = [0.1, 0.2, 0.7]
    # print("\n===== Perplexity for Interpolation for Dev data with values 0.1, 0.2, 0.7=======")
    # dataset_dev_model_smoothed = TrigramLanguageModel(datasetNewDev, defaultLambdaSet, smoothing_value=1,
    #                                                   smoothing=True)
    # print("unigram smoothed perplexity with Interpolation: ",
    #       calculate_unigram_perplexity(dataset_dev_model_smoothed, datasetNewDev))
    # print("bigram smoothed perplexity with Interpolation: ",
    #       calculate_bigram_perplexity(dataset_dev_model_smoothed, datasetNewDev))
    # print("trigram smoothed perplexity with Interpolation:  ",
    #       calculate_trigram_perplexity(dataset_dev_model_smoothed, datasetNewDev))
    #
    # # Question 4 - 2
    # # Calculatng Perplexity with interpolation on test data using values 0.1, 0.4, 0.5 - best from dev perplexity
    #
    # defaultLambdaSet = [0.1, 0.4, 0.5]
    # print("\n===== Perplexity for Interpolation for Test data with values 0.1, 0.4, 0.5=======")
    # dataset_test_model_smoothed = TrigramLanguageModel(datasetNewTest, defaultLambdaSet, smoothing_value=1,
    #                                                   smoothing=True)
    # print("unigram smoothed perplexity with Interpolation: ",
    #       calculate_unigram_perplexity(dataset_test_model_smoothed, datasetNewTest))
    # print("bigram smoothed perplexity with Interpolation: ",
    #       calculate_bigram_perplexity(dataset_test_model_smoothed, datasetNewTest))
    # print("trigram smoothed perplexity with Interpolation:  ",
    #       calculate_trigram_perplexity(dataset_test_model_smoothed, datasetNewTest))