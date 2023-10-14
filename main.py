import re
import math
import string
import operator
from collections import deque


# used for unseen words in training vocabularies
UNK = "<UNK>"
# sentence start and end
SENTENCE_START = "<START>"
SENTENCE_END = "<STOP>"

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
    fin = open(fileName, "rt")
    fout = open(fileName + "_" + "out.txt", "wt")
    special_char_list = ["(", ")", "?", "[", "*", "!", ","]
    for line in fin:
        #newLine = line
        newLine = re.sub(r' [^a-zA-Z0-9\n\s\.][^a-zA-Z0-9\n\.]', ' <UNK> ', line)
        for word in line.split():
            if word in wordDictionary.keys() and wordDictionary.get(word) < 5:
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
    def __init__(self, sentences, defaultLambdaSet, smoothing_value, smoothing=False):

        self.defaultLambdaSet = defaultLambdaSet
        self.smoothing_Value = smoothing_value
        self.unigram_frequencies = dict()
        self.corpus_length = 0

        for sentence in sentences:
            for word in sentence:
                self.unigram_frequencies[word] = self.unigram_frequencies.get(word, 0) + 1
                self.corpus_length += 1
        self.unique_words = len(self.unigram_frequencies) - 2
        self.smoothing = smoothing

    def calculate_unigram_probability(self, word):

        word_probability_numerator = self.unigram_frequencies.get(word, 0)
        word_probability_denominator = self.corpus_length
        if self.smoothing:
            if self.smoothing_Value != 0:
                word_probability_numerator += self.smoothing_Value
                word_probability_denominator += self.unique_words*self.smoothing_Value
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

        return math.pow(2, sentence_probability_log_sum) if normalize_probability else sentence_probability_log_sum


class BigramLanguageModel(UnigramLanguageModel):
    def __init__(self, sentences, defaultLambdaSet, smoothing_value, smoothing=False):
        UnigramLanguageModel.__init__(self, sentences, defaultLambdaSet, smoothing_value, smoothing)
        self.defaultLambdaSet = defaultLambdaSet
        self.smoothing_Value = smoothing_value

        self.bigram_frequencies = dict()
        self.unique_bigrams = set()
        for sentence in sentences:
            previous_word = None
            for word in sentence:
                if previous_word != None:
                    self.bigram_frequencies[(previous_word, word)] = self.bigram_frequencies.get((previous_word, word),
                                                                                  0) + 1
                    self.unique_bigrams.add((previous_word, word))
                previous_word = word
        self.unique__bigram_words = len(self.unique_bigrams)

    def calculate_bigram_probabilty(self, previous_word, word):
        bigram_word_probability_numerator = self.bigram_frequencies.get((previous_word, word), 0)
        bigram_word_probability_denominator = self.unigram_frequencies.get(previous_word, 0)

        if self.smoothing:
            if self.smoothing_value != 0:
                bigram_word_probability_numerator += self.smoothing_Value
                bigram_word_probability_denominator += self.unique_words*self.smoothing_Value

        return 0.0 if bigram_word_probability_numerator == 0 or bigram_word_probability_denominator == 0 else float(
            bigram_word_probability_numerator) / float(bigram_word_probability_denominator)

    def calculate_bigram_sentence_probability(self, sentence, normalize_probability=True):
        bigram_sentence_probability_log_sum = 0
        previous_word = None
        # print(sentence)
        for word in sentence:

            if previous_word != None:
                if not self.defaultLambdaSet:
                    bigram_word_probability = self.calculate_bigram_probabilty(previous_word, word)

                else:
                    bigram_word_probability = self.defaultLambdaSet[1] * self.calculate_bigram_probabilty(previous_word,
                                                                                                          word)
                bigram_sentence_probability_log_sum += math.log(bigram_word_probability, 2)

            previous_word = word
        return math.pow(2,
                        bigram_sentence_probability_log_sum) if normalize_probability else bigram_sentence_probability_log_sum


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
        self.trigram_frequencies = dict()
        self.unique_trigrams = set()

        for sentence in sentences:
            previous_word = None
            previous_second_word = None
            for word in sentence:
                if previous_word != None and previous_second_word != None:
                    if (previous_second_word, previous_word, word) in self.trigram_frequencies.keys():
                        self.trigram_frequencies[(previous_second_word, previous_word, word)] += 1
                    else:
                        self.trigram_frequencies[(previous_second_word, previous_word, word)] = 1

                previous_second_word = previous_word
                previous_word = word

            self.unique_trigram_words = len(self.trigram_frequencies)

    def calculate_trigram_probabilty(self, previous_second_word, previous_word, word):

        trigram_word_probability_numerator = self.trigram_frequencies.get((previous_second_word, previous_word, word),
                                                                          0)
        trigram_word_probability_denominator = self.bigram_frequencies.get((previous_second_word, previous_word), 0)

        if self.smoothing == True:
            if self.smoothing_value != 0:
                trigram_word_probability_numerator += self.smoothing_value
                trigram_word_probability_denominator += self.unique_words*self.smoothing_value

        return 0.0 if trigram_word_probability_numerator == 0 or trigram_word_probability_denominator == 0 else float(
            trigram_word_probability_numerator) / float(trigram_word_probability_denominator)

    def calculate_trigram_sentence_probability(self, sentence, normalize_probability=True):

        trigram_sentence_probability_log_sum = 0
        previous_word = None
        previous_second_word = None

        for word in sentence:
            if previous_word != None and previous_second_word != None:
                if not self.defaultLambdaSet:
                    trigram_word_probability = self.calculate_trigram_probabilty(previous_second_word, previous_word,
                                                                                 word)

                else:
                    trigram_word_probability = self.defaultLambdaSet[2] * self.calculate_trigram_probabilty(
                        previous_second_word, previous_word,
                        word)
                trigram_sentence_probability_log_sum += math.log(trigram_word_probability, 2)

            previous_second_word = previous_word
            previous_word = word
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
        # remove one for number of trigrams in sentence
        trigram_count += len(sentence) - 2

    return trigram_count


def calculate_unigram_perplexity(model, sentences):
    unigram_count = calculate_number_of_unigrams(sentences)

    sentence_probability_log_sum = 0
    for sentence in sentences:
        try:

            sentence_probablity = model.calculate_sentence_probability(sentence)
            individual_sentence_probability = math.log(sentence_probablity, 2)
            sentence_probability_log_sum -= individual_sentence_probability

        except Exception as e:
            sentence_probability_log_sum -= 0
    return math.pow(2, sentence_probability_log_sum / unigram_count)


def calculate_bigram_perplexity(model, sentences):
    number_of_bigrams = calculate_number_of_bigrams(sentences)
    bigram_sentence_probability_log_sum = 0
    for sentence in sentences:
        try:
            bigram_sentence_probability_log_sum -= math.log(model.calculate_bigram_sentence_probability(sentence), 2)

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
            trigram_sentence_probability_log_sum -= 0.0
    return math.pow(2, trigram_sentence_probability_log_sum / number_of_trigrams)


if __name__ == '__main__':
    trainingFile = '1b_benchmark.train.tokens'
    datasetTraining = read_file(trainingFile)
    defaultLambdaSet = []
    replace_Unknown_Word('1b_benchmark.train.tokens')
    datasetNewTraining = read_file('1b_benchmark.test.tokens_out.txt')
    dataset_model_training = TrigramLanguageModel(datasetNewTraining, defaultLambdaSet, smoothing_value=0)
    dataset_model_smoothed_training = TrigramLanguageModel(datasetNewTraining, defaultLambdaSet, smoothing_value=1, smoothing=True)

    devFile = '1b_benchmark.dev.tokens'
    datasetDev = read_file(devFile)
    replace_Unknown_Word(devFile)
    datasetNewDev = read_file('1b_benchmark.dev.tokens_out.txt')
    dataset_model_dev = TrigramLanguageModel(datasetNewDev, defaultLambdaSet, smoothing_value=0)
    dataset_model_smoothed_dev = TrigramLanguageModel(datasetNewDev, defaultLambdaSet, smoothing_value=1, smoothing=True)
    #sorted_vocab = data.sorted_vocabulary()

    testFile = '1b_benchmark.test.tokens'
    datasetTest = read_file(testFile)
    replace_Unknown_Word(testFile)
    datasetNewTest = read_file('1b_benchmark.test.tokens_out.txt')
    dataset_model_test = TrigramLanguageModel(datasetNewTest, defaultLambdaSet, smoothing_value=0)
    dataset_model_smoothed_test = TrigramLanguageModel(datasetNewTest, defaultLambdaSet, smoothing_value=1, smoothing=True)
    # sorted_vocab = dataset_model_smoothed_test.sorted_vocabulary()

    #Question 2 - Training Data Unsmoothed Perplexity
    print("== Training Data Unsmoothed PERPLEXITY == ")
    print("Training Data Perplexity for - unigram : ", calculate_unigram_perplexity(dataset_model_training, datasetNewTraining))
    print("Training Data Perplexity for - bigram: ", calculate_bigram_perplexity(dataset_model_training, datasetNewTraining))
    print("Training Data Perplexity for - trigram: ", calculate_trigram_perplexity(dataset_model_training,datasetNewTraining))

    # QUESTION 3 - 1 - Calculating Perplexity for alpha = 1 on training data
    dataset_training_model_smoothed = TrigramLanguageModel(datasetNewTraining, defaultLambdaSet, smoothing_value=1,
                                                           smoothing=True)
    print("======== Perplexity for Training Data with Smoothing value = 1 ==========")
    print("unigram smoothed Perplexity : ",
          calculate_unigram_perplexity(dataset_training_model_smoothed, datasetNewTraining))
    print("bigram smoothed Perplexity : ",
          calculate_bigram_perplexity(dataset_training_model_smoothed, datasetNewTraining))
    print("trigram smoothed Perplexity : ",
          calculate_trigram_perplexity(dataset_training_model_smoothed, datasetNewTraining))

    # # Question 2 - Dev Data Unsmoothed Perplexity
    print("== Dev Data Unsmoothed PERPLEXITY == ")
    print("Dev Data Perplexity for - unigram : ",calculate_unigram_perplexity(dataset_model_dev, datasetNewDev))
    print("Dev Data Perplexity for - bigram: ",calculate_bigram_perplexity(dataset_model_dev, datasetNewDev))
    print("Dev Data Perplexity for - trigram: ",calculate_trigram_perplexity(dataset_model_dev, datasetNewDev))
    #
    # # Question 2 - Test Data Unsmoothed Perplexity
    print("== Test Data Unsmoothed PERPLEXITY == ")
    print("Test Data Perplexity for - unigram : ",
          calculate_unigram_perplexity(dataset_model_test, datasetNewTest))
    print("Test Data Perplexity for - bigram: ",
          calculate_bigram_perplexity(dataset_model_test, datasetNewTest))
    print("Test Data Perplexity for - trigram: ",
          calculate_trigram_perplexity(dataset_model_test, datasetNewTest))

    print("======== Perplexity for Training Data with Smoothing value = 1 ==========")
    print("unigram smoothed Perplexity : ", calculate_unigram_perplexity(dataset_model_smoothed_training, datasetNewTraining))
    print("bigram smoothed Perplexity : ", calculate_bigram_perplexity(dataset_model_smoothed_training, datasetNewTraining))
    print("trigram smoothed Perplexity : ", calculate_trigram_perplexity(dataset_model_smoothed_training, datasetNewTraining))
    #
    # #QUESTION 3 - 1 - Calculating Perplexity for alpha = 1 on training data
    dataset_dev_model_smoothed = TrigramLanguageModel(datasetNewDev, defaultLambdaSet,
                                                           smoothing_value=1, smoothing=True)
    print("======== Perplexity for Dev Data with Smoothing value = 1 ==========")
    print("unigram smoothed Perplexity : ", calculate_unigram_perplexity(dataset_dev_model_smoothed, datasetNewDev))
    print("bigram smoothed Perplexity : ", calculate_bigram_perplexity(dataset_dev_model_smoothed, datasetNewDev))
    print("trigram smoothed Perplexity : ", calculate_trigram_perplexity(dataset_dev_model_smoothed, datasetNewDev))
    #
    # #QUESTION 3 - 2 - Calculating Perplexity for different smoothing value on Training Data
    dataset_training_model_smoothed = TrigramLanguageModel(datasetNewTraining, defaultLambdaSet,
                                                  smoothing_value=2, smoothing=True)
    print("\n======== Perplexity for Training Data with Smoothing value = 2 ==========")
    print("unigram smoothed Perplexity : ", calculate_unigram_perplexity(dataset_training_model_smoothed, datasetNewTraining))
    print("bigram smoothed Perplexity : ", calculate_bigram_perplexity(dataset_training_model_smoothed, datasetNewTraining))
    print("trigram smoothed Perplexity : ", calculate_trigram_perplexity(dataset_training_model_smoothed,datasetNewTraining))
    # #
    dataset_training_model_smoothed = TrigramLanguageModel(datasetNewTraining, defaultLambdaSet, smoothing_value=3, smoothing=True)
    print("\n======== Perplexity for Training Data with Smoothing value = 3 ==========")
    print("unigram smoothed Perplexity : ", calculate_unigram_perplexity(dataset_training_model_smoothed, datasetNewTraining))
    print("bigram smoothed Perplexity : ", calculate_bigram_perplexity(dataset_training_model_smoothed, datasetNewTraining))
    print("trigram smoothed Perplexity : ", calculate_trigram_perplexity(dataset_training_model_smoothed, datasetNewTraining))
    # #
    dataset_training_model_smoothed = TrigramLanguageModel(datasetNewTraining, defaultLambdaSet, smoothing_value=10, smoothing=True)
    print("\n======== Perplexity for Training Data with Smoothing value = 10 ==========")
    print("unigram smoothed Perplexity : ", calculate_unigram_perplexity(dataset_training_model_smoothed, datasetNewTraining))
    print("bigram smoothed Perplexity : ", calculate_bigram_perplexity(dataset_training_model_smoothed, datasetNewTraining))
    print("trigram smoothed Perplexity : ", calculate_trigram_perplexity(dataset_training_model_smoothed, datasetNewTraining))
    #
    # # QUESTION 3 - 2 - Calculating Perplexity for different smoothing value on Dev Data
    dataset_model_smoothed_dev = TrigramLanguageModel(datasetNewDev, defaultLambdaSet, smoothing_value=2, smoothing=True)
    print("\n======== Perplexity for Dev Data with Smoothing value = 2 ==========")
    print("unigram smoothed Perplexity : ",
          calculate_unigram_perplexity(dataset_model_smoothed_dev, datasetNewDev))
    print("bigram smoothed Perplexity : ",
          calculate_bigram_perplexity(dataset_model_smoothed_dev, datasetNewDev))
    print("trigram smoothed Perplexity : ",
          calculate_trigram_perplexity(dataset_model_smoothed_dev, datasetNewDev))
    # #
    dataset_model_smoothed_dev = TrigramLanguageModel(datasetNewDev, defaultLambdaSet, smoothing_value=3, smoothing=True)

    print("\n======== Perplexity for Dev Data with Smoothing value = 3 ==========")
    print("unigram smoothed Perplexity : ",
          calculate_unigram_perplexity(dataset_model_smoothed_dev, datasetNewDev))
    print("bigram smoothed Perplexity : ",
          calculate_bigram_perplexity(dataset_model_smoothed_dev, datasetNewDev))
    print("trigram smoothed Perplexity : ",
          calculate_trigram_perplexity(dataset_model_smoothed_dev, datasetNewDev))
    # #
    dataset_model_smoothed_dev = TrigramLanguageModel(datasetNewDev, defaultLambdaSet, smoothing_value=10, smoothing=True)
    print("\n======== Perplexity for Dev Data with Smoothing value = 10 ==========")
    print("unigram smoothed Perplexity : ",
          calculate_unigram_perplexity(dataset_model_smoothed_dev, datasetNewDev))
    print("bigram smoothed Perplexity : ",
          calculate_bigram_perplexity(dataset_model_smoothed_dev, datasetNewDev))
    print("trigram smoothed Perplexity : ",
          calculate_trigram_perplexity(dataset_model_smoothed_dev, datasetNewDev))
    #
    # # Question 3- 3 Calculating Test data perplexity with alpha = 10 - which gave best dev data  smoothing perplexity

    dataset_model_smoothed_test = TrigramLanguageModel(datasetNewTest, defaultLambdaSet, smoothing_value=1,
                                                      smoothing=True)
    print("\n======== Perplexity Test  Data with Smoothing value = 1 ==========")
    print("unigram smoothed Perplexity : ",
          calculate_unigram_perplexity(dataset_model_smoothed_test, datasetNewTest))
    print("bigram smoothed Perplexity : ",
          calculate_bigram_perplexity(dataset_model_smoothed_test, datasetNewTest))
    print("trigram smoothed Perplexity : ",
          calculate_trigram_perplexity(dataset_model_smoothed_test, datasetNewTest))
    #
    # #
    # # =================================== QUESTION 4 ===================================
    #
    # # Reporting Training Data Perplexity for λ1 = 0.1, λ2 = 0.3, λ3 = 0.6. and 5 additional Sets
    defaultLambdaSet = [0.1,0.3,0.6]
    print("\n===== Perplexity for Interpolation for training data with values 0.1, 0.3, 0.6=======")
    dataset_train_model_smoothed = TrigramLanguageModel(datasetNewTraining, defaultLambdaSet, smoothing_value=1, smoothing=True)
    print("unigram smoothed perplexity with Interpolation: ", calculate_unigram_perplexity(dataset_train_model_smoothed, datasetNewTraining))
    print("bigram smoothed perplexity with Interpolation: ", calculate_bigram_perplexity(dataset_train_model_smoothed, datasetNewTraining))
    print("trigram smoothed perplexity with Interpolation:  ", calculate_trigram_perplexity(dataset_train_model_smoothed, datasetNewTraining))
    # #
    defaultLambdaSet = [0.1, 0.4, 0.5]
    print("\n===== Perplexity for Interpolation for training data with values 0.1, 0.4, 0.5=======")
    dataset_train_model_smoothed = TrigramLanguageModel(datasetNewTraining, defaultLambdaSet, smoothing_value=1, smoothing=True)
    print("unigram smoothed perplexity with Interpolation: ",
          calculate_unigram_perplexity(dataset_train_model_smoothed, datasetNewTraining))
    print("bigram smoothed perplexity with Interpolation: ",
          calculate_bigram_perplexity(dataset_train_model_smoothed, datasetNewTraining))
    print("trigram smoothed perplexity with Interpolation:  ",
          calculate_trigram_perplexity(dataset_train_model_smoothed, datasetNewTraining))
    # #
    defaultLambdaSet = [0.2, 0.2, 0.6]
    print("\n===== Perplexity for Interpolation for training data with values 0.2, 0.2, 0.6=======")
    dataset_train_model_smoothed = TrigramLanguageModel(datasetNewTraining, defaultLambdaSet, smoothing_value=1, smoothing=True)
    print("unigram smoothed perplexity with Interpolation: ",
          calculate_unigram_perplexity(dataset_train_model_smoothed, datasetNewTraining))
    print("bigram smoothed perplexity with Interpolation: ",
          calculate_bigram_perplexity(dataset_train_model_smoothed, datasetNewTraining))
    print("trigram smoothed perplexity with Interpolation:  ",
          calculate_trigram_perplexity(dataset_train_model_smoothed, datasetNewTraining))
    # #
    defaultLambdaSet = [0.2, 0.3, 0.5]
    print("\n===== Perplexity for Interpolation for training data with values 0.2, 0.3, 0.5=======")
    dataset_train_model_smoothed = TrigramLanguageModel(datasetNewTraining, defaultLambdaSet, smoothing_value=1, smoothing=True)
    print("unigram smoothed perplexity with Interpolation: ",
          calculate_unigram_perplexity(dataset_train_model_smoothed, datasetNewTraining))
    print("bigram smoothed perplexity with Interpolation: ",
          calculate_bigram_perplexity(dataset_train_model_smoothed, datasetNewTraining))
    print("trigram smoothed perplexity with Interpolation:  ",
          calculate_trigram_perplexity(dataset_train_model_smoothed, datasetNewTraining))
    # #
    defaultLambdaSet = [0.1, 0.2, 0.7]
    print("\n===== Perplexity for Interpolation for training data with values 0.1, 0.2, 0.7=======")
    dataset_train_model_smoothed = TrigramLanguageModel(datasetNewTraining, defaultLambdaSet, smoothing_value=1, smoothing=True)
    print("unigram smoothed perplexity with Interpolation: ",
          calculate_unigram_perplexity(dataset_train_model_smoothed, datasetNewTraining))
    print("bigram smoothed perplexity with Interpolation: ",
          calculate_bigram_perplexity(dataset_train_model_smoothed, datasetNewTraining))
    print("trigram smoothed perplexity with Interpolation:  ",
          calculate_trigram_perplexity(dataset_train_model_smoothed, datasetNewTraining))
    #
    # # Reporting Dev Data Perplexity for λ1 = 0.1, λ2 = 0.3, λ3 = 0.6. and 5 additional Sets
    defaultLambdaSet = [0.1, 0.3, 0.6]
    print("\n===== Perplexity for Interpolation for Dev data with values 0.1, 0.3, 0.6=======")
    dataset_dev_model_smoothed = TrigramLanguageModel(datasetNewDev, defaultLambdaSet, smoothing_value=1,
                                                        smoothing=True)
    print("unigram smoothed perplexity with Interpolation: ",
          calculate_unigram_perplexity(dataset_dev_model_smoothed, datasetNewDev))
    print("bigram smoothed perplexity with Interpolation: ",
          calculate_bigram_perplexity(dataset_dev_model_smoothed, datasetNewDev))
    print("trigram smoothed perplexity with Interpolation:  ",
          calculate_trigram_perplexity(dataset_dev_model_smoothed, datasetNewDev))
    # #
    defaultLambdaSet = [0.1, 0.4, 0.5]
    print("\n===== Perplexity for Interpolation for Dev data with values 0.1, 0.4, 0.5=======")
    dataset_dev_model_smoothed = TrigramLanguageModel(datasetNewDev, defaultLambdaSet, smoothing_value=1,
                                                      smoothing=True)
    print("unigram smoothed perplexity with Interpolation: ",
          calculate_unigram_perplexity(dataset_dev_model_smoothed, datasetNewDev))
    print("bigram smoothed perplexity with Interpolation: ",
          calculate_bigram_perplexity(dataset_dev_model_smoothed, datasetNewDev))
    print("trigram smoothed perplexity with Interpolation:  ",
          calculate_trigram_perplexity(dataset_dev_model_smoothed, datasetNewDev))
    # #
    defaultLambdaSet = [0.2, 0.2, 0.6]
    print("\n===== Perplexity for Interpolation for Dev data with values 0.2, 0.2, 0.6=======")
    dataset_dev_model_smoothed = TrigramLanguageModel(datasetNewDev, defaultLambdaSet, smoothing_value=1,
                                                      smoothing=True)
    print("unigram smoothed perplexity with Interpolation: ",
          calculate_unigram_perplexity(dataset_dev_model_smoothed, datasetNewDev))
    print("bigram smoothed perplexity with Interpolation: ",
          calculate_bigram_perplexity(dataset_dev_model_smoothed, datasetNewDev))
    print("trigram smoothed perplexity with Interpolation:  ",
          calculate_trigram_perplexity(dataset_dev_model_smoothed, datasetNewDev))
    # #
    defaultLambdaSet = [0.2, 0.3, 0.5]
    print("\n===== Perplexity for Interpolation for Dev data with values 0.2, 0.3, 0.5=======")
    dataset_dev_model_smoothed = TrigramLanguageModel(datasetNewDev, defaultLambdaSet, smoothing_value=1,
                                                      smoothing=True)
    print("unigram smoothed perplexity with Interpolation: ",
          calculate_unigram_perplexity(dataset_dev_model_smoothed, datasetNewDev))
    print("bigram smoothed perplexity with Interpolation: ",
          calculate_bigram_perplexity(dataset_dev_model_smoothed, datasetNewDev))
    print("trigram smoothed perplexity with Interpolation:  ",
          calculate_trigram_perplexity(dataset_dev_model_smoothed, datasetNewDev))
    # #
    defaultLambdaSet = [0.1, 0.2, 0.7]
    print("\n===== Perplexity for Interpolation for Dev data with values 0.1, 0.2, 0.7=======")
    dataset_dev_model_smoothed = TrigramLanguageModel(datasetNewDev, defaultLambdaSet, smoothing_value=1,
                                                      smoothing=True)
    print("unigram smoothed perplexity with Interpolation: ",
          calculate_unigram_perplexity(dataset_dev_model_smoothed, datasetNewDev))
    print("bigram smoothed perplexity with Interpolation: ",
          calculate_bigram_perplexity(dataset_dev_model_smoothed, datasetNewDev))
    print("trigram smoothed perplexity with Interpolation:  ",
          calculate_trigram_perplexity(dataset_dev_model_smoothed, datasetNewDev))
    #
    # # Question 4 - 2
    # # Calculatng xPerplexity with interpolation on test data using values 0.1, 0.4, 0.5 - best from dev perplexity

    defaultLambdaSet = [0.1, 0.2, 0.7]
    print("\n===== Perplexity for Interpolation for Test data with values 0.1, 0.2, 0.7=======")
    dataset_test_model_smoothed = TrigramLanguageModel(datasetNewTest, defaultLambdaSet, smoothing_value=1,
                                                      smoothing=True)
    print("unigram smoothed perplexity with Interpolation: ",
          calculate_unigram_perplexity(dataset_test_model_smoothed, datasetNewTest))
    print("bigram smoothed perplexity with Interpolation: ",
          calculate_bigram_perplexity(dataset_test_model_smoothed, datasetNewTest))
    print("trigram smoothed perplexity with Interpolation:  ",
          calculate_trigram_perplexity(dataset_test_model_smoothed, datasetNewTest))
