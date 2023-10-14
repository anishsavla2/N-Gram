N-gram Language Modeling 

Overview
In this Repo, I built and evaluated unigram, bigram, and trigram language models using Maximum Likelihood Estimation (MLE) without any smoothing.

Handling Out-Of-Vocabulary Words
For tokens that occurred less than three times, I converted them to a special <UNK> token during training. The resulting vocabulary (which includes <UNK> and <STOP>, but excludes <START>) consists of 26,602 unique tokens.

Deliverables
In the writeup, I fully described my models and experimental procedures.
I provided graphs, tables, charts, etc. to support any claims.
Perplexity Scores
I reported the perplexity scores of the unigram, bigram, and trigram language models for my training, development, and test sets. I also discussed the experimental results.

Additive Smoothing
Implementation
I implemented additive smoothing for the unigram, bigram, and trigram models.

Perplexity Scores with Different α Values
For additive smoothing with α = 1, I reported the perplexity scores of the unigram, bigram, and trigram language models for my training and development sets. I also reported scores for two other chosen values of α > 0.

Smoothing with Linear Interpolation
Overview
To enhance the performance of my language model, I implemented linear interpolation smoothing between the MLE unigram, bigram, and trigram models.

Deliverables
I reported perplexity scores on training and development sets for various combinations of λ_1, λ_2, λ_3. I included scores for the values λ_1 = 0.1, λ_2 = 0.3, λ_3 = 0.6.




