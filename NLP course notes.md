# SIADS-632-DataMining-11

## N-gram Language Models Predicting words by assigning probabilities
[Intro](https://www.youtube.com/watch?v=oWsMIW-5xUc)
[Stanford video](https://www.youtube.com/watch?v=hB2ShMlwTyc)
[Texas University PDF link for formulas](https://www.cs.utexas.edu/~mooney/cs388/slides/equation-sheet.pdf)

Models that assign probabilities to sequences of words are called language model  or LMs. The simplest model that assigns probabilities to sentences and sequences of words: the n-gram. An n-gram is a sequence of n words: a 2-gram (which we’ll call bigram) is a two-word sequence of wordslike “please turn”, “turn your”, or ”your homework”, and a 3-gram (a trigram) is a three-word sequence of words like “please turn your”, or “turn your homework”.
We’ll see how to use n-gram models to estimate the probability of the last word of an n-gram given the previous words, and also to assign probabilities to entire sequences.

A model that computes either of these:
P(W) or P(w<sub>n</sub> | w<sub>1</sub>, w<sub>2</sub>,....w<sub>n-1</sub>) is called a language model.


Probabilities are essential in any task in which we have to identify words in noisy, ambiguous input.
 ### Speech recognition.
 For a speech recognizer to realizethat you said I will be back soonish and not I will be bassoon dish, it helps to know that back soonish is a much more probable sequence than  bassoon dish.
 
### Spelling correction or grammatical error correction
The phrase There are will be much more probable than Their are.

### Machine translation
If Chinese phrase is being converted to English, 
A probabilistic model of word sequences could suggest that briefed reporters on is a more probable English phrase than briefed to reporters.

### Augmentative and alternative communication


## Computing the probability CHAIN RULE OF PROBABILITY

P("its water is so transparent") = P(its) * P(water | its) * P (is | its water) * P(so ! its water is) * P(transparent !its water is so)

Could we just count and divide?
- No Too many possible sentences
- We'll never see enough data for estimating these

The intuition of the n-gram model is that instead of computing the probability of a word given its entire history, we can approximate the history by just the last few words.
The bigram model, for example, approximates the probability of a word given all the previous words P(w<sub>n</sub>|w<sub>1:n-1</sub>) by using only the conditional probability of the preceding word P(w<sub>n|</sub>w<sub>n-1</sub>). In other words, instead of computing the probability P(the|Walden Pond’s water is so transparent that) 
we approximate it with the probability P(the|that)
we are thus making the following approximation:
P(w<sub>n</sub>|w<sub>1:n-1</sub>) ~ P(w<sub>n<sub>|w</sub>n-1</sub>)

## Markov assumption

The assumption that the probability of a word depends only on the previous word is Markov called a Markov assumption. Markov models are the class of probabilistic models that assume we can predict the probability of some future unit without looking too far into the past. We can generalize the bigram (which looks one word into the past) to the trigram (which looks two words into the past) and thus to the n-gram (which looks n􀀀1 words into the past).

Thus, the general equation for this n-gram approximation to the conditional probability of the next word in a sequence is
P(wnjw1:n􀀀1)  P(wnjwn􀀀N+1:n􀀀1)

### How do we estimate these bigram or n-gram probabilities? 

"Language has long-distance dependencies"

#### Simplest case
P(w<sub>1</sub>, w<sub>2</sub>....w<sub>n</sub>) ~ Prduct of P(w<sub>i</sub>)
This will give us a very disjoint set of words whem predicting a sentence.

#### Bigram
Probability of a word is conditional upon just the previous word
P(w<sub>i</sub> | w<sub>1</sub>, w<sub>2</sub>,.... w<sub>i-1</sub>) ~ P(w<sub>i</sub> | w <sub>i-1</sub>)

The sentences generated from a Bi-gram model makes some more semantic sense than unigram

#### Tri-gram
Probability of a word is conditional upon the previous two words

[Stanford lecture](https://www.youtube.com/watch?v=MUNFfBGdF1k)

## Maximum Likelihood estimate

The MLE 
    - **of some parameter** of a model M from a training set T
    - maximizes the **likelihood of the training set** T given the model M

An intuitive way to estimate probabilities is called maximum likelihood estimation or MLE. We get maximum likelihood estimation the MLE estimate for the parameters of an n-gram model by getting counts from a normalize corpus, and normalizing the counts so that they lie between 0 and 1.

The below is MLE
P(w<sub>i</sub> | w <sub>i-1</sub>) = Count(w<sub>i-1</sub>, w<sub>i</sub>) /Count(w <sub>i-1</sub>)
(Joint probability divided by probability of previous word) 

Bigram table shows word in row index followed by word in column index count

|      | i | want | this |
|------|---|------|------|
| i    | 0 | 3    | 0    |
| want | 8 | 0    | 10   |
| this | 3 | 5    | 0    |

Now that you have the joint probability above which is Count(i, want) = 3
Dividing this by the unigram count Count(i)
This will give probability of i given want P(i|want) 

This is called **normalizing by the unigram**.

Once you have each of these probabilities, you can calculate the sentence probability

P(<\s>I want english food</\s>) =
P(I|,<\s>)
* P(want | I)
* P(english | want)
* P(food | english)
* P(</\s>|food)


#### We do everything in log space
    - Avoid underflow
    - Adding is faster than multiplying


#### N gram Corpus
 SRILM
 Google N-gram release.  
 Google Book N gram corpus 

## Evaluation and perplexity
How good is out model?
 - Model assigns **higher probability** to "real" or "frequently observed" sentences than
   ungrammatical or rarely observed sentences.
- We train parameters of our model on a training set.
- We test the model's performance on data we have not seen

### Extrinsic
We test the model and a competitor on **an external task** 
Example: Models for a system that is a spell check:
    - Evaluate the accuracy of two models 
    - Compare

Cons: It can take days or weeks to evaluate


### Intrinsic evaluation : [Perplexity](https://www.youtube.com/watch?v=NCyCkgMLRiY)
Intrinsic to the model itself rather than the application.

The idea is that the best model will return the best probability number - high probability.
The best model best predicts an unseen test set.

Perplexity is the probability of the test set normalized by the number of words.

It is the inverse of the nth square (normalization) of the probability of the test set.

Minimizing perplexity is the same as maximizing probability.

Perplixity is related the the average branching factor.
Weighted equivalent branching factor

## The Shannon Visualization method
- Choose a random bigram according to its **probability** :            (<s>, I)
- Now chosse a random bigram (w, x) according to **its probability**         I   want
- And so on until we choose </s>                                                 want Bhel
                                                                                      Bhel </s>

Then we string it all together to create the sentence.

## Problems with N gram models
- Weak generalization- If you train on the Wll street Journal - diificul to predict Shakespeare ad vice versa.
- Probability of having seen an n-gram can be zero - Perplexity is undefined.

## SMOOTHING

### Add-one estimation or Laplace smoothing
- Pretend that we saw each word one more time than we did.
- Just add one to all the counts
- MLE estimate  = Count of the Bigram divided by Count of the Unigram
    P<sub>MLE</sub>(w<sub>i</sub>|w<sub>i-1</sub>) = Count(w<sub>i-1</sub>,w<sub>i</sub> ) / Count(w<sub>i-1</sub>)

- Add-1 estimate  = Count of the Bigram divided by Count of the Unigram

P<sub>Add+1</sub>(w<sub>i</sub>|w<sub>i-1</sub>) = Count(w<sub>i-1</sub>,w<sub>i</sub> ) + 1 / (Count(w<sub>i-1</sub>) + V)

What this does is reassign some of the probabilities from each word to words that do not occur at all so that unseen data in test data can be handled better.

NOTE: Add one is a blunt instrument and is not used for n grams

## [Back off and Interpolation](https://www.youtube.com/watch?v=PC0nIk4-HoI)

Interpolation works better than Backoff

[Entropy from Statquest](https://www.youtube.com/watch?v=YtebGVx-Fxw&t=462s)

> Average number of question to ask to get the information required

    - Surprise is inversely related to the probaility but we need to take the log of the probability since a probability of 1 (totally probable) will then give us a Surprise of 0
    - Low entropy implies low Surprise and is the avarage of the product of the probability and the surprise. 

 Read this? https://towardsdatascience.com/the-relationship-between-perplexity-and-entropy-in-nlp-f81888775ccc   
      







