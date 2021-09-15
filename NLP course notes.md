# SIADS-632-DataMining-11

## N-gram Language Models Predicting words by assigning probabilities
[Intro](https://www.youtube.com/watch?v=oWsMIW-5xUc)
[Stanford video](https://www.youtube.com/watch?v=hB2ShMlwTyc)
[Stanford Text](https://web.stanford.edu/~jurafsky/slp3/3.pdf)
[Texas University PDF link for formulas](https://www.cs.utexas.edu/~mooney/cs388/slides/equation-sheet.pdf)

Models that assign probabilities to sequences of words are called language model  or LMs. The simplest model that assigns probabilities to sentences and sequences of words: the n-gram. An n-gram is a sequence of n words: a 2-gram (which we’ll call bigram) is a two-word sequence of wordslike “please turn”, “turn your”, or ”your homework”, and a 3-gram (a trigram) is a three-word sequence of words like “please turn your”, or “turn your homework”.
We’ll see how to use n-gram models to estimate the probability of the last word of an n-gram given the previous words, and also to assign probabilities to entire sequences. **The intuition of the n-gram model is that instead of computing the probability of a word given its entire history, we can approximate the history by just the last few words.** The assumption that the probability of a word depends only on the previous word is  called a **Markov assumption.**

A model that computes either of these:
P(W) or P(w<sub>n</sub> | w<sub>1</sub>, w<sub>2</sub>,....w<sub>n-1</sub>) is called a language model.

Probabilities are essential in any task in which we have to identify words in noisy, ambiguous input.
We’ll see how to use n-gram models to estimate the probability of the last word of
an n-gram given the previous words, and also to assign probabilities to entire sequences. In a bit of terminological ambiguity, we usually drop the word “model”,
and use the term n-gram (and bigram, etc.) to mean either the word sequence itself or the predictive model that assigns it a probability.

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

The below is MLE (Look into Page four of Chapter 3 in Stanford notes)
P(w<sub>i</sub> | w <sub>i-1</sub>) = Count(w<sub>i-1</sub>, w<sub>i</sub>) /Count(w <sub>i-1</sub>)
(Joint probability divided by probability of previous word. Note that the denominator has been simplified, since the sum of all bigram counts that start with
a given word w<sub>n−1</sub> must be equal to the unigram count for that word w<sub>n−1</sub>) 

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

**NOTE:** For longer grams  we’ll need to assume extra contexts to the left and right of the sentence end. For example, to compute trigram probabilities at the very beginning of the sentence,we use two pseudo-words for the first trigram (i.e., P(I|\<s>\<s>).

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
- Choose a random bigram according to its **probability** :            (<\s>, I)
- Now chosse a random bigram (w, x) according to **its probability**         I   want
- And so on until we choose <\/s>                                                 want Bhel
                                                                                      Bhel <\/s>

Then we string it all together to create the sentence.

## Problems with N gram models
- Weak generalization- If you train on the Wll street Journal - diificul to predict Shakespeare ad vice versa.
- Probability of having seen an n-gram can be zero - Perplexity is undefined.

## SMOOTHING
Why 
  - Becuase of zeroing out of certain words when not seen in original training corpus
  - Helps with generalizations
  - Works for words but not n-grams

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

** Read this?** https://towardsdatascience.com/the-relationship-between-perplexity-and-entropy-in-nlp-f81888775ccc   

## Backoff and Interpolation

Sometimes it helps to use **less** context
- Backoff
    - Use trigram if you have good evidence
    - Otherwise bigram, otherwise unigram

- Interpolation
   - Mix unigram, bigram, trigram

### Linear Interpolation
- Simple Interpolation
    P<sub>hat</sub>(w<sub>n</sub> | w<sub>n-1</sub> w<sub>n-2</sub>) 
    =  $\lambda$ <sub>1</sub> P (w<sub>n</sub> | w<sub>n-1</sub> w<sub>n-2</sub>) 
    + $\lambda$ <sub>2</sub> P (w<sub>n</sub> | w<sub>n-1</sub> ) 
    + $\lambda$ <sub>3</sub> P (w<sub>n</sub> )

Sum of $\lambda$ s equals 1


## Out of Vocabulary words
- Create an unknown word token \<UNK>

### Training of \<UNK> probabilities
- Create a fixed lexicon L of size V
- At text normalization phase (dividing by count of the unigrams), any training word not in L changed to \<UNK>
- Now train its probablitlies like a normal word
- At decoding time: Use UNK probabilities for any word not in training 

### Unigram prior smoothing

P<sub>Add-k</sub>(w<sub>i</sub> | w<sub>i-1</sub>) =

(COUNT(w<sub>i-1</sub>, w<sub>i</sub>) + m(1/V) ) / (Count(w<sub>i-1</sub>) + m)


P<sub>UnigramPrior</sub>(w<sub>i</sub> | w<sub>i-1</sub>) =

(COUNT(w<sub>i-1</sub>, w<sub>i</sub>) + m P( w<sub>i</sub>)) / (Count(w<sub>i-1</sub>) + m)


### Advanced smoothing algorithms
- Use the count of things we've seen **ONCE** to help estimate the count of things we've NEVER SEEN

- Good-Turing
- [Kneser-Ney](https://www.youtube.com/watch?v=cbAxvpBFyNU&t=358s)
- Witten-Bell

From above talk on Knesey-Ney:
Used for probability of an n-gram based on other n-grams
- Not every probability of a word following another word can exist and can lead to error.
- Better estimate of probabilties of lower-order unigrams Probablity of singleton and not just tuples- This is because some words are more frequesnt in the context of 
another word. Such as Fransisco is more frequent in the context of San
- The unigram us useful exactly when we haven't seen this bigram. For insttance say we want to complete a sentence I am lookig for my reading -----------. If Fransisco as a unigram has a gigher probability than glasses, then you might use that unigram probability to use Fransisco (when BACKING-off) but it's probability in the context is not being considered.
- P </continuation> -- Instead of considereing unigram probabiliies, we need to look into the set of preceding words that a particular word will complete (in a bi0gram model) Normalize this by the count of total number of word bigram types.
 Every bigram type was a novel continuation the first time it was seen.

> # TIMESERIES
> 
#### Notation N<sub>c</sub> Frequency of frequency of c

You are fishing and caught :
10 carp, 3 perch, 2 whitefish, 1 trout, 1 salmon , 1 eel = 18 fish

First create counts of all frequencies

N<sub>1</sub> - which is count of everything that appears once = 3
N<sub>2</sub> - which is count of everything that appears twice = 1
N<sub>3</sub> - which is count of everything that appears thrice = 0

How likely is it that next species is trout?
- 1/18

How likely is it that next species is new (catfish or bass)
- Let's use our estimate of things we saw ONCE to estimate new things
- 3/18 since N<sub>1</sub> = 3)

Assuming so, how  likely is it that next species is trout?
- Must be less than 1/18
- How to estimate?

P<sub>GT</sub>(things with zero frequency) = N<sub>1</sub>/N

C<sup>*</sup> = (C+1) N<sub>C+1</sub>/N<sub>C</sub> / N

##### Unseen (bass or catfish)
c = 0 
MLE =  0/18
P<sub>GT</sub>(unseen) = N<sub>1</sub>/N = 3/18

##### Seen once trout
c= 1
MLE p = 1/18
Adjusted after bringing in probabilities of unknown
C(trout) = 2* N<sub>2</sub>/N<sub>1</sub>

= 2/3 / 18

## Absolute Discounting Interpolation

Probablity of absolute discounting = 
Discounted bigram + Interpolation Weight
 
[The need for measures other than accuracy](https://www.youtube.com/watch?v=jrAyRCa7aY8&list=PLLssT5z_DsK8HbD2sPcUIDfQ7zmBarMYv&index=30&t=184s)

When the true positives are miniscule compared to the true negatives, it might be really easy to hit an accuracy value. 
 
![Wighted Harmonic Mean of recall and Precision](https://github.com/sjtalkar/sjtalkar.github.io/blob/main/Wighted%20Harmonic%20of%20recall%20and%20Precision.JPG)
 

 
# Stationarity
 
 The window-based statistical parameters of a stationary time series can be estimated in a meaningful way because the parameters do not vary over different
time windows. In such cases, the estimated statistical parameters are good predictors of future behavior. On the other hand, the current mean, variances, and statistical correlations of the series are not necessarily good predictors of future behavior in regression-based forecasting models for nonstationary series. Therefore, it is often advantageous to convert nonstationary series to stationary ones before forecasting analysis. After the forecasting has been performed on the stationary series, the predicted values are transformed back to the original representation, using the inverse transformation.
 
 White noise is the strongest form of stationarity with zero mean, constant variance, and zero covariance between series values separated by a fixed lag.
 
The model you use to predict and forecast in a time series depends on whether the series is stationary or non-stationary.
The characteristics of a stationary series :
 Constant mean (the trend line is parallel to the time axis) $\mu$
 Constant variance -volatility the height of the series is constant
 No seasonality - correlation between lags is 0 in the case of white noise
 
 
 On the other hand, the current mean, variances, and statistical correlations of the series are not necessarily good predictors of future behavior in regression-based
forecasting models for nonstationary series. Therefore, it is often advantageous to convert nonstationary series to stationary ones before forecasting analysis.

**This is referred to as covariance stationarity: The key observation is that it is much easier to either obtain or convert to series that exhibit weak stationarity properties. In such cases, unlike white noise, the mean of the series, and the covariance between approximately adjacent time series values may be nonzero but constant over time.  Useful for forecasting models that are dependent on specific parameters such as the mean and covariance.**

**In other nonstationary series, the average value of the series can be described by a trend-line that is not necessarily horizontal, as required by a stationary series. Periodically, the series will deviate from the trend line, possibly because of some changes in the generative process, and then return to
the trend line. This is referred to as a trend stationary series.Also very useful for creating effective forecasting models.**
 
 A common approach used for converting time series to stationary forms is differencing. In differencing, the time series value <sub>i</sub> is replaced by the difference between it and the previous value. Therefore, the new value y′i is as follows:
y′i = y<sub>i</sub> − y<sub>i−1</sub> 
If the series is stationary after differencing, then an appropriate model for the data is:
yi+1 = y<sub>i</sub> + e<sub>i+1</sub>
Here, ei+1 corresponds to white noise with zero mean. A differenced time series would have t−1 values for a series of length t because it is not possible for the first value to be reflected in the transformed series. 
 
In some cases, such as geometrically increasing series, the logarithm function is applied to the values in the series, before the differencing operation. For example, consider a time series of prices that increase at an approximately constant inflation factor. It may be useful to apply the logarithm function to the time series values, before the differencing operation. 
 
 
 
A different approach is to use seasonal differences when it is known that the series is stationary after seasonal differencing. The seasonal differences are defined as follows:
y′i = yi − yi−m (14.13)
Here m is an integer greater than 1.


### Handling missing values 

It is common for time series data to contain missing values. Furthermore, the values of the series may not be synchronized in time when they are collected by independent sensors. It is often convenient to have time series values that are equally spaced and synchronized across different behavioral attributes for data processing.

The most common methodology used for handling missing, unequally spaced, or unsynchronized values is **linear interpolation** - think IBR and terms. The
idea is to create estimated values at the desired time stamps.
y = y<sub>i</sub> + ((t-t<sub>i</sub>)/(t<sub>j</sub> - t<sub>i</sub>) * (y<sub>j</sub> - y<sub>i</sub>)

## Noise Removal
[Think R squared and the variability that cannot be explained by the model]
The approach used by most of the noise removal methods is to remove **short-term fluctuations**. It should be pointed out that the distinction between noise and interesting outliers is often a difficult one to make. Interesting outliers are fluctuations, caused by specific aspects of the data generation process, rather than artifacts of the data collection process.
- Binning
- Moving Average (rolling average) smoothing : similar to binning BUT
The main difference is that a bin is constructed starting at each timestamp in the series rather than only the timestamps at the boundaries of the bins. Therefore,
the bin intervals are chosen to be [t1, tk], [t2, tk+1], etc. This results in a set of overlapping interval.
**CONS** :Short-term trends are sometimes lost because of smoothing. Larger bin sizes result in greater smoothing and lag.
- Exponenetial smoothing
In exponential smoothing, the smoothed value y′i is defined as a linear combination of the current value yi, and the previously smoothed value y′i −1.




###  Autoregressive Models
> Univariate time series contain a single variable that is predicted using autocorrelations.

 Autocorrelations represent the correlations between adjacently located timestamps in a series. Typically, the behavioral attribute values at adjacently located timestamps are positively correlated. The autocorrelations in a time series are defined with respect to a particular value of the lag L. Thus, for a time series y1, . . . yn, the autocorrelation at lag L is defined as the Pearson coefficient of correlation between yt and yt+L.
 
Autocorrelation(L) =
Covariancet(yt, yt+L) / Variancet(yt)

The autocorrelation always lies in the range [−1, 1], although the value is almost always positive for very small values of L, and gradually drops off with increasing lag L. The positive correlation ** is a result of the fact that adjacent values of most time series are very similar,**  though the similarity drops off with increasing distance. High (absolute) values of the autocorrelation imply that the value at a given position in the series can be predicted as a function of the values in the immediately preceding window. This is, in fact, the key property that enables the use of the autoregressive model.
 
 
 In the autoregressive model, the value of yt at time t is defined as a linear combination
of the values in the immediately preceding window of length p.
yt =
<sup>p</sup><sub>i=1</sub>[$/sigma]ai · y<sub>t−i</sub> + c + ǫt 
 
A model that uses the preceding window of length p is referred to as an AR(p) model. The values of the regression coefficients a1 . . . ap, c need to be learned from the training data. The larger the value of p, the greater the lag that one is willing to incorporate in the autocorrelations. The choice of p should be guided by the level of autocorrelation.
 
Note that the model can be used effectively for forecasting future values, only if the key properties of the time series, such as the mean, variance, and autocorrelation do not change significantly with time. Because the autocorrelation often reduces with increasing values of the lag L, a value of p should be selected, so that the autocorrelation at lag L = p is small.
The coefficients a1, . . . ap, c can be approximated with least-squares regression, to minimize the square-error of the overdetermined system.

**Specifically, the R2-value **
which is also referred to as the coefficient of determination, measures the ratio of the white
noise to the series variance:
R<sup>2</sup> = 1− Mean<sub>t</sub> (ǫ<sup>2</sup>t )/ Variancet(y <sub>t</sub>)

The coefficient of determination quantifies the fraction of variability in the series that is explained by the regression, as opposed to random noise. It is therefore desirable for this coefficient to be as close to 1 as possible.

## > Autoregressive Moving Average Models

While autocorrelation is a useful predictive property of time series, it does not always explain all the variations. In fact, the unexpected component of the variations (shocks), does impact future values of the time series. This component can be captured with the use of a moving average model (MA). The moving average model predicts subsequent series values on the basis of the past history of deviations from predicted values. A deviation from a predicted value can be viewed as white noise, or a shock. This model is best used in scenarios where the behavioral attribute value at a timestamp is dependent on the history of shocks in the time series,
rather than the actual series values.

y<sub>t</sub> = <subi=1</sub>[$/sigma]qi=b<sub>i</sub> · ǫ<sub>t−i</sub> + c + ǫ<sub>t</sub>

The parameter c is the mean of the time series. The values of b1 . . . bq are the coefficients that need to be learned from the data. The moving average model is quite different from the autoregressive model, in that it relates the **current value to the mean of the series and the previous history of deviations from
forecasts,** rather than the actual values.

A more general model may be obtained by combining the power of both the autoregressive model and the moving average model. The idea is to learn the appropriate impact of
both the autocorrelations and the shocks in predicting time series values. The two models can be combined with p autoregressive terms and q moving average terms. This model is referred to as the ARMA model.

yt = p i=1 ai · yt−i + q i=1 bi · ǫt−i + c + ǫt

In general, it is advisable to select the values of p and q as small as possible, so that the model fits the data well


 
 ####  Multivariate Forecasting with Hidden Variables
 
 In practice, a given application may have thousands of time series, and there may be significant correlations both across different series and across time. Therefore, models are required that can combine the autoregressive correlations with the cross-series correlations for making forecasts.
 
 1. Construct the d × d covariance matrix of the multidimensional time series. Let the d × d covariance matrix be denoted by C. The (i, j)th entry of C is the covariance
between the ith and jth series. This step is identical to the case of multidimensional data, and the temporal ordering among the different values of Yi is not used at this stage. Thus, the covariance matrix only captures information about correlations across series, rather than correlations across time.
2. Determine the eigenvectors of the covariance matrix C as follows:
C = PΛPT 
Here, P is a d×d matrix, whose d columns contain the orthonormal eigenvectors. The matrix Λ is a diagonal matrix containing the eigenvalues. Let Ptruncated be a d × p
matrix obtained by selecting the p ≪ d columns of P with the largest eigenvalues. Typically, the value of p is much smaller than d. This represents a basis for the hidden series with the greatest variability.
3. A new multivariate time series with p hidden time series variables is created. Each d-dimensional time series data point Yi at the ith timestamp is expressed in terms of a p-dimensional hidden series data point. This is achieved by using the p basis vectors derived in the previous step. Therefore, the p-dimensional hidden value Zi = (z1i . . . z<sup>p</sup><sub>i<sub> )
is derived as follows:
Z<sub>i</sub> = Y<sub>i</sub>P<sub>truncated</sub>\
 
 
 https://stats.stackexchange.com/questions/119746/what-is-the-proper-association-measure-of-a-variable-with-a-pca-component-on-a/119758#119758
 https://stats.stackexchange.com/questions/143905/loadings-vs-eigenvectors-in-pca-when-to-use-one-or-another
 
In PCA, you split covariance (or correlation) matrix into scale part (eigenvalues) and direction part (eigenvectors). You may then endow eigenvectors with the scale: loadings. So, loadings are thus become comparable by magnitude with the covariances/correlations observed between the variables, - because what had been drawn out from the variables' covariation now returns back - in the form of the covariation between the variables and the principal components. Actually, loadings are the covariances/correlations between the original variables and the unit-scaled components. This answer shows geometrically what loadings are and what are coefficients associating components with variables in PCA or factor analysis. 
 
 
 
 
 
 










