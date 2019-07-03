**Financial Machine Learning in Asset Management**

1.  Reinforcement learning and anomaly detection.

2.  Visualisation and normal Pythonic functions

3.  Cash Flow

4.  Remember to red through paper ML finance notes.

<!-- -->

1.  Such a beautifully easy RL algorithm:
    <https://github.com/teddykoker/blog/blob/master/notebooks/trading-with-reinforcement-learning-in-python-part-two-application.ipynb>

<!-- -->

1.  Advances and his met-labelling technique: to be honest this is more
    about writing:
    <https://github.com/timothyyu/ml_monorepo/blob/master/finance_ml/examples/Labeling.ipynb>

2.  Your restaurant valuation example comes to mind.

"If you are like me, you have wasted exasperating hours studying journal
articles, trying to figure out exactly how the author's algorithm works
and what parameter choices he or she prefers, when a short code would
have answered these questions immediately. Code segments appear in
rather few journal articles these days, and this is a trend that I hope
we can persuade publishers to reverse." [Lloyd N.
Trefethen](https://people.maths.ox.ac.uk/trefethen/publication/PDF/2005_114.pdf)
for Oxford University Computing Laboratory.

Only use positions sizing for discrete trading strategies, then kelly
criterion. the Kelly-optimal portfolio lies on Markowitz Efficient
Frontier. You know what is great, when you use etf's its survivor bias
free.

ML notes:

Like any good technology, ML aggregates hundreds to thousands of signals
to indicate the persuasion of the price to move in a particular
direction. A lot of good data sits on government websites, you just have
to find it. Using full Kelly is far too aggressive, its far better to
use Half-Kelly and set your risk at half the optimal.

There is only one reliable law in finance, if you want to know the value
of something find something similar whose price the market tells you.

Machine learning are used to investigate regime changes such as the
prediction of financial market downturns. Algorithms are being tested on
order book data of exchanges, such as prices volume and timing
properties of buyers and sellers for all instruments to determine how to
minimise the cost of executing trades. At an individual trade level, the
algorithm shaves off fractions of a percent on the trading cost; but
this has significant consequence for a large book with high volume of
trades.

Although the commercial potential of AI is enormous, trading results
will side with groups with bigger machines or more innovative data
scientists.

Machine learning's (ML) effect on finance takes a few forms: 1.
academic, 2. front office and, 3. back office. Within academia there are
two notably different ways of applying machine learning, these
differences are divided between traditional and mathematical finance.
This paper will delve into both academic and industry machine learning.
The front office does and will keep on benefiting from academic research
by traditional and mathematical academic finance research. Academic
research generally require some transformation to make it 'practical'.
Both the front and back office benefits from business machine learning
(BML), which is simply machine learning that improves the automatability
of work processes. This paper applies machine learning models from an
applied finance side as opposed to an automation side. It thus has a
financial machine learning (FinML) as opposed to BML. First, a little
bit of self-flagellating, fi this is the worst and most uninteresting
topic to apply machine learning models too, if it was not for the
bright-minds working on this problem and the vast amount of data flagged
open for this problem, the field wouldn't be as far as it is today. I
also think that I am talking for a lot of model developers, in that they
are simply looking at finance to learn, maybe some also want to make
money. Second this paper is not meant to be read, it's a ctr+fkr, let
your eyes guide you and follow the blue links for gold.

Although we often see a mismatch between industry and academia; that is
not that much the case for machine learning, it is a tightly weaved
between academia and industry, not just in finance but in all fields
utilising the technology. The demand for innovation is flowing down from
FIs and reaching academia. As a result machine learning applications in
traditional and mathematical finance once more splits into two, this
time for a more banal reason, regulation. Financial machine learning is
divided into groups that work on the interpretability and explainability
of their models, let's call them the b-hatters and groups that work on
the predictive value, the y-hatters. The b-hatters focus on developing
models that would please causal theorists and regulators and the
y-hatters side with prediction science and hedge-funds.

In saying that not a lot of innovation has been driven by the demand for
alpha, instead finance have become very adoptive of AI innovations in
other fields, this feels very reminiscent to the econophysics
revolution. The prevailing strategy is one of experimentation, 'oh that
convolutional neural network that outperformed Mr. Go player
extraordinaire can easily be adapted for time series data'. 'Yeah, why
not give it a shot'. And secondly one of acquisition, acquiring
expertise and acquiring data. Sounds just like the econophysics
revolution, get data, get physicist and allow them to run wild (for a
while), let them test the waters like a meta-level humanoid
reinforcement learning machine. In saying that some innovation has
trickled down from finance, such as the pandas project, and now apache
arrow from Wes McKenzie.

The b-hatting generally sides with traditional finance and y-hatting
with financial mathematics and engineering. This might turn out to be a
false dichotomy, in my own research I have found both to be extremely
important for hypothesis and prediction tasks. As we dig a little deeper
we come to conclude that finance is merely an embodiment of services.
Financial services can broadly be divided into, corporate and retail
banking, investment banking, insurance services, payment services,
investment services, financial management, and other advisory services.
In this paper the focus in on investment services. This paper will pay
specific attention to asset management and brokerage services as opposed
to wealth management, private equity and venture capital management, not
to say that they won't benefit from the same machine learning
developments.

Machine learning becomes very valuable as a technology replacer, in the
same way that the coffee from a plunger might taste just as good as the
coffee from an expensive machine, machine learning will most likely
produce the same results as previously obtain, but for less of an
effort. Machine learning's benefit is not the creation of alpha, it is
the maintenance of alpha. Machine learning is important to know, not
just because of its applied benefit, but also to know its weaknesses.
Future discretionary firms would want to market as a means to side-step
technologically enabled investment by asking the question, what can't
machines and machine-human collaboration do well, and focusing on those
areas.

In fact there is not a lot of people in the field that actually knows
what they are doing at least not yet. Books are written and titles are
flung around while management comes to grip what this revolution really
means to their teams, does it mean we should fundamentally rebuild our
systems from the ground up or just fling more data to our current models
and change employee titles hire those hungry for action grads and
appease clients.

Front office asset management can be broken into the following tasks:
portfolio optimisation, risk measures, capital management,
infrastructure and deployment, and sales and marketing. In this paper
the focus is on portfolio optimisation, risk and capital management.

1.  **Portfolio Optimisation:**

    a.  Trading Strategies

    b.  Weight and Strategy Optimisation.

2.  **Risk Measurement:**

    c.  Extreme Risk

    d.  Simulation

3.  **Capital Management**

    e.  Kelly Criterion

4.  **Infrastructure and Deployment**

    f.  Cloud

    g.  Diagnostics

    h.  Monitoring and Testing

5.  **Sales and Marketing**

    i.  Product Recommendation

    j.  Customer Service

**Techniques**

Machine learning techniques can largely be broken into the following
techniques, the processing of unstructured data, supervised learning,
validation techniques, unsupervised learning and reinforcement learning.
In finance, I see a very bright future for unstructured data processing,
natural language processing in finance, further I see a bright future
for unsupervised learning and reinforcement learning.

1.  **Data Processing**

    a.  Natural Language Processing

        i.  Text Extraction

        ii. Word Embeddings

        iii. Topic Modelling

    b.  Image and Voice Recognition

    c.  Feature Generation

2.  **Supervised Learning**

    d.  Algorithms

        iv. Gradient Boosting

            1.  LightGBM

            2.  XGBoost

            3.  Constraint

        v.  Neural Networks

            4.  CNN

            5.  RNN

            6.  GAN

    e.  Tasks

        vi. Regression

        vii. Classification

    f.  Analysis

        viii. Cross Sectional

        ix. Time Series

3.  **Validation Techniques**

    g.  Visual Exploration

    h.  Table Exploration

    i.  Feature Importance

    j.  Feature Selection

    k.  Cross Validation

4.  **Unsupervised Learning**

    l.  Traditional

        x.  Dimensionality Reduction

        xi. Clustering

        xii. Anomaly Detection

        xiii. Group Segmentation

    m.  Neural and Deep Learning

        xiv. Autoencoders

        xv. Boltzmann Machines

        xvi. Deep Belief Networks

        xvii. Generative Adversarial Networks

    n.  Semisupervised Learning

        xviii.  Mixture Models and EM

        xix. Co-Training

        xx. Graph Based

        xxi. Humans

5.  **Reinforcement Learning**

    o.  Markov Decision Process and Dynamic Programming

    p.  Monte Carlo Methods

    q.  Temporal Difference Learning

    r.  Multi-armed bandit

    s.  Deep (Recurrent) Q Network

    t.  Actor Critic Network

    u.  Inverse Reinforcement Learning

**Portfolio Optimisation**

**[Trading Strategies]{.underline}**

Within portfolio optimisation we have trading strategies and strategy
and or security weight optimisation. First we will look at trading
strategies after which we will consider weight optimisation. The first
trading strategy, and the one I find particularly interesting due to its
classifying nature are [event-driven arbitrage]{.underline}, this has
been my own are of interest, where I focus on earnings surprise,
bankruptcy and restaurant facility closure prediction. Then others
include [factor investing]{.underline} in which a fund automatically
buys assets that exhibit a trait associated with promising investment
returns. [Risk parity]{.underline} is up next, where one diversifies
across assets according to the volatility they exhibit. Where one asset
class's volatility exceeds another, rebalancing can occur by selecting
individual units within each asset class, or using leverage. Then you
have [systematic global macro]{.underline} which relies on macroeconomic
principles to trade across asset classes and countries. Other techniques
include [statistical arbitrage]{.underline}, it seeks mispricing by
detecting security relationships and potential anomalies believing the
anomaly will return to normal. [CTA]{.underline} is another strategy
where one takes a position in an asset only after a trend appears in the
pricing data and lastly normal [fundamental]{.underline} investing
strategies that rely on manager expertise but enhanced by technologies.
I am not going to give oxygen to high-frequency trading. In the
following examples, sometimes the code would be shown, sometimes the
data, and sometimes both.

***In many of the strategies below, I might include the data but not the
data processing, modelling or trading steps/code and visa versa. This is
mostly due to copyright and data policy concerns. Where the work is my
own, I try to, as best I can provide for the full pipeline.***

**Tiny CTA**

Credit *Man Group*

Momentum refers to the persistence in returns. Winners tend to keep on
winning and losers keep on losing. In this momentum strategy, a
CTA-momentum signal is built on the cross-over of exponentially weighted
moving averages. In essence one selects three sets of time-scales with
each set consisting of short and long exponentially weighted moving
averages (EWMA), 2.
$S_{k} = \left( 8,16,32 \right),\ L_{k} = \left( 24,48,96 \right).$ Each
of these numbers translates in a decay factor that is plugged into the
standard definition of EWMA. The half-life is given by:

$$HL = \ \frac{log(0.5)}{log(\frac{n - 1}{n})}$$

For each $k = 1,2,3$ one calculate

$$x_{k} = EWMA\left\lbrack P \middle| S_{k} \right\rbrack - EWMA\lbrack P|L_{k}\rbrack$$

The next step is to normalise with a moving standard deviation as a
measure of the realised 3-months normal volatility (PW=63)

$$y_{k} = \frac{x_{k}}{Run.StDev\lbrack P|PW\rbrack}$$

The series is normalised with the realised standard deviation over the
short window (SW=252)

$$z_{k} = \frac{y_{k}}{Run.StDev\lbrack y_{k}|PW\rbrack}$$

Next one calculate an intermediate signal for each $k = 1,2,3$ via a
response function $R$

$$\left\{ \begin{matrix}
u_{k} = R(z_{k}) \\
R\left( x \right) = \ \frac{xexp(\frac{- x^{2}}{4})}{0.89} \\
\end{matrix} \right.\ $$

Then the final CTA momentum signal is the weighted sum of the
intermediate signal where we have chosen equal weights,
$w_{k} = \frac{1}{3}$

$$S_{\text{CTA}} = \sum_{K = 1}^{3}{w_{k}u_{k}}$$

Here I did not include the actual machine learning code, however this is
relatively easy to add. Here one should look at all the arbitrarily
chosen parameters like $S_{k},\ L_{k,\ }\text{and\ }w_{k}$ and optimise
with a reinforcement learning algorithm using gradient descent to
achieve the best Sharpe ratio or simply returns. See the next strategy
for an implementation of a tiny reinforcement-learning algorithm.

> *Resources:*
>
> See this
> [paper](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2695101)
> and
> [blog](https://www.linkedin.com/pulse/implement-cta-less-than-10-lines-code-thomas-schmelzer/)
> for further explanation.

**[Data](https://drive.google.com/open?id=12BB8KpFYJSx41yvHhtoLYE_ZZOHNamP8),
[Code](https://drive.google.com/open?id=1EwbHhBZL_PRTphR25EbMQA9dV7jC4CjT)**

**Tiny RL**

In this example we will make use of gradient descent to maximise a
reward function. The Sharpe ratio will be used as the reward function.
The Sharpe ratio is used as an indicator to measure the risk adjusted
performance of an investment over time. Assuming a risk free rate of
zero, the Sharpe ratio can be written as:

$$S_{T} = \frac{A}{\sqrt{B - A^{2}}}$$

Further, to know what percentage of the portfolio should but the asset
in a long only strategy, we can specify the following function which
will generate a value between 0 and 1.

$$F_{t} = tanh(\theta^{T}x_{t})$$

The input vector is
$x_{t} = \lbrack 1,\ r_{t - M},\ \ldots,\ F_{t - 1}\mathbf{\rbrack}$
where $r_{t}\mathbf{\ }$is the percent change between the asset at time
$t$ and $t - 1$, and *M* is the number of time series inputs. This means
that at every step the model will be fed its last position and a series
of historical price changes that is can use to calculate the next
position. Once we have a position at each time step, we can calculate
our returns $R$ at each time step using the following formula. In this
example, $\delta$ is the transaction cost.

$$R_{t} = F_{t - 1}r_{t} - \delta|F_{t} - F_{t - 1}|$$

To perform gradient descent one must compute the derivative of the
Sharpe ratio with respect to theta, or $\frac{dS_{T}}{\text{dθ}}$ using
the chain rule and the above formula. It can be written as:

$$\frac{dS_{T}}{\text{dθ}} = \sum_{t = 1}^{T}{\left( \frac{dS_{T}}{\text{dA}}\frac{\text{dA}}{dR_{t}} + \frac{dS_{T}}{\text{dB}}\frac{\text{dB}}{\text{dθ}} \right).(}\frac{dR_{t}\ }{dF_{t}}\frac{\text{dF}}{\text{dθ}} + \frac{dR_{t}}{dF_{t - 1}}\frac{dF_{t - 1}}{\text{dθ}})$$

*Resources:*

> See this
> [paper](http://cs229.stanford.edu/proj2006/Molina-StockTradingWithRecurrentReinforcementLearning.pdf)
> and/or [blog](https://teddykoker.com/) for further explanation.
>
> [Data](https://drive.google.com/open?id=1CgwzyNqzizJYT8OxN9gD4f9t02gr7ghb),
> [Code](https://drive.google.com/open?id=1IRrR6kWjunERzZqrszJ9_q-C1Yj5L0Qj)

**Tiny VIX CMF**

*With Andrew Papanicolaou*

VIC and futures on the VSTOXX are liquid so are ETNs/ETFs on VIX and
VSTOXX. Prior research shows that the curves exhibit stationary
behaviour with mean reversion toward a contango. First one can imitate
the futures curves and ETN price histories by building a model and then
use that model to manage the negative roll yield. The Constant Maturity
Futures (CMF) can be specified as follows:

Denote $\theta = T - t$ to have constant maturity,
$\ V_{t}^{\theta} = F_{t,t + 0}$, for
$t \leq T_{1} \leq t + \theta \leq T_{2}$ define

$$a\left( t \right) = \ \frac{T_{2} - (t - \theta)}{T_{2} - T_{1}}$$

And note that:

-   $0 \leq a\left( t \right) \leq 1$

-   $a\left( T_{1} - \theta \right) = 1\ and\ a\left( T_{2} - \theta \right) = 0$

-   $\text{linear\ in\ t}$

The CMF is the interpolation,

$$V_{t}^{\theta} = a\left( t \right)F_{t}^{T1} + \left( 1 - a\left( t \right) \right)F_{t}^{T2}$$

Where $V_{t}^{\theta}$ is a stationary time series.

One can then go on to define the value of the ETN so that you take the
roll yield into account. I simply want to focus on maturity and
instrument selection, and therefore ignored the roll yield and simply
focused on the CMFs. But, if you are interested, the value of the ETN
can be obtained as follow.

$$\frac{dl_{t}}{l_{t}} = \ \frac{a\left( t \right)dF_{t}^{T1} + \left( 1 - a\left( t \right) \right)dF_{t}^{T2}}{a\left( t \right)F_{t}^{T1} + \left( 1 - a\left( t \right) \right)F_{t}^{T2}} + rdt$$

Where *r* is the interest rate.

Unlike the previous approach, this strategy makes use of numerical
analyses before a reinforcement learning step. First out of seven
securities (J), establish a matrix of 1 and 0 combinations for
simulation purpose. Creating a matrix of $2^{7} - 2 = 126$ combinations.
Then use a standard normal distribution to randomly assign weights to
each item in the matrix. Create an inverse matrix and do the same. Now
normalise the matrix so that each row equals one to force neutral
portfolios. The next part of the strategy is to run this exact same
simulation N (600) number of times depending on your memory capacity as
this whole trading strategy is serialised. Thus each iteration (N)
produces normally distributed long and short weights (W) that have been
calibrated to initial position neutrality (Long Weights = Short
Weights). The final result is 15,600 trading strategies.

The next part of this system is to filter out strategies with the
following criteria. Select the top 5% percent of strategies for the
highest median cumulative sum over the period. Of that selection, select
the top 40% for the lowest standard deviation. Of that group select 25%
again from the highest median cumulative sum strategies. Of the
remaining strategies iteratively remove high correlated strategies until
only 10 strategies remain. With that remaining 10 strategies, which have
all been selected using only training data, use training data again to
formulise a reinforcement learning strategy using a simple MLP neural
network with two hidden layers. And finally test the results on an out
of sample test set. No structure or hyperparameter selection was done on
a development set, as a result, it is expected that results can be
further improved.

> *Resources*:
>
> [Data](https://drive.google.com/open?id=1Yv2_mTjZMANoL9fM0ajOsOFEc9MJZAMU),
> [Code](https://drive.google.com/open?id=186j-gtkXCgzj06WCWDAU9yhYXP9SfgLu)

**Quantamental Strategies**

In my experience, Quantamental strategies are intelligent data driven
valuations. A good example of a machine learning Quantamental strategy
is a project I worked on in late 2017. The core goal of the project was
to estimate a fair valuation of privately owned restaurant chains. A
secondary consequence of the project was the ability to predict whether
a publicly traded restaurant was under of overvalued. To do this
alternative data was gathered on employee, customer, and shareholder and
management sentiment at the company and where possible the individual
locations level.
[Data](https://drive.google.com/open?id=1b0OXiSKnacEDftYKgov619SCfXwpcUWT)
was gathered from more than ten alternative sources using
[web-scrapers](https://drive.google.com/drive/folders/12aZ7vg_3HIdPYZ4GavYY7BjptlAPGFtc?usp=sharing).
This includes LinkedIn, Yelp, Facebook, Instagram and Glassdoor data.
What followed was the use of an open sourced gradient boosting model
from Yandex know as Catboost. In my final implementation I preferred a
model called XGBoost, this is not shown in the code.

Here is how one might come to understand a GBM model. The algorithms for
regression and classification only differ in the loss function used; the
method is otherwise the same. To create a GBM model we have to establish
a loss function, *L* to minimise, so as to optimise the structure and
performance of the model. This function has to be differentiable, as we
want to perform a process of steepest descent, which is an iterative
process of attempting to reach the global minimum of a loss function by
going down the slope until there is no more room to move closer to the
minimum. For the regression task, we will minimise the mean squared
error (MSE) loss function. The focus here is on ${f(\mathbf{x}}_{i})$ as
this is the compressed form of the predictor of each tree *i*.

+--+-------------------------------------------------------------------------+--+
|  | $$L(\theta) = \sum_{i}^{}{(\mathbf{y}_{\mathbf{i}} - {f(x}_{i}))}^{2}$$ |  |
|  |                                                                         |  |
|  | $$L(\theta) = \sum_{i}^{}{(\mathbf{y}_{i} - {y\hat{}}_{i})}^{2}$$       |  |
+--+-------------------------------------------------------------------------+--+

Further, it is necessary to minimise the loss over all the points in the
sample, $(\mathbf{x}_{i},\ y_{i})$:

+--+--------------------------------------------------------------------+--+
|  | $$f(\mathbf{x}) = \sum_{i = 1}^{N}{L(\theta)}$$                    |  |
|  |                                                                    |  |
|  | $$f(\mathbf{x}) = \sum_{i = 1}^{N}{L(y_{i},{f(\mathbf{x}}_{i}))}$$ |  |
+--+--------------------------------------------------------------------+--+

At this point we are in the position to minimise the predictor function,
${f(\mathbf{x}}_{i})$, w.r.t. x since we want a predictor that minimises
the total loss of $f(\mathbf{x})$. Here, we simply apply the iterative
process of steepest descent. The minimisation is done in a few phases.
The first process starts with adding the first and then successive
trees. Adding a tree emulates adding a gradient based correction. Making
use of trees ensures that the generation of the gradient expression is
successful, as we need the gradient for an unseen test point at each
iteration, as part of the calculation $f(\mathbf{x})$. Finally, this
process will return $f(\mathbf{x})$ with weighted parameters. The
detailed design of the predictor, $f(\mathbf{x})$, is outside the
purpose of the study, but for more extensive computational workings see
the next section. For a more in-depth elucidation see the endnote.

Among other things, the algorithm predicted that BJ's restaurants market
value was trading about 40% below its competitors with similar
characteristics; within the year, the price just about doubled compared
to competitors. This project was especially interesting because no
company specific financial data was used as inputs to the model, and the
response variable was simply the market value. In the process, I also
developed an [interactive
report](https://github.com/firmai/interactive-corporate-report) that is
now open sourced. If you have a look at the report, the light blue line
signifies what the 'AI' recommends the cumulative return should have
been after 5 years, whereas the dark blue line is the cumulative return
over time for BJ's being about 40 percentage points lower than what the
AI believed to be a 'fair' market value. I am not going to make
everything open for this project, but the above links would give you
some crumbs[^1].

> *Resources:*
>
> [Web-scrapers](https://drive.google.com/drive/folders/12aZ7vg_3HIdPYZ4GavYY7BjptlAPGFtc?usp=sharing),
> [Data](https://drive.google.com/open?id=1b0OXiSKnacEDftYKgov619SCfXwpcUWT),
> [Code](https://drive.google.com/open?id=1PqtFfcr1ejreGr6XIoZCs8jsD7AccuL7),
> [Interactive
> Report](https://github.com/firmai/interactive-corporate-report).

**Event Driven Arbitrage**

In this section we can look at two event driven strategies, the first is
an earnings prediction strategy. For the classification task, the
response variable for the machine learning model is the occurrence of an
earnings surprise. An earnings surprise is simply defined as a
percentage change from the analyst\'s EPS expectation as described in
the data section and the actuals EPS as reported by the firm. In this
study, we include percentage thresholds, *s*, as a means of expressing
the magnitude of a surprise so as to construct various tests.

$\text{\ SURP}_{\text{itsx}} = 1\mathbf{\rightarrow}\ $Neutral;

$\text{SURP}_{\text{itsx}}\  = 2,\ \ where\ \ \ \ \frac{\ {\text{EPSAC}_{\text{it}} - EPSAN}_{\text{\ it}}}{\text{EPSAN}_{\text{it}}} - 1 > s\mathbf{\rightarrow}$
Positive;

$\ \text{SURP}_{\text{itsx}}\  = 0,\ \ wh\text{ere}\text{\ \ \ \ }\frac{\ {\text{EPSAC}_{\text{it}} - \text{EPSAN}}_{\ \text{it}}}{\text{EPSAN}_{\text{it}}} - 1 < - s\mathbf{\rightarrow}$
Negative.

To provide some clarity, *i* is the firms in the sample, *t* is the time
of the quarterly earnings announcement, *s* is the respective constant
surprise threshold, 5%, 10% or 15%, *x* is a constant percentage of the
sample selected sorted by date of earnings announcement, *EPSAN* is the
earnings per share forecast of analysts and *EPSAC* is the actual
earning per share measure as reported by the firm. This surprise measure
is simply the difference between the actual and expected EPS scaled by
the expected EPS.

Below is high-level pseudo code to provide a better understanding of
some of the core concepts of the black-box model such its relationship
with the training set, test set, prediction values, and metrics
$\left( \mathbf{1} \right)\ \ TrainedModel = ModelChoice;\ \left( \mathbf{2} \right)\ Predictions = TrainedModel(TestInputs)$;
$\left( \mathbf{3} \right)\ Metrics = Functions(TestTarget,\ Predictions)$.
For an earnings classification it can look as follows:

$\left( \mathbf{1} \right)\mathbf{\ }\ Classifier = XGBoostTreeClassifier\left( \text{Trai}n_{X},\ \text{SURP}_{its(x)},\ Paramaters \right);\ \left( \mathbf{2} \right)\text{\ \ }\text{PredSURP}_{\text{ith}} = Model\left( \text{Tes}t_{X} \right);$
$\left( \mathbf{3} \right)\ \ Metrics = Functions\left( \text{SURP}_{its(1 - x)},\ \text{PredSURP}_{\text{ith}} \right)$.

The prediction values, $\text{PredSURP}_{\text{ith}},$ of the classifier
is a categorical variable that falls within the values
$\left\{ 0,1,2 \right\} \rightarrow \ \{ Negative\ Surprise,\ No\ Surprise,\ Positive\ Surprise\}$,
for surprises of different thresholds, $h\ \{ 5\%,\ 10\%,\ 15\%\}$. If
we assume that the training set is 60% of the original data set, then
the training set\'s target value is $\text{SURP}_{its\ (60\%)}$, being
the first 60% of the dataset ordered by date. As a consequence, the test
set\'s target values are $\text{SURP}_{its\ (40\%)}$, the last 40% of
the dataset. The metrics for a classification task comprise of accuracy
(proportion of correctly classified observations), precision (positive
predictive value), recall (true positive rate), and confusion
matrices/contingency tables

The following expresses a simple strategy by going long on stocks that
are expected to experience a positive surprise tomorrow (t), at closing
today (t-1) and liquidating the stocks at closing tomorrow (t). The
stocks are equally weighted to maintain well-diversified returns for the
day, as there is on average only four firms in a portfolio of expected
surprises but there can be as few as one firms in a portfolio[^2].
Earnings surprises are often termed as soft events in books on event
driven investment strategies. Event driven investment forms a large part
of the hedge fund industry, accounting for about 30% of all \$2.9
trillion assets under management according to Hedge Fund Research 2015.
Event driven strategies are a great way to benefit from increased
volatility. The strategies recommended in this section fully invests
capital in each event, it is therefore smart to include some sort of
loss minimisation strategy.

As a result, the first strategy incorporates a stop loss for stocks that
fell more than 10%; 10% is only the trigger, and a conservative loss of
20% is used to simulate slippage. This is done by comparing the opening
with the low price of the day. There are an endless number of
opportunities and strategies that can be created; it is, therefore,
important to select simple strategies not to undermine the robustness of
this study. In saying that the choice of slippage is not backed by any
literature and is an arbitrary albeit conservative choice to the
strategy. I have created three different portfolios consisting of firms
that are expected to experience a positive surprise above a 5%, 10%.,
and 15% threshold. P = {$P_{5},\ P_{15},\ P_{30}\}$ is a ) is a simple
return calculation for an earnings announcement of firm *i* at time *t*,
where the daily low price, $\text{Pl}_{\text{ti}}$, is not more than
-10% lower than the closing price $P_{\left( t - 1 \right)i}$. If it is,
then a slippage loss of -20% is allocated to the return quarter of that
firm.

$$\text{\ \ R}_{\text{jt}} = \frac{\left( P_{\text{ti}} - P_{\left( t - 1 \right)i} \right)}{P_{\left( t - 1 \right)i}}\ ,\ \ if\ \ \frac{\ \text{Pl}_{\text{ti}} - P_{\left( t - 1 \right)i}}{P_{\left( t - 1 \right)i}} > - 10\%,\text{\ \ R}_{\text{jt}} = - 20\%$$

In this equation, $i$ is the firms that have been predicted to
experience an earnings surprise at *t*. $P_{\left( t - 1 \right)i}$ is
the closing price of the common stock of firm *i* on date *t-1*.
$\text{Pl}_{\text{ti}}$ is the daily low of the common stock price of
firm *i* on date *t-1*. The equal weighted return of a portfolio of
surprise firms is then calculated as so,

  -- ---------------------------------------------------------------------------- -----
     $$R_{\text{pt}} = \ \frac{1}{n}\sum_{j = 1}^{n_{\text{pt}}}R_{\text{jt}}$$   (1)
  -- ---------------------------------------------------------------------------- -----

In this equation, $j$ is all the firms that experience surprises on date
$t$. $R_{\text{it}}$ is the return on the common stock of firm *j* on
date *t*. $n_{\text{pt}}$ is the number of firms in portfolio *p* at the
close of the trading on date *t-1*. The equation below is the
five-factor asset pricing model. In this equation, $R_{\text{pt}}\ $is
the return on portfolio $p$ for period $t$, that has been predicted to
experience an earnings surprise event at *t*. $R_{\text{Ft}}$ is the
risk-free rate. $R_{\text{Mt}}$ is the value-weighted return of the
market portfolio. $\text{SM}B_{t}$, $\text{HM}L_{t}$, $\text{RM}W_{t}$
and $\text{CM}A_{t}$ are the respective differences between diversified
portfolios of small stocks and big stocks, high and low B/M stocks,
robust and weak profitability stocks, and low and high investment
stocks. To perform the regressions, the respective daily values were
obtained from Kenneth French\'s website[^3].

$$\text{\ \ \ R}_{\text{pt}}\ –\ RF_{t}\  = a_{i} + bi\ (R_{\text{Mt}}–R_{\text{Ft}}) + s_{i}\text{SM}B_{t} + h_{i}\text{HM}L_{t} + r_{i}\text{RM}W_{t} + c_{i}\text{CM}A_{t} + e$$

$$R_{\text{cum}} = \prod_{t = 1}^{m}{\left( 1 + R_{\text{pt}} \right) - 1}$$

Figure 1: Portfolio Value - Large Firms 15% Surprise Prediction
Strategies

[\[CHART\]]{.chart}

  -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
  **This portfolio reports the cumulative returns of a buying and holding positive and negative surprise portfolios for all firms in the sample with a market value of \$10 Billion. On average, there are about four firms for each portfolio day. On days where no trading surprises occur position in the market is taken. The band in the middle is the significance band obtained by a Monte Carlo simulation from randomly predicting and taking a position in 774 firms before an earnings announcement. The chart also reports the cumulative portfolio return of the market as calculated from the market returns obtained from French\'s website. The chart shows that negative surprises, pretty much tracks the market portfolio. It is possible that some return can be earned by shorting these surprises over certain periods, but on average it is not a very profitable strategy due to the small amount of shorting opportunities. In total, there is 2944 trading days, for the long strategy, 215 of these days are returns from earnings surprises comprising 774 firms and for the short strategy 62 days comprising 234 firms.**
  -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

[**Code**](https://drive.google.com/open?id=1KtGauKizS8QISuDCW0SwIxbYPeBwTQxF)

**Statistical Arbitrage:**

[Pairs Trading]{.underline}

Pairs trading is a simple statistical arbitrage strategy that involves
finding pairs that exhibit similar historic price behaviour and then
once they divergence betting on the expectation that they will
convergence. In a universe of assets over time, $\mathbf{X}_{t}$, pick
individual assets, $X_{t}^{x}$, so that
${corr(X}_{t}^{x_{1}},X_{t}^{x_{2}})$ exceeds some threshold $P$. We can
let $X_{t}^{1}$ and $X_{t}^{2}$ denote the prices of two correlated
stocks, if we believe in the mean-reverting nature[^4] and the ground
truth of the shared correlation, we can initiate a mean-reverting
process ($Z_{t}$) as follow, $Z_{t} = X_{t}^{1} - K_{0}X_{t}^{2}$, where
$K_{0}$, is a positive scalar such that at initiation $Z_{t} = 0$.

This mean reverting process, i.e., the change in mean, is governed by
$\text{dZ}_{t} = a\left( b - Z_{t} \right)dt + \sigma dW_{t},\ \text{\ Z}_{0} = 0$,
where $a > 0$ is the rate of reversion, $b$ the equilibrium level,
$\sigma > 0$ the volatility, and $W_{t}$ the standard Brownian motion.
This Brownian motion exists within the mean-reverting process, hence it
can't have drift. $Z_{t}$ is thus the pairs position, 1 share long in
$X_{t}^{1}$, $K_{0}$ shares short in $X_{t}^{2}$ and $\text{dZ}_{t}$
describes the underlying dynamic under strict mathematical assumptions.

Given the construction, we expect the prices of assets to revert to
normal. We can specify an easy and arbitrary trading rule, which
specifies that if $abs(X_{t}^{1} -$
${K_{0}X}_{t}^{2})/(\frac{1}{n}\sum_{t = 1}^{n}{abs(X_{t}^{1} - \ {K_{0}X}_{t}^{2}))}$
\> 0.1, then you sell the highly priced asset and buy the lowly priced
asset, and you do the reverse as soon as the above formula hits 0.5,
whereby you assume that this might be the new natural relational
equilibrium. In practice we also impose additional state constraints,
for example, we require, $Z_{t} \geq M$, where $M$ is the stop
loss-level in place for unforeseeable events and to satisfy a margin
call, further alterations should include transaction and borrowing
costs. Pair selection and trading logic can be much more involved than
the above. Instead of using simple trading rules, dynamic model based
approaches can be used to define the trading logic. The before mentioned
mean reversion process can be discretised into an AR process where the
parameters can be estimated iteratively and used in the trading rule,
this is called the Stochastic Spread Method. Interesting, within the
reinforcement learning algorithm you only have to do trading logic, the
pairs selection step can be skipped or swolled, I guess?

Most exit and entry strategies in pairs trading such as distance,
rolling OLS cointegration, and Kalman filter cointegration methods. One
can also make use of

The trading logic can be turned into an easy reinforcement-learning
problem. There can also be multiple methods to do the similarity
selection, see this. Pairs [trading
visualisation](https://github.com/marketneutral/pairs-trading-with-ML/blob/master/Pairs%2BTrading%2Bwith%2BMachine%2BLearning.ipynb)
in tsne. Here is another interesting one, pairs trading using variations
encoders,
(<https://github.com/ml-hongkong/stock2vec/blob/master/stock_clustering.ipynb>)

[RL:]{.underline}

A few researchers have done exactely that, see there
[paper](https://github.com/wywongbd/statistical-arbitrage-18-19/blob/master/reports/FYP_Final_Report_LZ2.pdf)
and associated
[notebooks](https://github.com/wywongbd/statistical-arbitrage-18-19/tree/master/jupyter_py).
. Use this notebook to play around with a non-machine learning python
notebook, [this
notebook](https://github.com/JerBouma/Statistical-Arbitrage-Algorithmic-Trading/blob/master/Pairs%20Trading%20Jupyter%20Notebook.ipynb).
Further look at this options arbitrage model, and see if you can
identify how you might include reinforcement learning to this problem
set, [options arbitrage
solution](https://github.com/JerBouma/Option-Arbitrage/blob/master/Options%20Arbitrage.ipynb)
[and a treasury future trading
script.](https://github.com/jerryxyx/TreasuryFutureTrading/blob/master/BackTestingScript.ipynb)

The approach specified in this example is model free; it is a rule-based
approach. In general, I would advise against cointegration and instead
expand the time granularity, i.e. move from daily to weekly returns.

**Factor Investing**

[This
analysis](https://docplayer.net/120877135-Industry-return-predictability-a-machine-learning-approach.html)
will look at using traditional factors as used in finance, in
combination with machine learning models to identify things. It might be
constrained at the level of selection, machines can be left to figure
these relationships out, the same can be said not just for the ratio
pairs but the ratios itself. It starts with factor collection and
pre-processing, then screening factors using correlation.

In this scenario, factors can be anything fundamental, be it industry
grouping return or Fama-French factors. The first example we will look
at is the use of machine learning tools to analyse industry return
predictability based on the lagged industry returns across the economy.
Annualised return on a strategy that long the highest and short the
lowest predicted returns, returns an alpha of 8%. In this approach, one
has to be careful for multiple testing and post-selection bias. In this
scenario a LASSO regression is used, firs of all we can formulate a
standard looking predictive regression framework:

$$\mathbf{y}_{\mathbf{i}} = a_{i}^{*}\mathbf{\gamma}_{\mathbf{T}} + \mathbf{X}\mathbf{b}_{\mathbf{i}}^{\mathbf{*}} + \mathbf{\varepsilon}_{\mathbf{i}}\ \ \ for\ i = 1,\ldots,N,$$

Where

$$y = \left\lbrack r_{i,1}\ldots r_{i,T} \right\rbrack\ \ ;\ X = \ \left\lbrack x_{1}\ldots x_{N} \right\rbrack\ \ ;x_{j} = \ \left\lbrack r_{i,0}\ldots r_{i,T - 1} \right\rbrack\ \ for\ j = 1,\ldots,N$$

$$b_{i}^{*} = \left\lbrack b_{i,1}^{*}\ldots b_{i,N}^{*} \right\rbrack\ \ ;\ \varepsilon_{i} = \ \left\lbrack \varepsilon_{i,1}\ldots\varepsilon_{i,T} \right\rbrack\text{\ \ }$$

In addition, the lasso objective
$\mathbf{(\gamma}_{\mathbf{T}}\mathbf{)}$ can be expressed as follows,
where $\vartheta_{i}$ is the regularisation parameter.

$$\underset{a_{1} \in \mathbf{R},\ b_{i} \in R^{N}}{\arg\min}(\frac{1}{2T}\left| \left| y_{i} - a_{i}\mathbf{\gamma}_{\mathbf{T}}\mathbf{- X}\mathbf{b}_{\mathbf{i}} \right| \right|_{\mathbf{2}}^{\mathbf{2}}\mathbf{+}\vartheta_{i}\left| \left| \mathbf{b}_{\mathbf{i}} \right| \right|_{\mathbf{1}}\mathbf{)}$$

The LASSO generally performs well in selecting the most relevant
predictor variables. Some argue that the LASSO penalty term over shrinks
the coefficient for the selected predictors. In that scenario, one can
use the selected predictors an estimate the coefficients using OLS. This
sub model -- OLS in this case -- can be replaced by any other machine
learning regressor. In fact the main and sub model can both be machine
learning regressor, the first selecting features and second predicting
the response variable based on those features.

For practice, you can have a look at the following repositories and see
if you can identify any machine learning use cases. [Factor
analysis]{.underline}: 1. [Mutual
funds](https://github.com/garvit-kudesia91/factor_analysis/blob/master/Factor%20Analysis%20of%20Mutual%20Funds.ipynb),
2. [Fama](https://github.com/ZhangZhiHu/assetPricing), 3.
[Equity](https://github.com/datudar/equity-risk-model/blob/master/equity_risk_model.py)
factors, 4. [Penalised](https://github.com/erikcs/penfmb) factors, 5.
Factor [momentum](https://github.com/aconstandinou/factor_momentum) and
6.
[Residual](https://factorinvestingtutorial.wordpress.com/9-residual-momentum-david-blitz/)
momentum.

**Systematic Global Macro**

[Oil and
currency](https://github.com/je-suis-tm/quant-trading/tree/master/Oil%20Money%20project)
relationships. [Seems to be
doing](https://github.com/robcarver17/pysystemtrade) a bit of global
systematic trading. There is clearly not too much here, better follow
the CTA strategies. Returns across
[industries](https://github.com/druce/Machine-learning-for-financial-market-prediction/blob/master/Replicate%20Paper%20by%20Rapach%2C%20Strauss%2C%20Tu%2C%20Zhou.ipynb),
using ml.

When oil exists a bear market then the currency of oil producing nations
would also rebound. Instead of correlation, I would argue for a method
called causal correlation. I would get to it soon once I have two time
series. One needs a relative stabiliser currency to regress against.
That is unrelated to the currencies at hand. Something like the JPY is a
good candidate. Thus one would use the price of other currencies and
Brent to identify whether the Norwegian currency is undervalues. Will
use elastic net as the machine learning technique. It is a good tool
when multicollinearity can be of issue. Now that the predictors are in
place, one have to set up pricing signal, one sigma two-sided range is
the common practice in arbitrage, and for that reason, always go lower
or higher, you do not want to be caught up in common practice. We short
if it spikes above the upper threshold and long on the lower threshold.
The stop loss will be set at 2 standard deviation. At that point, one
can expect our interpretation of the underlying model to be wrong.

Warning: If you are in marketing or sales and you are used to the idea
that people brag about their large code base let me tell you that less
code is usually better.

**Sentiment Strategies:**

[A large web](https://github.com/je-suis-tm/web-scraping) scrape of
multiple news sources. Here is some more [AI and
NLP](https://github.com/andyli1688/AI-AND-NLP) exercise.

-   [NLP](https://github.com/toamitesh/NLPinFinance) - This project
    assembles a lot of NLP operations needed for finance domain.

-   [Earning call
    transcripts](https://github.com/lin882/WebAnalyticsProject) -
    Correlation between mutual fund investment decision and earning call
    transcripts.

-   [Buzzwords](https://github.com/swap9047/Cutting-Edge-Technologies-Effect-on-S-P500-Companies-Performance-and-Mutual-Funds) -
    Return performance and mutual fund selection.

-   [Fund
    classification](https://github.com/frechfrechfrech/Mutual-Fund-Market-Clusters/blob/master/Initial%20Data%20Exploration.ipynb) -
    Fund classification using text mining and NLP.

-   [NLP Event](https://github.com/yuriak/DLQuant) - Applying Deep
    Learning and NLP in Quantitative Trading.

-   [Financial Sentiment
    Analysis](https://github.com/EricHe98/Financial-Statements-Text-Analysis) -
    Sentiment, distance and proportion analysis for trading signals.

-   [Financial Statement
    Sentiment](https://github.com/MAydogdu/TextualAnalysis) - Extracting
    sentiment from financial statements using neural networks.

-   [Extensive
    NLP](https://github.com/TiesdeKok/Python_NLP_Tutorial/blob/master/NLP_Notebook.ipynb) -
    Comprehensive NLP techniques for accounting research.

-   [Accounting
    Anomalies](https://github.com/GitiHubi/deepAI/blob/master/GTC_2018_Lab-solutions.ipynb) -
    Using deep-learning frameworks to identify accounting anomalies.

-   First you can look at parsing [earnings call
    transcripts](https://github.com/erikcs/ConferenceCalls).

**Unsupervised Strategies**

-   This would be fast and easy to do.

**Here we go, what about a spot outliers -\> short strategy.
<https://github.com/huseinzol05/Stock-Prediction-Models/blob/master/misc/outliers.ipynb>**

[Some factor](https://github.com/sanjeevai/multi-factor-model) investing
using PCA.
[More](https://github.com/joelQF/quant-finance/blob/master/Machine-Learning-and-Reinforcement-Learning-in-Finance/Trading%20Strategy%20based%20on%20PCA.ipynb),

Also have a [PCA
trading](https://github.com/joelQF/quant-finance/blob/master/Machine-Learning-and-Reinforcement-Learning-in-Finance/Trading%20Strategy%20based%20on%20PCA.ipynb)
strategy.

Maybe create this, or include in others? Maybe not. Like this, finance
[graph
theory](https://github.com/evijit/Finance_Graph_Theory/blob/master/GrowthModels.ipynb)
investing. This big one by boris is good, it also includes some [feature
engineering](https://github.com/borisbanushev/stockpredictionai/blob/master/readme2.md#corrassets),
you have to actually use it. [VAE to embed
stocks](https://github.com/ml-hongkong/stock2vec). Here do PCA. To be
frank this graph theory does not work -- But I understand how one could
possibly play around with it.

**Risk Parity:**

Some [code on website](https://quantdare.com/risk-parity-in-python/) on
risk parity. Maybe a bit should be said about
[hierarchy](https://quantdare.com/hierarchical-clustering-of-etfs/)
parity. Some more risk parity
[stuff](https://github.com/dppalomar/riskParityPortfolio). I fear this
has way more to do with [portfolio optimisation,]{.underline} so I will
leave it there.

**Evolutionary Strategy**:

A free agent.

This code holds great prospect for an [evolutionary
strategy](https://github.com/huseinzol05/Stock-Prediction-Models/blob/master/free-agent/evolution-strategy-bayesian-agent.ipynb).

**Supervised Learning Strategy**:

[This](https://github.com/huseinzol05/Stock-Prediction-Models/blob/master/free-agent/evolution-strategy-bayesian-agent.ipynb)
is not just great, it's excellent, I can write a whole book based on it.
Okay time to go balls deep.

\- Stacking

\- Deep Learning

\- Evolutionary Strategy

**Agent Strategy**

**Real Time Agent**

From here, the easiest solution is to

Intuitively, this means that the conditional expectation of the future
value of the process, given all its historical values, equals to its
current value.

evolve a strategy as follows,

Markov means where go next depends at most on where we are now. Any
process with independent increments has the Markov property, e.g.,
Brownian motion. Martingale means that we expect the future value to be
the current value. Standard Brownian motion has the Markov property and
is a martingale. General Brownian motion with drift has the Markov
property and is NOT a martingale.

**HFT:**

This is fine it should be done.

[Tick data
trading](https://github.com/rorysroes/SGX-Full-OrderBook-Tick-Data-Trading-Strategy)
strategy.

**Risk and Hedging -- Once you get into risk this would be interesting,
also for now, I am not touching options.**

[Model free](https://github.com/jerryxyx/MonteCarlo) Monte Carlo
approach to American options hedging, this guy really has some good
research. Look at the
[omega](https://quantdare.com/omega-ratio-the-ultimate-risk-reward-ratio/)
as opposed to sharp ratio, and then also Marco's diluted sharp ratio.
Risk [solving](https://github.com/jcrichard/pyrb) algorithm. [Cost
sensitive](https://github.com/albahnsen/ML_RiskManagement)
classifications.

**Personal analysis on some derivatives**

[Derivatives](https://github.com/RobinsonGarcia/delta-hedging/blob/master/A1-Part2.ipynb)
is cool. Actually this is delta hedging so part of risk.

[Option
hedging](https://ipythonquant.wordpress.com/2018/06/05/option-hedging-with-long-short-term-memory-recurrent-neural-networks-part-i/)
with LTSTM.

**Feature Importance Methods**

Table 1: Earnings Related Variable Importance and Response Direction for
Classification

  **Name**                                                                                                                                                                                                                                                                                                                                                                                                               **Short Description**                                                                **Score**   **D**       
  ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- ------------------------------------------------------------------------------------ ----------- ----------- --
  $${est\_ avg}_{t}$$                                                                                                                                                                                                                                                                                                                                                                                                    This time period\'s analyst EPS forecast                                             0.247       \-          
  $$\text{dif}f_{- 1}$$                                                                                                                                                                                                                                                                                                                                                                                                  The difference between the past actual EPS, $p_{- 1}$ and $p_{- 2}$                  0.119       -/+         
  $$p_{- 1}$$                                                                                                                                                                                                                                                                                                                                                                                                            Actual EPS $t_{- 1}$                                                                 0.082       \-          
  $$d\_ e\_ diff_{- 4}$$                                                                                                                                                                                                                                                                                                                                                                                                 Difference between the past actual, $p_{- 4}$ and forecast ${est\_ avg}_{t - 4}\ $   0.073       \+          
  $$\text{dif}f_{- 4}$$                                                                                                                                                                                                                                                                                                                                                                                                  The difference between actual EPS $p_{- 4}$ and actual $p_{- 1}$                     0.060       -/+         
  Other                                                                                                                                                                                                                                                                                                                                                                                                                  57 other earnings-related variables.                                                             0.212       
  **Total**                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               **0.794**   
  **This table identifies the most important variables as identified by the Gini importance which measures the average gain in information. The variable importance measure (Score) is based on all variables. *D* identifies the direction of the \'coefficient\' as identified by the partial dependence process. See** **Error! Reference source not found., the graphs from which the directions are identified.**                                                                                                                

The most important earnings-related variable is the forecast itself,
${est\_ avg}_{t}$ this is expected because the purpose of the model is
to identify deviations from this amount and the actual, it, therefore,
provides a measure of reference to the other variables of the model.
Countless papers have identified the performance of analysts\' forecasts
in forecasting earnings as recapped by Brown (1987b). The lower the
forecasted EPS the more likely a surprise is to occur all else equal,
32% of the outcome is attributable to this value. A lower EPS could
proxy for a smaller firm or any other attributes. The second and fifth
most important variable, is the difference between the actual earnings
between $t_{- 1}\ $and $t_{- 2},\ $called $\text{dif}f_{- 1}$, and the
difference between $t_{- 1}\ $and $t_{- 4},\ $called $\text{dif}f_{- 4}$
These are novel variables not yet identified by past research. It
basically says that the past increases in earnings are an important
variable for predicting future surprises, which makes intuitive sense.
If the value is very high, surprises become more likely. Surprises are
also more likely if the value gets very low. For a lot of the
observations, the value is very small, decreasing the likelihood of
surprises. The measure is u-shaped, which is indicative of a sort of
variance measure. The next important value is the actuals earnings at
time $t_{- 1}$, called $p_{- 1}$. Research by Bradshaw et al. (2012),
have shown that the past annual earnings, often outperform, not just
mechanical time series models, but also analyst\' forecasts. Similarly,
past quarterly earnings also seem to be an important value in predicting
the next quarter\'s earnings surprise and is similarly shaped to the
analyst forecast, ${est\_ avg}_{t}$. The relationship shows that where
$p_{- 1}$ is large and ${est\_ avg}_{t}$ is simultaneously low, then a
positive surprise is likely to occur more than 90% of the time, all else
equal. Further, where $p_{- 1}$ is low and ${est\_ avg}_{t}$ is high
then a surprise is unlikely to occur. The next important variable is the
difference four quarters ago, i.e., one year between the forecast,
${est\_ avg}_{t - 4\ }$ and the actual value, $p_{- 4}$. The importance
of this variable was also expected as Easterwood & Nutt (1999) and Fried
& Givoly (1982) separately showed that past errors tend to persist. The
larger the difference, the higher the likelihood of surprise. Other
variables that showed an above 2% importance includes rolling averages
and weighted rolling averages of the difference between past earnings
and analyst forecasts, and the standard deviation of analyst forecasts.

![](media/image1.png){width="6.263888888888889in"
height="3.53125in"}Figure 2: Partial Dependence of Class Probabilities
on Earnings Related Feature Combinations for Classification

  ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
  **The figures indicate the percentage increase in the likelihood that an earnings surprise will occur, all else equal. The dashed lines identify the space where earnings surprises are less likely to occur. The small ticks on the axes is an indication of the underlying distribution. As the colours get warmer a surprise is more likely to occur, the colder the colour the less likely a surprise is to occur. These graphs show the partial dependence relationships between two variables. On the *left,* it can be seen that a surprise is more likely if both** $\text{dif}f_{- 1}$ **and** $\text{dif}f_{- 4}$ **are large and that surprises are less likely when both these values are around the mean. This would indicate that there is a predictable trend, hence a lower likelihood of surprise. Another interesting observation is that if the longer trend** $\text{dif}f_{- 4}$ **is large, and shorter-term earnings decreases,** $\text{dif}f_{- 1}$ **is negative, then the small blip is short-lived and likely to be corrected in the next period as can be seen with the high likelihood of surprise in this area, \>79%, i.e., the top left corner. On the *right,* as previously noted, when** $p_{- 1}$ **is large and** ${est\_ avg}_{t}$ **is low, a surprise is likely, \>90%, and where** $p_{- 1}$ **is low and** ${est\_ avg}_{t}$ **high a surprise is unlikely.**
  ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

It is not always the best practice to look at the isolated effect of an
input variable on the response variable, for that reason I have included
the above dependence plots between two input variables and the output.
Out of the above list, there are more than 32 $(2^{5}$) ways to conjure
up directional relationships. Due to the nature of nonlinear
relationships, to conceptually understand the web of relationship, it is
better to describe a simple tree explanation of the above variables and
relationships for a combination of variables that would lead to a
perfect surprise prediction. The following gives the explanation of what
would lead to a close to perfect surprise prediction. If the current
analysts forecast is low while the past earnings are high, there is a
higher likelihood of a positive surprise this period; this is likely due
to analysts being conservative. Considering the previous relationship,
if in the past it has been shown that analysts are conservative and that
surprises transpired, i.e., $d\_ e\_ diff_{- 4}$, then the likelihood of
surprises increase even more. Furthermore, considering the above
relationships, if it is further noted that there is a large difference
in earnings between the last two periods EPS, $p_{- 1}$ and $p_{- 2}$,
i.e., $\text{dif}f_{- 1}$, then the likelihood of surprise increases
even more. The same holds for $\text{dif}f_{- 4}$. Overall these
variables accounted for around 80% of the variable importance. Referring
back to *Table 5*, it has also been shown, using an inductive method to
theory testing that these variables accounted for 70% (10.4 p.p.) of the
total improvement over the benchmark.

Different ways to explain feature importance.

Table 2: Category Importance Analysis

                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            (1)   (2)           (3)             (4)   (5)   (6)   (7)     (8)      (9)   (10)   (11)
  ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- ----- ------------- --------------- ----- ----- ----- ------- -------- ----- ------ ------
  Category                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  CI    RCLI Top 50   Post GFC RCLI   Pot   F     wF    24 CI   PCA 10   RIT   Avg.   Fin.
  Assets & Liabilities                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      1     1             1               1     1     1     1       1        1     1.0    1
  *Solvency*                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                2     3             2               2     2     2     3       2        2     2.2    2
  Income                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    3     2             3               3     4     5     2       4        3     3.2    3
  *Valuation & Profitability*                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               4     4             4               4     3     4     5       3        4     3.9    4
  Equity                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    5     5             6               5     5     3     4       6        5     4.9    5
  Expense                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   6     6             5               6     7     6     6       10       7     6.6    6
  *Efficiency*                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              7     7             7               8     6     7     7       9        6     7.1    7
  *Other*                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   8     8             8               9     8     8     10      8        8     8.3    8
  *Liquidity*                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               9     9             9               10    8     9     9       5        10    8.7    9
  Cash Flow                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 10    10            10              7     10    10    8       7        9     9.0    10
  This table is an attempt to regroup categories where there is a strong correlation 80% + and to calculate the rank of the categories according to 9 different predictive importance strategies. This table calculates the equal weighted average of nine ranking methods (10). (1) Is the normal importance measure (gain measure) calculated for all variables in every category. (2) Is the gain measure for only the top 50 variables. (3) Is the gain measure after the GFC. (4) Is the ranking according to the potency measure, being the category importance weighted by the number of variables in each category. (5) Is a measure (FScore) that tallies the amount of variable splits across all trees within each category. (6) Measures the FScore weighted by the probability of the splits to take place. (7) Is the overall gain measure for a newly created model that only incorporates the 24 best variables. (8) Is the importance of the first PCA component for each category. (9) Avg. is the equal-weighted rank average for each category. (10) Is the final ranking obtained by ranking the raw average. When percentage growth measures were removed from the categories, all categories remained unchanged apart from a swap between *Other* and *Liquidity*. A further split in category where solvency ratios were split between capital structure, and coverage and cash flow ratioratios resulted in the following rank among categories, (1) asset and liabilities (2) Income (3) *valuation and profitability*, (4) *capital structure,* (5) equity, (6) *interest coverage,* (7) expense, (8) *efficiency*, (9) *cash flow* ratios, (10) other ratios (11) *liquidity ratios*, (12) cash flow values. The ratio values are italicised.                                                                                       
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                

Figure 3: Interaction Pair Partial Dependence Plots (Depth Two)

![](media/image2.png){width="6.191666666666666in" height="7.0625in"}

  ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
  **(1) Top left is the interaction between Interest and Related Expense (xint) and the EPS Excluding Extra. Items (epspx) and resulting response. (2) Top right is the interaction between Price to Sales (ps) and Long-Term Debt (dltt). (3) Bottom left is the interaction between Total Debt to Invested Capital (totdebt\_invcap) and Income Before Extraordinary Items (ibc) and the interaction effect on the bankruptcy outcome. (4) Bottom right is the interaction between Total Liabilities (lt) and the EPS Excluding Extra. Items (epspx).**
  ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Table 3: Cross Tab - Top Variable Interactions

                 totdebt\_invcap   ps    lt    xint   pi
  -------------- ----------------- ----- ----- ------ -----
  ibc            779               704   63    45     13
  pi             66                585   209   338    0
  epspx          228               76    551   509    34
  dltt           17                418   156   34     14
  debt\_assets   43                127   239   77     390
  ppent          71                279   82    28     61

  ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
  This table represents the most important interaction pairs as measured by the gain statistic at an interaction depth of two. The table has been constructed to highlight the top ten interactions. For completeness, the surrounding interactions have also been included. Variables vertically follows, Income Before Extraordinary Items (ibc), Pre-tax Income (pi), EPS Excluding Extra. Items (epspx), Long Term Debt (dltt), Total Debt to Total Assets (debt\_assets), Property Plant & Equipment (ppent). And horizontally, Total Debt to Invested Capital (totdebt\_invcap), Price to Sales (ps), Total Liabilities (lt), Interest and Related Expense (xint).
  ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Table 4: Depth 3 - Interaction Analysis.

                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       Term 1     Sign   Term 2           Sign   Term 3            Sign   RII   Gain
  -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- ---------- ------ ---------------- ------ ----------------- ------ ----- ------
  (1)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  epsfx      \-     ibc              \-     totdebt\_invcap   \+     100   456
  (2)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  debt\_at   \+     pi               \-     rd\_sale          \-     95    435
  (3)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  dpc        \-     equity\_invcap   \+     ps                \-     92    419
  (4)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  ibc        \-     ps               \+     totdebt\_invcap   \+     88    402
  (5)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  dltt       \+     ibc              \-     ps                \-     84    383
  (6)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  dltt       \+     pi               \-     ps                \-     84    382
  (7)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  ibc        \-     ps               \-     ps\_prtc          \-     83    378
  (8)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  ibc        \-     ibc              \-     totdebt\_invcap   \+     79    362
  (9)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  dltt       \+     ps               \-     txt               \+     76    348
  (10)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 ibc        \-     ppent            \+     ps                \-     74    336
  (11)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 at         \+     debt\_at         \+     epspx             \-     68    310
  Out of the top 50 variable list, there are millions of ways to conjure up directional relationships. Due to the nature of nonlinear relationships, to conceptually understand the web of relationship, it is best to identify the top interaction pairs. This table represents the most important interaction pairs as measured by the gain statistic at an interaction depth of three. For easier reading, I also report the relative interaction importance (RII). The sign purely indicates the average direction of each variable. The interaction terms are much more informative than single standing variables. Interactions is at the core of what gradient boosting tree models are all about. Unique *Terms 1* are EPS (Diluted) - Excl. Extra. Items (epsfx), Assets - Total (at). Unique *Terms 2* are Common Equity/Invested Capital (equity\_invcap). Unique *Terms 3* are Research and Development/Sales (rd\_sale) and Income Taxes - Total (txt).                                                                            

Figure 4: Bubble Plot and Ranking of each Model\'s Most Important
Categories.

![](media/image10.png){width="6.177083333333333in"
height="4.430555555555555in"}

  -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
  This figure reports the relative importance of the five outcome classification models and the associated accounting dimensions. There is a large amount of heterogeneity between the different classification models.
  -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Table 5: Reverse Induction Test

  ROC (AUC)               (1)     (2)     (3)        (4)      (5)      (6)
  ----------------------- ------- ------- ---------- -------- -------- -------
                          Full    A&L     Solvency   Income   Equity   P&V
  A - All                 0.959   0.942   0.944      0.948    0.955    0.952
  B - CV                  0.947   0.930   0.935      0.937    0.943    0.941
  C - Time CV             0.957   0.940   0.942      0.946    0.954    0.950
  D - Average             0.954   0.941   0.944      0.947    0.955    0.951
  Relative Contribution   \-      100     82         63       22       39

This table compares the various sets of variables against each other
using an inductive testing technique to identify the importance of
groups of variables that explain the model success. (1) Is the
performance of a model that contains all the variables. (2) - (6)
removes variables that fall within the respective asset & liability,
solvency, income, liability and profitability, and valuation categories
from the model after which the model is retrained and tested to identify
the extent to which each category contributes to the model. Liang et al.
(2016) similarly used categories to test the associated AUCs. The
following relative contributions are unreported in the table, expense
(10), efficiency (8), liquidity (8), Other (8), Cash flow (4). The
relative contribution is calculated by using the average of three
different performance techniques to ensure the robustness of the
results. The first three rows are three different methods to measure the
performance of the newly constructed model, (A) the first method is an
80% train and 20% split result in time-series. (B) is a random 10-fold
performance validation split. (C) is a variant of the 10-fold
performance split, but in time series; it is arguably the most robust
method. (D) is the average across all three methods, and the value used
to calculate the Relative Contribution. All these splits are done after
the cross-validation and model development steps.

**Cross Validation**

Hopefully it's clear that
[backtesting](https://github.com/ThomasGrivaz/ai-for-trading/blob/master/backtesting.ipynb)
will form part of the cross-validation process. Here is an actual
walk-forward
[method](https://github.com/convergenceIM/invest-ML/blob/master/04_Walk_Forward_Modeling.ipynb).
Here for sports is some [ensemble
modelling](https://github.com/convergenceIM/alpha-scientist/blob/master/content-draft/06_Ensemble_Modeling.ipynb).

Table 6: Model Comparison Using Different Performance Validation
Procedures

+-----------+-----------+-----------+-----------+-----------+-----------+
|  Metrics  | (1)       | (2)       | (3)       | (4)       | 95%       |
|           |           |           |           |           | Confidenc |
|           | All Data  | Time-Spli | K-Fold    | Time      | e         |
|           |           | t         | (KF)      | Split     | (+/-)     |
|           |           |           |           | Fold      |           |
|           |           | (TS)      |           | (TSF)     |           |
+===========+===========+===========+===========+===========+===========+
| ROC AUC   | 0.9587    | 0.9655\*\ | 0.9467\*\ | 0.9570    | 0.0142    |
| Sore      |           | *         | *         |           |           |
+-----------+-----------+-----------+-----------+-----------+-----------+
| Accuracy  | 0.9755    | 0.9837    | 0.9682    | 0.9712    | 0.0163    |
| Score     |           |           |           |           |           |
+-----------+-----------+-----------+-----------+-----------+-----------+
| False     | 0.0037    | 0.0069    | 0.0028    | 0.0039    | 0.0015    |
| Positive  |           |           |           |           |           |
| Rate      |           |           |           |           |           |
| (p-value) |           |           |           |           |           |
+-----------+-----------+-----------+-----------+-----------+-----------+
| Average   | 0.1414    | 0.0825    | 0.1301    | 0.1052    | 0.0707    |
| Log       |           |           |           |           |           |
| Likelihoo |           |           |           |           |           |
| d         |           |           |           |           |           |
+-----------+-----------+-----------+-----------+-----------+-----------+

This table compares the performance of the best models that resulted
from different out-of-sample performance tests. (1) The original \"All
Data\" model allocates 60% of the observation to the training set, 15%
to the development of validation test set and 25% to the test set. The
15% is used to measure and improve the performance of the model. The
observations to each of the splits is randomly selected. (2) TS is a
simple ordering of the observation in time series and the creation of
longitudinal training - 60%, validation - 15% and test set splits -25%,
this method ensures that there is no information leakage from the future
observations. (3) KF is a randomised cross-sectional method that
scrambles the observations and splits them into training and test sets
and calculates the average metrics from 10 different iterations or
folds. (4) TSF is the most robust method and has also led to the model
with the best generalisable performance as evidenced by the battery of
metrics - It is a longitudinal blocked form of performance-validation
that suits this form of bankruptcy prediction, it uses the strengths of
both (2) and (3). All statistical comparisons are made against the model
called \"All Data.\" **\*p\<.1 \*\* p\<.05 \*\*\* p\<.01. Significance
levels are based on a two-tailed Z test.**

Table 7: Model Comparison Using Different Inputs

   Metrics                 All Data   50 Variables Model   One Year Before Bankruptcy   Two Years Before Bankruptcy
  ------------------------ ---------- -------------------- ---------------------------- -----------------------------
  ROC AUC Sore             0.9587     0.9408\*\*\*         0.9666\*\*                   0.9434\*\*\*
  Accuracy Score           0.9755     0.9700               0.9860                       0.9837
  False Positive Rate      0.0037     0.0056               0.0010                       0.0002
  Average Log Likelihood   0.1414     0.1795               0.1682                       0.2206

This table compares the performance of model that includes only 50 of
the most predictive variables as inputs, a model that only includes
bankruptcy observations one or two years before the filing. All
statistical comparisons are made against the model called \"All Data.\"
**\*p\<.1 \*\* p\<.05 \*\*\* p\<.01. Significance levels are based on a
two-tailed Z test.**

**Model Selection**

Table 8: XGBoost and Deep Learning Model Performance Comparison

   Metrics                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            XGBoost Model   Deep Feed Forward Network   Deep Convolutional Neural Network   Logit Model
  --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- --------------- --------------------------- ----------------------------------- --------------
  ROC AUC Sore                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        0.9587          0.8444\*\*\*                0.9142\*\*\*                        0.7092\*\*\*
  Accuracy Score                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      0.9755          0.9324                      0.9518                              0.6856
  False Positive Rate                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 0.0037          0.0666                      0.0296                              0.2296
  Average Log Likelihood                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              0.1414          0.5809                      0.2996                              1.1813
  **This table illustrates the performance of two deep learning models against the XGBoost Model. The Feed Forward Network is a deep learning network that does not circle back to previous layers. The Convolutional Neural Network is a biologically inspired variant of MLP, popularised by recent image classification studies. The best possible Logit model was established by choosing a selection of the best variables. Further results include the isolation of the 10 best predictor variables (using the Gini Index) in all models, this produced similar results to the above table both in extent and in rank. \*p\<.1 \*\* p\<.05 \*\*\* p\<.01. Significance levels are based on a two-tailed Z test to identify the statistically significant difference between all contender models and the best performing model, which is made possible due to the cross-validation process.**                                                                                   

Table 9: XGBoost and Decision Tree Ensemble Model Performance Comparison

   Metrics                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          XGBoost Model   AdaBoost Model   Random Forest Model   Stacked Model   
  ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- --------------- ---------------- --------------------- --------------- --
  ROC AUC Sore                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      0.9587          0.9291\*\*\*     0.9275\*\*\*          0.9495\*\*      
  Accuracy Score                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    0.9755          0.9612           0.9576                0.9681          
  False Positive Rate                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               0.0037          0.0185           0.0370                0.0074          
  Average Log Likelihood                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            0.1414          0.1913           0.2409                0.1613          
  **This figure illustrates the performance of the XGBoost model with two other tree ensemble models, AdaBoost and a Random Forest Model. AdaBoost and XGBoost are both ensembles that seek to convert weak learners into a single strong learner. AdaBoost adds weak learners according to performance by changing the sample distribution. In XGBoost the weak learner\'s trains on the residuals to become a \'strong\' learner. Random forests are simply a multitude of decision trees. All three underlying models have decision trees as the base learner. The stacked model is a combination of the AdaBoost, Convolutional Neural Network, Feed Forward Network and Random Forest models into one big model by using the four models\' predicted outcomes as inputs to a decision tree model. \*p\<.1 \*\* p\<.05 \*\*\* p\<.01. Significance levels are based on a two tailed Z test.**                                                                          

**Results and Metrics**

<https://github.com/convergenceIM/alpha-scientist/tree/master/content>
Some great things to say about the different metrics we can use to look
at the results on the test set.

**Classification**

The prediction values, $\text{PredEPS}_{\text{ith}},$ of the Regressor
is a continuous variable {1.34, 1.56\...}. Analogous to the classifier,
if we assume that the training set comprise 60% of the original data
set, then the training set\'s target values are
$\text{EPSAC}_{it\ (60\%)}$, being the first 60% of the dataset ordered
by date. As a consequence, the test set\'s target value is
$\text{EPSAC}_{it\ (60\%)}$, the last 40% of the dataset. The metrics
comprise of Residual Mean Squared (RMSE) and Mean Absolute (MAE) Errors.
The algorithms for classification and regression primarily differ in the
loss function used; the process is otherwise the similar.

The evaluation of a successful model can start wit an accuracy measure.
The accuracy can be defined as the percentage of correctly classified
instances (observations) by the model. It is the number of correctly
predicted surprises (true positives) and correctly predicted
non-surprises (true negatives) in proportion to all predicted values. It
incorporates all the classes into its measure
$(TP + TN)/(TP + TN + FP + FN)$, where *TP, FN, FP* and *TN* is the
respective true positives, false negatives, false positives and true
negatives values for all classes. The measure can otherwise be
represented as follows:
$\text{\ acc}\left( y,\widehat{y} \right) = \ \frac{1}{n_{\text{samples}}}\sum_{i = 0}^{n_{\text{samples}} - 1}{1(}{\widehat{y}}_{i} = y_{i})$.

Table 10: Surprise Breakdown Random Guessing Confusion Matrix

  Random Confusion Matrix                                                                                                                       **Random Guessing**   Marginal Sum of Actual Values                          
  --------------------------------------------------------------------------------------------------------------------------------------------- --------------------- ------------------------------- ---------- ----------- --------
                                                                                                                                                Neutral               Negative                        Positive               
  **Actual**                                                                                                                                    Neutral               **89020**                       24590      55890       169500
                                                                                                                                                Negative              24590                           **6792**   15439       46821
                                                                                                                                                Positive              55890                           15439      **35090**   106419
  Marginal Sum of Predictions                                                                                                                   169500                46821                           106419     322740      
  **This table is formed by \'randomly choosing the observations\' by allocating the observations according to the underlying distribution.**                                                                                

The multiclass ROC is a universal way to identify the performance of a
classification model. The AUC (area under curve) score provides an
integral based performance measure of the quality of the classifier. It
is arguably the best single number machine learning researchers have in
measuring the performance of a classifier. The middle line is a line of
random ordering. Therefore, the tighter the ROC-curves fit to the left
corner of the plot, the better the performance. Two other measures
included in the graph is a macro-average measure that equal weight to
each class (category) and a micro-average measure that looks at each
observation. AUC values of 0.70 + is generally expected to show strong
effects. The ROC test is the first indication that the model performance
is significantly different from null. In subsequent tables*,* **Error!
Reference source not found.** and **Error! Reference source not
found.***,* I will also test the statistical significance of this
outperformance.

Figure 5: Multiclass Receiver Operating Characteristic (ROC) for a 15%+
Surprises Strategy

![](media/image11.png){width="5.325694444444444in"
height="3.890972222222222in"}

  ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
  **This figure reports the ROC and the associated area under the curve (AUC). The ROC is measured for three different classes, class 0 is the negative surprise class, 1 is the neutral class and 2 is the positive surprise class. The macro-average measure is equal weighted to each class and a micro-average measure looks at each observation weight. The random ordering or luck line is plotted diagonally through the chart. The associated curves are associated with a good classification model.**
  ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Table 11: Healthy and Bankrupt Confusion Matrix.

  Aggregated Health and Bankrupt Firms Matrix                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        **Predicted**   Sample Proportion                  
  ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ --------------- ------------------- -------------- ------
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     Healthy         Bankrupt                           
  **Actual**                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         Healthy         **29041 - TN**      116 - FP       0.96
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     Bankrupt        805 - FN            **258 - TP**   0.03
  Precision                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          0.97            0.69                30220          
  Improvement                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        0.01            0.66                \-             
  **This bankruptcy prediction task solves for a binary classification problem that produces a 2×2 matrix. The columns of the matrix represent the predicted values, and the rows represent the actual values for bankrupt and healthy firm predictions. In the cross-section of the rows and columns, we have the True Positive (TP), False Negative (FN - type II error), False Positive (FP - type I error)), and True Negative (TN) values. The sample proportion on the far right is equal to all the actual observations of a certain classification divided by all the observations. The *precision* is calculated by dividing the true positives (Bankruptcies) with the sum of itself and the false negatives (Healthy). An example along the second column:** $258/(116 + 258)\  = \ 69\%$. The improvement is the percentage point improvement the prediction model has over a random choice benchmark.                                                      

Table 12: Random Guessing Confusion Matrix.

  Aggregated Health and Bankrupt Firms Matrix                                                                                                                                                        **Random Guess**   Marginal Sum of Actual Values                 
  -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- ------------------ ------------------------------- ------------- -------
                                                                                                                                                                                                     Healthy            Bankrupt                                      
  **Actual**                                                                                                                                                                                         Healthy            **28131 - TN**                  1026 - FP     29157
                                                                                                                                                                                                     Bankrupt           1026 - FN                       **37 - TP**   1063
  Marginal Sum of Guesses                                                                                                                                                                            29157              1063                            1063          
  **This table is formed by \'randomly choosing the observations\' by allocating the observations according to the underlying distribution, as presented by Sample Proportion in** *Table 11**.***                                                                    

[Filing Outcomes Prediction]{.underline}

Filing Outcomes 
================

The prediction of all filing outcomes is contingent on a correctly
predicted bankruptcy outcome. All these models use simple accounting
value inputs the year before the filing date. The summary statistics of
filing outcomes can be found in **Error! Reference source not found.**.
As mentioned in the **Error! Reference source not found.**, filing
outcomes have great economic consequence for creditors and shareholders.
Stakeholders would want to know the likelihood of a litigated bankruptcy
occurring as well as the likely filing outcomes associated with the
bankruptcy. In *Table 13*, I present the performance of five different
filing outcome models. The first of these five is the chapter prediction
model. It involves a prediction task of whether the bankruptcy will
finally be filed under chapter 7 or chapter 11. The chapter prediction
model performed the best of all other filing outcomes models. It
achieved an AUC of 0.88. The survival prediction model that identifies
whether the firm would emerge from bankruptcy performed second best with
an AUC of 0.73. The prediction task that attempts to predict whether
assets will be sold in a 363 Asset sale or by other means, came in third
with an AUC of 0.64. The duration task, which involves the prediction of
whether the bankruptcy proceedings would endure for longer than one year
came in second to last with an AUC of 0.62. And lastly, the tort task
had an AUC score of 0.54 which is only slightly higher than random. All
prediction tasks performed better than random guessing.

Table 13: Binary Classification Performance for Predicting Bankruptcy
Characteristics.

+---------+---------+---------+---------+---------+---------+---------+
| Binary  | ROC AUC | Accurac | Average | Average | False   | False   |
|         | Sore    | y       | Log     | Precisi | Positiv | Negativ |
| Classif |         | Score   | Likelih | on      | e       | e       |
| ication |         |         | ood     | Score   | Rate    | Rate    |
|         |         |         |         |         |         |         |
| Model   |         |         |         |         |         |         |
+=========+=========+=========+=========+=========+=========+=========+
| Duratio | 0.62    | 0.56    | 0.67    | 0.69    | 0.66    | 0.26    |
| n       |         |         |         |         |         |         |
+---------+---------+---------+---------+---------+---------+---------+
| Surviva | 0.73    | 0.69    | 0.59    | 0.80    | 0.61    | 0.12    |
| l       |         |         |         |         |         |         |
+---------+---------+---------+---------+---------+---------+---------+
| Chapter | 0.88    | 0.95    | 0.50    | 0.70    | 0.05    | 0.20    |
+---------+---------+---------+---------+---------+---------+---------+
| Asset   | 0.64    | 0.66    | 0.61    | 0.39    | 0.27    | 0.55    |
| Sale    |         |         |         |         |         |         |
+---------+---------+---------+---------+---------+---------+---------+
| Tort    | 0.54    | 0.90    | 0.40    | 0.17    | 0.05    | 0.83    |
+---------+---------+---------+---------+---------+---------+---------+
| This    |         |         |         |         |         |         |
| table   |         |         |         |         |         |         |
| reports |         |         |         |         |         |         |
| six     |         |         |         |         |         |         |
| importa |         |         |         |         |         |         |
| nt      |         |         |         |         |         |         |
| metrics |         |         |         |         |         |         |
| for     |         |         |         |         |         |         |
| five    |         |         |         |         |         |         |
| alterna |         |         |         |         |         |         |
| tive    |         |         |         |         |         |         |
| classif |         |         |         |         |         |         |
| ication |         |         |         |         |         |         |
| tests   |         |         |         |         |         |         |
| to      |         |         |         |         |         |         |
| predict |         |         |         |         |         |         |
| the     |         |         |         |         |         |         |
| outcome |         |         |         |         |         |         |
| of      |         |         |         |         |         |         |
| predict |         |         |         |         |         |         |
| ed      |         |         |         |         |         |         |
| bankrup |         |         |         |         |         |         |
| tcies.  |         |         |         |         |         |         |
| *Durati |         |         |         |         |         |         |
| on*     |         |         |         |         |         |         |
| classif |         |         |         |         |         |         |
| ication |         |         |         |         |         |         |
| is the  |         |         |         |         |         |         |
| first   |         |         |         |         |         |         |
| task to |         |         |         |         |         |         |
| predict |         |         |         |         |         |         |
| the     |         |         |         |         |         |         |
| binary  |         |         |         |         |         |         |
| outcome |         |         |         |         |         |         |
| .       |         |         |         |         |         |         |
| This    |         |         |         |         |         |         |
| task    |         |         |         |         |         |         |
| involve |         |         |         |         |         |         |
| s       |         |         |         |         |         |         |
| the     |         |         |         |         |         |         |
| predict |         |         |         |         |         |         |
| ion     |         |         |         |         |         |         |
| of      |         |         |         |         |         |         |
| whether |         |         |         |         |         |         |
| or not  |         |         |         |         |         |         |
| the     |         |         |         |         |         |         |
| disposi |         |         |         |         |         |         |
| tion    |         |         |         |         |         |         |
| will    |         |         |         |         |         |         |
| take    |         |         |         |         |         |         |
| longer  |         |         |         |         |         |         |
| than a  |         |         |         |         |         |         |
| year    |         |         |         |         |         |         |
| after   |         |         |         |         |         |         |
| the     |         |         |         |         |         |         |
| initial |         |         |         |         |         |         |
| filing. |         |         |         |         |         |         |
| *Surviv |         |         |         |         |         |         |
| al*     |         |         |         |         |         |         |
| predict |         |         |         |         |         |         |
| s       |         |         |         |         |         |         |
| a       |         |         |         |         |         |         |
| binary  |         |         |         |         |         |         |
| outcome |         |         |         |         |         |         |
| as to   |         |         |         |         |         |         |
| whether |         |         |         |         |         |         |
| or a    |         |         |         |         |         |         |
| firm    |         |         |         |         |         |         |
| will    |         |         |         |         |         |         |
| re-emer |         |         |         |         |         |         |
| ge      |         |         |         |         |         |         |
| out of  |         |         |         |         |         |         |
| bankrup |         |         |         |         |         |         |
| tcy     |         |         |         |         |         |         |
| and     |         |         |         |         |         |         |
| remain  |         |         |         |         |         |         |
| in      |         |         |         |         |         |         |
| busines |         |         |         |         |         |         |
| s       |         |         |         |         |         |         |
| for     |         |         |         |         |         |         |
| longer  |         |         |         |         |         |         |
| than 5  |         |         |         |         |         |         |
| years.  |         |         |         |         |         |         |
| The     |         |         |         |         |         |         |
| *Chapte |         |         |         |         |         |         |
| r*      |         |         |         |         |         |         |
| task    |         |         |         |         |         |         |
| predict |         |         |         |         |         |         |
| s       |         |         |         |         |         |         |
| whether |         |         |         |         |         |         |
| the     |         |         |         |         |         |         |
| bankrup |         |         |         |         |         |         |
| tcy     |         |         |         |         |         |         |
| filing  |         |         |         |         |         |         |
| would   |         |         |         |         |         |         |
| be      |         |         |         |         |         |         |
| convert |         |         |         |         |         |         |
| ed      |         |         |         |         |         |         |
| to      |         |         |         |         |         |         |
| Chapter |         |         |         |         |         |         |
| 7 or    |         |         |         |         |         |         |
| whether |         |         |         |         |         |         |
| it will |         |         |         |         |         |         |
| be a    |         |         |         |         |         |         |
| Chapter |         |         |         |         |         |         |
| 11      |         |         |         |         |         |         |
| filing. |         |         |         |         |         |         |
| The     |         |         |         |         |         |         |
| *Asset  |         |         |         |         |         |         |
| Sale*   |         |         |         |         |         |         |
| model   |         |         |         |         |         |         |
| predict |         |         |         |         |         |         |
| s       |         |         |         |         |         |         |
| whether |         |         |         |         |         |         |
| the     |         |         |         |         |         |         |
| debtor  |         |         |         |         |         |         |
| will    |         |         |         |         |         |         |
| sell    |         |         |         |         |         |         |
| all or  |         |         |         |         |         |         |
| substan |         |         |         |         |         |         |
| tially  |         |         |         |         |         |         |
| all the |         |         |         |         |         |         |
| assets  |         |         |         |         |         |         |
| during  |         |         |         |         |         |         |
| the     |         |         |         |         |         |         |
| Chapter |         |         |         |         |         |         |
| 11      |         |         |         |         |         |         |
| proceed |         |         |         |         |         |         |
| ings.   |         |         |         |         |         |         |
| The     |         |         |         |         |         |         |
| *Tort*  |         |         |         |         |         |         |
| classif |         |         |         |         |         |         |
| ication |         |         |         |         |         |         |
| task    |         |         |         |         |         |         |
| seeks   |         |         |         |         |         |         |
| to      |         |         |         |         |         |         |
| predict |         |         |         |         |         |         |
| whether |         |         |         |         |         |         |
| the     |         |         |         |         |         |         |
| bankrup |         |         |         |         |         |         |
| tcy     |         |         |         |         |         |         |
| would   |         |         |         |         |         |         |
| occur   |         |         |         |         |         |         |
| as a    |         |         |         |         |         |         |
| result  |         |         |         |         |         |         |
| of      |         |         |         |         |         |         |
| tortiou |         |         |         |         |         |         |
| s       |         |         |         |         |         |         |
| actions |         |         |         |         |         |         |
| such as |         |         |         |         |         |         |
| product |         |         |         |         |         |         |
| liabili |         |         |         |         |         |         |
| ty,     |         |         |         |         |         |         |
| fraud,  |         |         |         |         |         |         |
| pension |         |         |         |         |         |         |
| ,       |         |         |         |         |         |         |
| environ |         |         |         |         |         |         |
| mental  |         |         |         |         |         |         |
| and     |         |         |         |         |         |         |
| patent  |         |         |         |         |         |         |
| infring |         |         |         |         |         |         |
| ement   |         |         |         |         |         |         |
| claims. |         |         |         |         |         |         |
| The     |         |         |         |         |         |         |
| above   |         |         |         |         |         |         |
| metrics |         |         |         |         |         |         |
| have    |         |         |         |         |         |         |
| been    |         |         |         |         |         |         |
| fully   |         |         |         |         |         |         |
| defined |         |         |         |         |         |         |
| in      |         |         |         |         |         |         |
| table   |         |         |         |         |         |         |
| X.      |         |         |         |         |         |         |
+---------+---------+---------+---------+---------+---------+---------+

**Regression**

You need to have benchmarks to know if your model outperforms, in
general you might compete against other machine learning model, or
simply with more traditional mechanistic models. The first time-series
forecast model, looks at the average value of the last 4 quarters, and
is shown to produces surprisingly good results.

$${EpsEst4Q}_{\text{it}} = \frac{1}{4}(\sum_{p = 1}^{4}{\text{EPSAC}_{i(t - p)\left( x \right)})}$$

The next model, assumes that
$\text{EPSAC}_{\text{it}\left( x \right)} = \ \text{EPSAC}_{i(t - 4)\left( x \right)} + \varepsilon_{t}\ ,\ $where
the stochastic error term, $\varepsilon_{t}$, is a sequence of
identically distributed uncorrelated random variables with common mean 0
and variance $\sigma_{0}^{2}$. Therefore, in this equation, we simply
drop the error term and use the past earnings 4 quarters ago, it is
essentially a simplified seasonal autoregressive model.

  -- ----------------------------------------------------------------------------- -----
     $$\text{EpsEstRW}_{\text{it}} = \ \text{EPSAC}_{i(t - 4)\left( x \right)}$$   (2)
  -- ----------------------------------------------------------------------------- -----

The next model takes the average EPS over the last 4 years in a seasonal
manor by only incorporating the same quarter every year.

  -- --------------------------------------------------------------- -----
     $${EpsEst4Y}_{\text{it}} = \frac{1}{4}(\sum_{\begin{matrix}     (3)
     p = 4k \\                                                       
     k\epsilon\{ 1\ldots 4\} \\                                      
     \end{matrix}}^{4}{\text{EPSAC}_{i(t - p)\left( x \right)})}$$   
  -- --------------------------------------------------------------- -----

I have also incorporated a model called Last Quarter, that measures the
value of the quarter just before the current earnings period. It also
produces surprisingly good results.

  -- ----------------------------------------------------------------------------- -----
     $$\text{EpsEstLQ}_{\text{it}} = \ \text{EPSAC}_{i(t - 1)\left( x \right)}$$   (4)
  -- ----------------------------------------------------------------------------- -----

The next model is a more involved ARIMA models as conceptualized by Box
and Jenkins, that combines autoregression with moving averages terms
that are linear to the time series $\text{EP}S_{t},$ with white noise
terms $\varepsilon_{t}$. Here, we use the Griffen-Watts model in the Box
and Jenkins notation, (0,1,1) (0,1,1). Our ARIMA model requires 6 years
of complete data and can be represented in the following form:

  -- ------------------------------------------------------------------------------------------------------------------------------------------------ -----
     $$\left( 1 - B \right)\left( 1 - B^{4} \right)\text{EP}S_{t} = \left( 1 - \theta B \right)\left( 1 - \theta_{4}B^{4} \right)\varepsilon_{t},$$   (5)
  -- ------------------------------------------------------------------------------------------------------------------------------------------------ -----

where, *B* is the back-shift operator (Such that
$B \times EPS_{t} = EPS_{t - 1\ }\text{\ and\ }B^{4} \times \text{EPS}_{t} = \text{EPS}_{t - 4}$)
$\varepsilon_{t}$, the white noise and $\theta$, the regular
moving-average parameter and $\theta_{4}$, the seasonal moving average
parameter, forecasted from historical time series data. Forecasts were
made for each quarter of the following year. Note that if
$\theta_{4} = 1$, the equation reduces
to$\left( 1 - B \right)S_{t} = \left( 1 - \theta B \right)\varepsilon_{t},$
which represents a non-seasonal, integrated moving average process. The
constituents of all the predictions models can be found in the following
set:

  ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ -----
  $$\text{EpsEst}_{\text{it}}\ \{\text{PredEPS}_{\text{ith}},\ EpsEst4Q,\ {EpsEst4Y}_{\text{it}},\ \text{EpsEstRW}_{\text{it}},\ \text{EpsEstLQ}_{\text{it}},\ ARIMA\}$$   (6)
  ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ -----

The individual differences between the forecasted and actual EPS are
called residuals, the Residual Mean Squared Error (RMSE) and Mean
Absolute Error (MAE) aggregates all the residuals into a single measure
of predictive power. These are the main error measures that we would be
using to measure the performance of the various prediction models.

     $$RMSE = \ \sqrt{\frac{\sum_{i = 1}^{n}{(\text{EPSAC}_{\text{it}\left( 1 - x \right)} - \ \text{EpsEst}_{\text{it}})}^{2}}{n}}$$   (7)   
  -- ---------------------------------------------------------------------------------------------------------------------------------- ----- --
     $$MAE = \ \frac{1}{n}\sum_{i = 1}^{n}\left| \text{EPSAC}_{\text{it}\left( 1 - x \right)} - \ \text{EpsEst}_{\text{it}} \right|$$   (8)   

[\[CHART\]]{.chart}Figure 6: Aggregated MAE Across All Tests

  ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
  **This figure reports the aggregated MAE tests over multiple test periods. The four periods and associated patterns at the bottom of the graph. The results visualised here can be seen in** **Error! Reference source not found. as part of the *full* sample results. The above chart presents both an ML model that does incorporate analysts\' forecasts (ML), and a model that does not incorporate analyst forecasts (ML Ex-forecast) It is, therefore, unlike the time-series models that only use past actual EPS values. Further, using an OLS regression the Analyst Forecast has been bias corrected (Analyst Unbiased). From left to right, the above models relate to the following, as presented in the equations in the study,** $\widehat{p},\ {est\_ avg}_{t},\ EpsEst4Q,\ {EpsEst4Y}_{\text{it}},\ \text{EpsEstRW}_{\text{it}},\ \text{EpsEstLQ}_{\text{it}}\ \text{and}\text{\ ARIMA}$.
  ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

List of viable functions within stock trading:

1.  Directional predictions.

    a.  Reinforcement Learning trading strategy.

    b.  Forecasting [long
        term](https://github.com/Hvass-Labs/FinanceOps/blob/master/01_Forecasting_Long-Term_Stock_Returns.ipynb)
        stock returns,
        [two](https://github.com/Hvass-Labs/FinanceOps/blob/master/01B_Better_Long-Term_Stock_Forecasts.ipynb),
        [three](https://github.com/Hvass-Labs/FinanceOps/blob/master/01C_Theory_of_Long-Term_Stock_Forecasting.ipynb)

    c.  [Short term](https://github.com/anfederico/Clairvoyant) stock
        movement.

2.  Statistical Moment Predictions

    d.  [Correlational](https://github.com/imhgchoi/Corr_Prediction_ARIMA_LSTM_Hybrid)
        prediction.

    e.  Volatility prediction.

3.  Pairs Trading

    f.  [Actor critic](https://github.com/jjakimoto/DQN).

4.  Continuous value prediction

    g.  [Time series
        regression](https://github.com/RajatHanda/Finance-Forecasting/blob/master/LSTM-RNN/Amazon_Stock_Prediciton.ipynb)
        for a single continuous value prediction using RNN.
        ([Two](https://github.com/VivekPa/IntroNeuralNetworks/blob/master/LSTM_model.py))

    h.  Long term stock forecasting.

5.  Phase Change:

    i.  [Predicting market
        bottoms](https://github.com/borisbanushev/stockpredictionai/blob/master/readme2.md#reinforcementlearning)
        using mixture models.

6.  Probability weighted directions.

7.  Feature or variable Extraction

    j.  [Neural network extraction](https://github.com/VivekPa/AIAlpha)
        using stacked autoencoders.

    k.  Analytical extraction using time series TSFresh

8.  Unsupervised factor betas.

9.  Anomaly detection for a pairs trading framework.

10. News and sentiment incorporation.

    l.  [Headline
        analysis](https://github.com/keon/deepstock/tree/master/news-analysis)
        using CNN on time series data.

    m.  [Stock symbol](https://github.com/achillesrasquinha/bulbea)
        twitter sentiment score.

Additional things of import, [exit
rules](https://github.com/jjakimoto/DQN),
[start-to-finish](https://github.com/BlackArbsCEO/mixture_model_trading_public),

**Weight optimisation using signals.**

1.  Optimisation using
    [Signals](https://github.com/Hvass-Labs/FinanceOps/blob/master/03_Portfolio_Optimization_Using_Signals.ipynb).

2.  Multi-objective optimisation.

If you were forced by gunpoint to build a system that incorporates all
this technologies you would do it as follow.

Statistical arbitrage, you can also use the prediction on various stock
prices to find the expected correlation between the stocks.

**Additional Financial Disciplines**

Before I sign off, lets quickly run through some applications in the
other half of finance services, being corporate and retail banking,
investment banking, insurance services, payment services, financial
management, and advisory services.

1.  Corporate and Retail Banking

2.  Investment Banking

3.  Insurance Services

4.  Payment Services

5.  Financial Management

    -   Public Finance

    -   Financial Economics

    -   Management Accounting

6.  Advisory Services

    -   Personal Finance

    -   Consulting

7.  Investment Services

    -   Private Equity and Venture Capital

    -   Wealth Management

    -   [Broker Dealer]{.underline}

<!-- -->

6.  [Broker Dealer]{.underline}

    k.  Liquidity

        i.  Trading Strategies

        ii. Weight and Strategy Optimisation.

    l.  Strat:

        iii. Extreme Risk

        iv. Simulation

    m.  Capital Management

7.  Academic Finance

**Deployment**

This is awesome, loads your model into a [flask
API](https://nbviewer.jupyter.org/github/albahnsen/ML_RiskManagement/blob/master/notebooks/11_Model_Deployment.ipynb).
How to deploy your model, also deploy as pip project would be cool.
Actually there is a lot of fun deployment things you can do. There is
some great
[deployment](https://medium.com/@maheshkkumar/a-guide-to-deploying-machine-deep-learning-model-s-in-production-e497fd4b734a)
articles out there. Great answer and
[grid](https://www.quora.com/How-do-you-take-a-machine-learning-model-to-production)
of how to take these models into product, especially also well written.

**Data Processing:**

-   [Advanced
    ML](https://github.com/BlackArbsCEO/Adv_Fin_ML_Exercises) -
    Exercises too Financial Machine Learning (De Prado).

-   [Advanced ML II](https://github.com/hudson-and-thames/research) -
    More implementations of Financial Machine Learning (De Prado).

Feature Engineering:

Here is a notebook specifically [for feature
engineering](https://github.com/joelQF/quant-finance/blob/master/reference-notebooks/feature_engineering.ipynb),
this person has many alternative implantations too.

**Some further things to note:**

8.  [Broker Dealer]{.underline}

    n.  Liquidity

        v.  Market Making

        vi. Predicting Customer Needs

        vii. Shocks and volatiliy

    o.  Strat:

        viii. Extreme Risk

        ix. Simulation

    p.  Capital Management

Numerical computation, exploratory data analysis, Bayesian statistics,
diagnostics and monitoring, optimisation and estimation, inference

I guess its good to stat off with coding languages and build up from
there. Python is a good mid-level language, further adaptions can drive
it to a well performing language. Scala is interesting. And c has been
with us for ages. If I had to hedge my bets, I would do a 60/20/20 on
Python/Scala/C.

I guess you are running the regressions for individual stocks, as
opposed to portfolios of stocks. Given that individual stock return data
is very noisy, it is not surprising that you are getting low R\^2s. Fama
and French (1993) get R\^2s of around 90% because the dependent
variables in their regressions are excess returns on size and
book-to-market sorted **portfolios**, so the firm-specific noise has
been averaged out. That said, low R\^2s are not a problem if your
purpose is to calculate idiosyncratic volatility.

People really recommend these books, more to be done:

1.  Financial Shenanigans

2.  **The Art of Short Selling**

<!-- -->

1.  **You Can Be A Stock Market Genius**

2.  **The Intelligent Investor**

3.  

The different flavour of quant investing:
<https://www.bloomberg.com/news/articles/2018-10-02/your-guide-to-the-many-flavors-of-quant-investing-quicktake>

International finance seems to be increasingly interesting:
<http://www.onlinefinancedegree.org/quintessential-reading-list/>

Portfolio management,

asset management, private equity and venture capital. Asset management
here also refers to fund management. Fund mangagers have more assets
than banks. Asset management are slowly contributing in higher
proportion to group profits over years.

Within trading you might have hedge fund, mutual fund management.

The term finance encompasses a eight trillion dollar industry, second
only to information technology. Banking, consumer finance,
insurance/mortgages, Financial management -- budgeting, strategic,
construction etc -- possibly more accounting. Then you have quantitative
finance.

For example in banking and insurance we have: consumer finance,
valuation, fraud, insurance and risk and then on the business side,
management and operations.

Using the word quantitative is ludicrous, they all are quantitative

Banks are not allowed to steer wealth clients towards their own
investment products without proper disclosure.

Most banks have retail banking, commercial banking, investment banking
and asset & wealth management.

(funds management). Front office fund management can largely be divided
into sales, risk, portfolio & investment management. The financial
machine learning in this paper focuses on risk, portfolio and investment
management. Sales benefits mostly from business machine learning, not so
much financial machine learning.

Retail and commercial Insurance and banking still far outweigh other
industries when it comes to revenue. Actuarial advances in machine
learning has mostly play its part in their early adoption.

If I had to list them, banking insurance, investment services, financial
management, payment services, and lastly advisory services.

, financial economics, quantitative finance, investment finance, risk
management.

Accounting is the language of business, but it needs a few ingredients
like incentive structures and perception to take the words of the
language and turn it into song. Finance is appalling in that the work
done in finance is also the techniques used to minimise the losses of
the financial industry.

International finance is an interesting

There is about 18 trillion dollars of assets held in commercial banks.
About 2.4 trillion in commercial and industrial loans.

Typically commercial loans are the biggest, then residential mortgages,
both accounting for about 70% of loans, then consumer loans, credit card
loan and commercial real estate loans collectively account for about 10%
each.

Asset management profits remain around \$40 billion.

Asset managers draw about 683 billion

Asset management net income is around \$60 billion, revenue \$200
billion and assets of \$60 trillion. Shit asset under management is not
the same as assets, in fact these asset managers owes them money. So
here you would have to get the assets under management (i..e. loans and
reserves) and asset data. This is a good website to try and understand
the industry size. Wait I guess it is assets, but also
loans.<https://csimarket.com/Industry/Industry_Data.php?s=700>

For me assets are not that important, neither is revenue, what is more
important is the net income, i.e. the actual value that these firms
represent, revenue, assets and employees might have some other reasons
to be referred to.

The profit margins on finance is not all that high, the profit margins
on REITs are quite high though. IT has a big profit margin though.

Stochastic processes are often used in conjunction with Monte Carlo
methods to derive fair values for over the counter derivatives. This is
a very valid point. Other techniques might inlcude machine leanring to
ingest etc. If the current price of a security reflects all historical
price data, then the expected price of the security tomorrow *with
respect to*historical price data is greater than today\'s price because
securities are *risky* assets and we should be rewarded for taking on
that risk, This is called a Submartingale random walk and it is the
underlying assumption of most quantitative models. The Submartingale
Random Walk Hypothesis is a *consequent* of the Efficient Market
Hypothesis which is the *consequent* of active investment,

f the markets are random walks, what type of distribution to they have?

Generally speaking you get two types of randomness tests: parametric and
nonparametric.

-   Parametric tests assume something about the underlying distribution

-   Nonparametric tests don\'t assume anything about the underlying
    distribution

A good test of the random walk hypothesis makes few assumptions about
the data \... but may, as a result, be less powerful than another more
specific test.

This, and biases in randomness tests, are the reason why I believe we
should ensemble randomness tests together when testing the random walk
hypothesis.

And if the markets are random walks, are they random in all frequencies?

Most statistical tests of randomness are conducted on returns computed
over a specific period of time, usually daily. But just because daily
returns are random doesn\'t automatically imply that weekly or monthly
returns are random too.

As such most randomness tests are conducted in multiple frequencies or
rather, across multiple lags. Eugene Fama\'s original paper looked at
lags from one to ten days.

All of these issues are what inspired me to write the emh package for R.
This package, which we will be going through shortly, makes it
increadibly easy to run a suite of randomness tests on a financial time
series object and extract the results of each test in the suite on the
data sampled at different frequencies.

The Five Types of Randomness Tests
----------------------------------

For weak-form market efficiency testing there are five types of
randomness tests,

9.  Runs - the number of runs or the length of the longest (+) or (-)
    run

10. Serial correlation - non-zero correlation between an asset and
    itself

11. Unit roots - the presence of reversion to the trend line after
    shocks

12. Variance ratios - higher or lower variances at *specific
    frequencies*

13. Complexity tests - non-computability or compressibility in markets

These are all univariate tests done to determine whether a time series
of returns,
![F,{6a85839e-aa35-4bee-8593-0b8a798cb7ab}{18},0.6666667,0.6666667](media/image13.png "Image download failed."){width="0.375in"
height="0.375in"}, is random with respect to itself,
![F,{6a85839e-aa35-4bee-8593-0b8a798cb7ab}{19},0.8333333,0.6666667](media/image13.png "Image download failed."){width="0.375in"
height="0.375in"}. Multivariate tests do exist but have, unfortunately,
not been applied very often in the context of market efficiency testing
\...

Security market prices cannot be fully described by random walks. There
is little evidence to support the Efficient Market or Random Walk
Hypotheses so the true story is probably that security prices contain
some noise and some signal.

As such, many of our quantitative models are incorrect meaning that
assets are mispriced and risk is misunderstood. But let\'s not \"throw
the baby out with the bathwater\". Random walks aren\'t perfect but they
are the best tool we have right now and they do work.

That said, in the long run I think that we, as an industry, need to
adopt semi-stochastic models learnt by Machine Learning algorithms. If
ML can learn to [write like
Shakespeare](https://techcrunch.com/2014/01/26/swift-speare/), I\'m sure
it can learn how to simulate stock markets better than random walks!

Further than that you can look into passive equity and fixed income,
mixed and active for all types.

Although one might enjoy scoffing at the Fama-French models, they do
pick up betas uncorrelated with the market, while simultaneously
explaining a lot of the variation in price.

Financial institutions (FIs) can broadly be divided into those that
provide financial services

Finance is particularly

Banks don't really make money on the reserves.

Asset managers have about \$40 trillion under management, private equity
and venture capital firms, probably have around \$5 trillion under
management

Meta-models, great example of layered optimisation:
<https://quant.stackexchange.com/questions/12774/do-hedge-fund-trading-desks-use-portfolio-optimization>

Ideas:

1.  What you do is write the book with a lot of Collaboratory notebooks.

2.  Use the notes that you have collected on the subject and your
    monopoly theories.

    a.  Also the JP morgan document might in itself be quite useful.

3.  Book should be as prescient as possible.

4.  Talk about surveys ask people to take them.

5.  The point is not to have the research in-house rather refer to
    reputable sources, if you have to do it yourself, that is fine too.

6.  I would have to show some mathematical work, but show some
    higher-level interpretation.

7.  You want to make it ageless, add high level math in the appendix.

8.  I would have to unfortunately steal this code and create
    Collaboratory files.

    b.  To make it your own, you can attempt to add the mathematics to
        the file.

9.  New rule, only include stuff with machine learning applications.

10. There is an interesting relationship between capital management,
    trading strategies and portfolio optimisation that I don't yet
    understand:

    c.  In a trading strategy you have money management for each
        individual security, in various securities, you have money
        management too, but it is continuous as opposed to discrete
        (discrete Kelly criterion), when it is continuous you use
        portfolio optimisation. So you can use portfolio optimisation at
        the security level too.

11. I can only take part credit for this repository: a lot of the work
    would not be possible without the contribution of the following
    people to the open source community:

    d.  Jacque Joubert:

    e.  Relies santar

    f.  Caspar Darso.

12. Feature engineering you can add your own additions too.

13. You can probably find a way to get your prior Dropbox code into
    Google Colaboratory -- Lets call it snips.

14. All google Colab files should collectively appear on one file in the
    beginning.

15. Take a stab at code reproducibility -- say that all code should be
    reproducible and be able to execute it in the cloud without
    restrictions.

16. I am not adding short selling, but you can eventually add VIX.

17. At no point do I give credit where credit is due.

18. In case I made a mistake somewhere, see this for all the code and
    all the data.

19. You will have to elucidate the SHAP value calculation,.

20. Further you would have to talk about statistical significance and
    how it can be achieved in finance.

21. You are going to need a section called techniques.

Hi Guys,

I thought I should write a few chapters to explain to you how the field
off financial machine learning (FinML) is developing. It's also a good
way for me to consolidate my thoughts. Although it is still a relatively
juvenile field in finance (note not a juvenile technology), researchers
have been slow to provide structure to the field. This is my attempt.

Derek

H

Resources;

1.  Handson Unsupervised ML looks interesting:
    <https://www.amazon.com/Hands-Unsupervised-Learning-Using-Python/dp/1492035645>

2.  This looks good even uses stock trading using RL - all of these
    books seem to have financial applications, it might also have a
    github,
    <https://www.amazon.com/Deep-Reinforcement-Learning-Hands-Q-networks/dp/1788834240/ref=zg_bs_271581011_7?_encoding=UTF8&psc=1&refRID=P97H4PZ2F0G4932CYTNT>

3.  Maybe also look at hands-on RL or the seminal text books

4.  Already good books on GAN

5.  Also a good NLP book that came out recently,
    <https://www.amazon.com/Natural-Language-Processing-Action-Understanding/dp/1617294632/ref=zg_bs_271581011_13?_encoding=UTF8&psc=1&refRID=P97H4PZ2F0G4932CYTNT>

6.  Another good overleaf:
    <https://www.overleaf.com/articles/machine-learning-for-trading/bbckwqwmdnrw>

7.  Ito's formula, diffusion process, etc.

8.  There is another book for Python on Finance: Mastering Python for
    Finance: Implement advanced <https://www.amazon.com/dp/1789346460> -
    he seems to have more of a high frequency slant.

9.  Handson python book
    <https://www.amazon.com/Hands-Python-Finance-implementing-strategies/dp/1789346371/ref=pd_rhf_dp_s_vtp_ses_clicks_nonshared_1_2/140-0723777-7830921?_encoding=UTF8&pd_rd_i=1789346371&pd_rd_r=cd93c25a-d429-4b1d-b94f-dd30bdcd28a9&pd_rd_w=rjj5Q&pd_rd_wg=GVDNS&pf_rd_p=0be1f44a-f7fa-49fd-981f-05211a6bdf7e&pf_rd_r=4HEDN8GRBTVP6MWPNXDA&psc=1&refRID=4HEDN8GRBTVP6MWPNXDA>

10. Great topics on feature generation:
    <https://github.com/krishnaik06/Feature-Engineering>

11. Here you go mastering python for finance, for some reason it has
    like no stars, I think this guy is actionally good:
    <https://github.com/jamesmawm/mastering-python-for-finance-second-edition>

12. [Good mathematic finance and machine learning
    library.](https://github.com/thalesians)

13. Really nice dashboard python:
    <https://github.com/ranaroussi/quantstats>

14. Wrong language systemic risk:
    <https://github.com/TommasoBelluzzo/SystemicRisk>

15. Misreporting -- matlab again -
    <https://github.com/TommasoBelluzzo/HFMRD>

16. Important advanced FML resource,
    <http://reasonabledeviations.science/notes/adv_fin_ml/>

17. [Good NLP Corpus](https://github.com/trinker/lexicon)

18. [Bayesian Asset Allocation](https://github.com/fxing79/ibaa)

19. 20. [Excellent capital markets economic data that supplements
    FRED](https://capitalmarketsdata.blogspot.com/2019/05/cmd-md-monthly-database-for-macro.html)[Good
    website
    marco](https://gmarti.gitlab.io/qfin/2018/10/02/hierarchical-risk-parity-part-1.html)
    -- excellent website

21. [Probabilistic time series:](https://github.com/awslabs/gluon-ts)

22. 23. [What this website is amazing,
    https://quantdare.com/ranking-quality/](https://quantdare.com/ranking-quality/)[Here
    you go, the Kelly criterion good
    website.](https://quantdare.com/kelly-criterion/)

24. Remember about black arbs: a lot of interesting work:
    <http://www.blackarbs.com/notable-projects>

25. [Excellent website brining all quant research under
    one](https://quantocracy.com/).

26. More contacts for
    [linkeding](https://conference.unicom.co.uk/sentiment-analysis/2018/zurich/).

27. Not sure what to say, some [time
    series](https://github.com/mrefermat/FinancePhD/tree/master/FinancialExperiments)
    experimentations. - Nice SARIMAX and Decomposition.

28. Here is
    [returns](https://github.com/joelQF/quant-finance/blob/master/Artificial_IntelIigence_for_Trading/cumsum_and_cumprod.ipynb)
    done the right way.

29. I like this idea, multual fund recommender.
    <https://github.com/frechfrechfrech/Mutual-Fund-Recommender>

30. [Some of that JP Morgan
    implementation.](https://github.com/salimngit/DeepFin-Series-JPMorgan/blob/master/Machine%20Learning%20Tutorial/Machine%20Learning%20Tutorial%20Series.ipynb)

31. Future, adaptive asset allocation -- looks intresting.
    <https://github.com/darwinsys/Trading_Strategies/blob/master/ETF/Adaptive%20Asset%20Allocation.ipynb>

Additional Work:

1.  <https://github.com/chicago-joe/IB_PairsTrading_Algo>

2.  Traingle returns:
    <https://de.scalable.capital/quants-perspective/return-triangles>

3.  ML finance critixism,
    <https://de.scalable.capital/quants-perspective/return-triangles>

4.  Really good cash-flow spending [using
    keras](https://github.com/druce/safewithdrawal_tensorflow/blob/master/Safe%20Retirement%20Spending%20Using%20Certainty%20Equivalent%20Cash%20Flow%20and%20TensorFlow.ipynb).

5.  This is great: Deep Learning Empirical Asset Pricing -- It even has
    WRDS file associated:
    <https://github.com/yolsever/ML-in-equity-prediction>

6.  Absolutely great websites for links on data and replication
    websites: <https://www.sebastianstoeckl.com/links/>

7.  Excellent website for additional python and SEC stuff:
    <http://kaichen.work/?cat=10>

8.  Del Prado has some actual python: <http://www.quantresearch.org/>

9.  Great notebook on efficient markets:
    <https://github.com/StuartGordonReid/Python-Notebooks/blob/master/RFinance%20Talk%20Market%20Efficiency.ipynb>

10. So awesome great article on stochastic processes:
    <https://github.com/StuartGordonReid/Python-Notebooks/blob/master/Stochastic%20Process%20Algorithms.ipynb>

11. Machine learning framework for stock selection:
    <https://github.com/fxy96/Stock-Selection-a-Framework> and
    <chrome-extension://cbnaodkpfinfiipjblikofhlhlcickei/src/pdfviewer/web/viewer.html?file=https://arxiv.org/pdf/1806.01743.pdf> -
    Genetic and some crazy things going on.

12. Some finance replication papers:
    <https://github.com/HoangT1215/Finance-papers-replication>

13. Advanced quantopian projects:
    <https://github.com/clumdee/Python-for-Finance/tree/master/11-Advanced-Quantopian-Topics> -
    bor

14. Trading behaviour using google trends -- excellent:
    <https://github.com/twiecki/financial-analysis-python-tutorial/blob/master/2.%20Pandas%20replication%20of%20Google%20Trends%20paper.ipynb>

15. Microstructure python-
    <https://github.com/vgreg/python-finance-unimelb2018/blob/master/notebooks/microstructure/ASX_Microstructure.ipynb>

16. 3\. Large scale equity trading:
    <https://github.com/yolsever/ML-in-equity-prediction>

17. 4\. The point is probably not that you want to do the above, but that you
    want to use it in your future prjects,

18. 

> Machine Learning Numpy clearly has some future.
> <https://github.com/huseinzol05/Machine-Learning-Numpy>.

19. Do what one might call an extensive data exploration exercise :
    <https://github.com/huseinzol05/Stock-Prediction-Models/blob/master/misc/tesla-study.ipynb>

20. Under visual exploration you can add numerical techniques and play
    with for example the following,
    <https://github.com/huseinzol05/Stock-Prediction-Models/blob/master/simulation/mcmc-stock-market.ipynb>

Shit this unsupervised fools book should probably be read:

Ankur A. Patel is the Vice President of Data Science at 7Park Data, a
Vista Equity Partners portfolio company. At 7Park Data, Ankur and his
data science team use alternative data to build data products for hedge
funds and corporations and develop machine learning as a service (MLaaS)
for enterprise clients. MLaaS includes natural language processing
(NLP), anomaly detection, clustering, and time series prediction. Prior
to 7Park Data, Ankur led data science efforts in New York City for
Israeli artificial intelligence firm ThetaRay, one of the world\'s
pioneers in applied unsupervised learning.\
\
Ankur began his career as an analyst at J.P. Morgan, and then became the
lead emerging markets sovereign credit trader for Bridgewater
Associates, the world\'s largest global macro hedge fund, and later
founded and managed R-Squared Macro, a machine learning-based hedge
fund, for five years. A graduate of the Woodrow Wilson School at
Princeton University, Ankur is the recipient of the Lieutenant John A.
Larkin Memorial Prize.\
\
He currently resides in Tribeca in New York City but travels extensively
internationally.

False strategies:

<https://github.com/Rachnog/Deep-Trading/blob/master/top-ten-mistakes/Illustration%20of%20a%20fake%20strategy.ipynb>

Discretization is the name given to the processes and protocols that we
use to convert a continuous equation into a form that can be used to
calculate numerical solutions. **Dichotomization**is the special case of
discretization in which the number of discrete classes is 2, which can
approximate a continuous variable as a [binary
variable](https://en.wikipedia.org/wiki/Binary_variable). Whenever
continuous data is **discretized**, there is always some amount
of [discretization
error](https://en.wikipedia.org/wiki/Discretization_error). The goal is
to reduce the amount to a level
considered [negligible](https://en.wiktionary.org/wiki/negligible) for
the [modeling](https://en.wikipedia.org/wiki/Conceptual_model) purposes
at hand. Here is some information on how one can discretise and

Interesting people do grid searches for cointegration too, clearly an
overfitting exercise.

**Websites in General:**

Hudson and Thames Research:
<http://www.quantsportal.com/open-source-hedge-fund/>

Has some cool analyses in R thought,
<http://www.reproduciblefinance.com/2019/02/18/2018-sector-analysis-part-2/>

![Machine Learning Map](media/image14.png){width="6.268055555555556in"
height="4.37257874015748in"}

To be honest, this is an amazing result:

Using a Financial Training Criterion Rather than a Prediction Criterion - [Yoshua Bengio](https://www.worldscientific.com/doi/abs/10.1142/S0129065797000422)(1997)
==================================================================================================================================================================

The application of this work is to decision making with financial time
series, using learning algorithms. The traditional approach is to train
a model using a prediction criterion, such as minimizing the squared
error between predictions and actual values of a dependent variable, or
maximizing the likelihood of a conditional model of the dependent
variable. We find here with noisy time series that better results can be
obtained when the model is directly trained in order to maximize the
financial criterion of interest, here gains and losses (including those
due to transactions) incurred during trading. Experiments were performed
on portfolio selection with 35 Canadian stocks.

[^1]: Supervised learning refers to the mathematical structure
    describing how to make a prediction $\mathbf{y}_{\mathbf{i}}\ $given
    $\mathbf{x}_{\mathbf{i}}$. The value we predict,
    $\mathbf{y}_{\mathbf{i}},$ has different interpretations depending
    on whether we are implementing a classification or a regression
    task. In classification task prediction,
    $\mathbf{y}_{\mathbf{i}},\ $is the probability of an earnings
    surprise event of some specified threshold. For the regression task,
    the $\mathbf{y}_{\mathbf{i}}$ is the actual EPS value for the
    quarter. The inputs, $\mathbf{x}_{\mathbf{i}},$ have been selected
    based on selection procedures applied once for each of the models.
    Apart from the different prediction types, in the classification
    task, the model gets logistic transformed to obtain a vector of
    probabilities for each observation and associated categories. In
    supervised learning, parameters play an important role. The
    parameters are the undetermined part that we are required to learn
    using the training data. For example, in a linear univariate
    regression,
    ${y\hat{}}_{i} = \sum_{j}^{}{\theta_{j}\mathbf{x}_{\mathbf{\text{ij}}}}\ $,
    the coefficient $\theta$ is the parameter.

    The task is ultimately to find the best parameters and to choose a
    computationally efficient way of doing so. This choice is largely
    driven by whether or not we are working with a regression or
    classification problem. To measure a model\'s performance, given
    some parameter selections, we are required to define an objective
    function. The following is a compressed form of the objective
    function, $Obj(\Theta) = L(\theta) + \Omega(\Theta)$. In this
    equation, L is the training loss function; the regularisation term
    is $\Omega$. The training loss function tests the predictive ability
    of the model using training data. A commonly used method to
    calculate the training loss is the mean squared error,
    $L(\theta) = \sum_{i}^{}{(\mathbf{y}_{i} - {y\hat{}}_{i})}^{2}$.
    Thus, the parameters get passed into a model that calculates,
    $\mathbf{y\hat{}}_{\mathbf{i}}$, a series of predictions, that gets
    compared against the actual values in a mean squared error function
    to calculate the loss. The regularisation term controls the
    complexity of the model, which helps to avoid overfitting. The
    Extreme, X, of the XGBoost model, relates to an extreme form of
    regularisation that controls for over-fitting, leading to improved
    performance over other models. There are countless ways to
    regularise models, in essence, we constrain a model by giving it
    fewer degrees of freedom; for example, to regularise a polynomial
    model, we can transform the model to reduce the number of polynomial
    degrees. The tree ensemble can either be a set of classification or
    a set of regression trees. It is usually the case that one tree is
    not sufficiently predictive, hence the use of a tree ensemble model
    that sums the predictions of many trees together. Mathematically,
    the model can be written in the following form
    ${\widehat{y}}_{i} = \sum_{k = 1}^{K}{f_{k}(\mathbf{x}_{\mathbf{i}})},\ f_{k} \in F$.
    Here, *K* is the number of trees, and *f* represents one possible
    function from the entire functional space *F*. *F* is a set of all
    possible classification and regression trees (CARTs). This
    expression then simply adds multiple models together that lives
    within the allowable CART function space. Therefore, combining the
    model, the training loss and regularisation function, we can gain
    our objective function and seek to optimise it, the function can be
    written as follows,
    $\text{Obj}\left( \theta \right) = \sum_{i}^{n}{l\left( \mathbf{y}_{\mathbf{i}},\text{\ \ }{\widehat{y}}_{i}^{(t)} \right)} + \sum_{k = 1}^{K}{{\Omega(f}_{i})}$.
    Thus far, the model is similar to that of a random forest, the
    difference being in how the models are trained.

    For the next part, we have to let the trees learn, so for each tree,
    we have to describe and optimise an objective function, we can start
    off by assuming the following function,
    $\text{Obj} = \sum_{i}^{n}{l\left( y_{i}, \right)} + \sum_{i = 1}^{t}{{\Omega(f}_{i})}$.
    By looking at the function it is important that we identify the
    parameters of the trees. We want to learn the functions, $f_{i}$,
    each which contains a tree structure and associated leaf scores.
    This is more complex than traditional methods where you can simply
    take the gradient and optimise for it. Instead, Gradient Boosting
    uses an additive strategy, whereby we learn to adjust and add an
    extra tree after each iteration. We write our prediction value at
    step *t* as ${\widehat{y}}_{i}^{(t)}$, so that we have
    ${\widehat{y}}_{i}^{(t)} =$
    $\sum_{k = 1}^{t}{f_{k}\left( \mathbf{x}_{\mathbf{i}} \right) = {\widehat{y}}_{i}^{(t - 1)} +}f_{t}\left( \mathbf{x}_{\mathbf{i}} \right)$.
    Then we simply choose the tree that optimises our objective,
    $\text{Obj}^{(t)} = \sum_{i}^{n}{l\left( y_{i},{\widehat{y}}_{i}^{(t)} \right)} + \sum_{k = 1}^{t}{{\Omega(f}_{i}) = \ \sum_{i = 1}^{n}{l\left( \mathbf{y}_{\mathbf{i}},{\widehat{y}}_{i}^{(t - 1)} \right) +}f_{t}\left( \mathbf{x}_{\mathbf{i}} \right) + {\Omega(f}_{t}) + constant\ }$.
    By using MSE as the loss function, it becomes
    $\text{Obj}^{(t)} = \ \sum_{i = 1}^{n}{\left\lbrack 2\left( {\widehat{y}}_{i}^{\left( t - 1 \right)} - \mathbf{y}_{\mathbf{i}} \right)f_{t}\left( \mathbf{x}_{\mathbf{i}} \right) + \ {f_{t}\left( \mathbf{x}_{\mathbf{i}} \right)}^{2} \right\rbrack +}$
    ${\Omega(f}_{t}) + constant$. The form of MSE is easy to deal with.
    The Taylor expansion can simply be taken to the second order.
    $\text{Obj}^{(t)} = \ \sum_{i = 1}^{n}\left\lbrack l\left( y_{i}\mathbf{,}{\widehat{y}}_{i}^{\left( t - 1 \right)} \right)\mathbf{+}g_{i}f_{t}\left( \mathbf{x}_{\mathbf{i}} \right) + \ {\frac{1}{2}h}_{j}{f_{t}}^{2}\left( \mathbf{x}_{\mathbf{i}} \right) \right\rbrack\  + \ {\Omega(f}_{t}) + constant$,
    where $g_{i}$ and $h_{i}$ is defined as,
    $g_{i} = \partial_{{y\hat{}}_{i}^{(t - 1)}}\ l(\mathbf{y}_{\mathbf{i}},{\widehat{y}}_{i}^{(t - 1)})$,
    $h_{i} = \partial_{{y\hat{}}_{i}^{(t - 1)}}^{2}\ l(\mathbf{y}_{\mathbf{i}},{\widehat{y}}_{i}^{(t - 1)})$.
    After all the constants are removed, then the objective at *t* get
    transformed to,
    $\sum_{i = 1}^{n}\left\lbrack g_{i}f_{t}\left( \mathbf{x}_{\mathbf{i}} \right) + \ {\frac{1}{2}h}_{j}{f_{t}}^{2}\left( \mathbf{x}_{\mathbf{i}} \right) \right\rbrack\  + \ {\Omega(f}_{t})$.
    This then becomes an adjusted optimization function for the new
    tree. Although we have looked at the training step, we have not
    looked at regularisation yet. The next step is to specify how
    complex the tree should be, ${\Omega(f}_{t})$. To do this we can
    improve the tree definition to *F(x),*
    $f_{t}\left( \mathbf{x} \right) = w_{q\left( \mathbf{x} \right)},\ w \in \mathbb{R}^{T},\ q:\mathbb{R}^{m} \rightarrow \left\{ 1,\ 2,\ldots,T \right\}.$
    Here *w* represents the scores of the leaves presented in vector
    form and *q* represents a function that assigns each point to the
    appropriate leaf, lastly *T* denotes how many leafs there are. The
    complexity can be defined as
    $a\ \Omega(f) = \gamma T + \frac{1}{2}\lambda\sum_{j = 1}^{T}w_{j}^{2}$;
    there are more ways to formulate and define how complex a model is
    or should be in practice, but this one is quite practical and easy
    to conceptualise. Once the tree model is described, the objective
    value w.r.t. the *t-th* tree can be written as follows:
    $\text{Obj}^{(t)} = \ \sum_{i = 1}^{n}\left\lbrack g_{i}f_{t}\left( \mathbf{x}_{\mathbf{i}} \right) + \ {\frac{1}{2}h}_{j}{f_{t}}^{2}\left( \mathbf{x}_{\mathbf{i}} \right) \right\rbrack\  + \ \gamma T + \frac{1}{2}\lambda\sum_{j = 1}^{T}{w_{j}^{2}\ }$
    =
    $\sum_{J = 1}^{T}\left\lbrack \left( \sum_{i \in I_{j}}^{}g_{i} \right)w_{j} + \ \frac{1}{2}\left( \sum_{i \in I_{j}}^{}g_{i} + \lambda \right)w_{j}^{2} \right\rbrack\  + \ \gamma T$,
    where
    $I_{j} = \left\{ i \middle| q\left( x_{j} \right) = j \right\}\text{\ \ }$represents
    a full set of all the data points. as have been assigned to the
    *j-th* leaf. The equation can then further be compressed by
    describing $G_{j} = \ \sum_{i \in I_{j}}^{}g_{i}$ and
    $H_{J} = \ \sum_{i \in I_{j}}^{}h_{i}$, then $\text{Obj}^{(t)} =$
    $\sum_{J = 1}^{T}\left\lbrack G_{j}w_{j} + \ \frac{1}{2}\left( H_{j} + \lambda \right)w_{j}^{2} \right\rbrack\  + \ \gamma T$.
    In the preceding equation the weights, $w_{j}$ are independent w.r.t
    each other, the form
    $G_{j}w_{j} + \ \frac{1}{2}\left( H_{j} + \lambda \right)w_{j}^{2}$
    is quadratic, and the best weight for a structure *q(x)* is given by
    the following expression.
    $w_{j}^{*} = \  - \frac{G_{j}}{H_{j} + \lambda}$ ,
    $\text{obj}^{*} = \  - \frac{1}{2}\sum_{j = 1}^{T}\frac{G_{j}^{2}}{H_{j} + \lambda}\  + \ \gamma T$.
    This equation measures how good a tree structure $q(x)\ $is. A lower
    score is better for the ultimate structure of a tre. Now that we
    know how to measure the fittingness of a tree is, we can identify
    all the trees and select the best one. It is, however, not possible
    to approach it this way and instead has to be done for one depth
    level of a tree at a time. This can be approached by splitting a
    leaf into two sections and then recording its gain. The following
    equation represents this process,
    $Gain = \frac{1}{2}\left\lbrack \frac{G_{L}^{2}}{H_{L} + \lambda} + \frac{G_{R}^{2}}{H_{R} + \lambda} - \frac{\left( G_{l} + G_{l} \right)^{2}}{H_{L} + H_{R} + \lambda} \right\rbrack - \gamma$.
    If the gain obtained is equal to or smaller than $\gamma$, then it
    would be better if we do not add the branch to the tree, this is
    often referred to as the pruning technique. We basically search for
    the ultimate split, if all instances are sorted in order, we simply
    scan left to right to sufficiently calculate the structure scores of
    all possible solutions and then identify the most efficient split.

[^2]: For robustness, I have also tested for value weighted returns
    which showed a slight increase in improvement.

[^3]: http://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data\_library.html

[^4]: A Markovian process is a random (stochastic) process -- a
    collection of random variables indexed by time or space - that
    satisfies the Markov property of memorylessness. A Gaussian process
    is also stochastic where linear combinations of variables are
    normally distributed. A stochastic process that satisfies both of
    these criteria is a Gauss-Markov stochastic process. In a
    Gauss-Markov process, the mean can still be changed, by simply
    adding one to each finite time step, further the memorylessness and
    normal distribution can still be preserved while increasing
    volatility of the process. Additional criteria of stationarity can
    exist, in which case the memoryless, normally distributed process is
    also temporally homogenous in that the mean and variance does not
    change over time -- adding this additional criteria leads to a
    stochastic process called Ornstein-Uhlenbeck -- this process
    exhibits the mean reverting characteristic.

    I like this, it's a cool story: The first example, is about the
    Netflix challenge where Netflix offered a million dollar prize for
    the "best suggested movies to watch" algorithm. Jaffray Woodriff,
    founder and CEO of Quantitative Investment Management (\$3B in AUM),
    competed in the contest and took 3rd! Woodriff is a big proponent of
    ensemble methods and mentions such in his Hedge Fund Market Wizards
    chapter of Jack Schwager's great book. The team that actually won
    the contest was in a close race with the second place team and wound
    up running away with the first place prize by using an ensemble
    method applied to the other top 30 competitors' models. I am trying
    to convey financially incentivized practicality here; the story
    comes from Ensemble Methods in Data Mining by Giovanni Seni and John
    Elder with a foreword by Jaffray Woodriff.
