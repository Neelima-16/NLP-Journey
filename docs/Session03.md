Linear Algebra

**\
📌 Why Linear Algebra in NLP?**

In NLP, we convert words, sentences, and documents into numerical
formats --- usually **vectors** or **matrices** --- so that computers
can process them.

For example:

- Word embeddings like **Word2Vec**, **GloVe**, or **BERT embeddings**
  turn words into vectors.

- Similarity between words is measured via vector operations.

- Sentence transformers create high-dimensional sentence vectors.

And here's where linear algebra steps in.\
\
 

**📌 Linear Algebra for NLP --- Quick Cheat-Sheet**

  ------------------------------------------------------------------
  **Concept**        **What it is**           **Use in NLP**
  ------------------ ------------------------ ----------------------
  **Scalar**         Single number            Word frequency count

  **Vector**         1D array of numbers      Word embeddings
                                              (Word2Vec etc.)

  **Matrix**         2D array of numbers      Word co-occurrence
                                              matrix

  **Tensor**         Multi-dimensional array  Deep learning models
                                              (BERT)

  **Dot Product**    Measure of similarity    Cosine similarity
                     between vectors          calculation

  **Norm**           Length of a vector       Normalizing word
                                              vectors

  **Cosine           Angle between two        Check similarity of
  Similarity**       vectors                  words/sentences

  **Matrix           Combine or transform     Neural network layers,
  Multiplication**   data                     embeddings

  **Eigenvalues &    Find directions of       Dimensionality
  Eigenvectors**     maximum variance         reduction (PCA, LSA)\
                                              \
                                               
  ------------------------------------------------------------------

 

 

Graph theory & Networks

**\
📌 What is Graph Theory in NLP?**

A **graph** is a collection of:

- **Nodes (vertices)** --- which represent entities like words,
  documents, or people

- **Edges (connections)** --- which show relationships or interactions
  between them

In NLP, we often model text as a graph to:

- Find how words are related

- Build word co-occurrence networks

- Extract relationships between entities

- Detect communities or topics in a text

**📊 Example:**

**Words as nodes**

**Co-occurrence as edges**

 

arduino

CopyEdit

\"Neelima loves AI and AI loves NLP\"

Nodes:

- Neelima

- loves

- AI

- and

- NLP

Edges:

- (Neelima --- loves)

- (loves --- AI)

- (AI --- loves)

- (loves --- NLP)

**📌 Where is Graph Theory Used in NLP?**

✅ **Knowledge Graphs:**

Google Search Knowledge Graphs, showing related facts and entities

✅ **Word Co-occurrence Networks:**

Visualizing how words are related in a text corpus

✅ **Community Detection:**

Finding groups of words or topics that frequently occur together

✅ **Text Summarization:**

Graph-based ranking (like TextRank algorithm)

✅ **Named Entity Linking:**

Connecting recognized names to external knowledge bases via graph

✅ **Relation Extraction:**

Mapping how people, places, and events are related in a document

 

**Example Exercise: Build a Word Co-occurrence Graph using NetworkX**

**Let\'s do this practically --- tiny word graph.**

**Install:**

 

bash

CopyEdit

pip install networkx matplotlib

**Python Code 📜**

 

python

CopyEdit

import networkx as nx\
import matplotlib.pyplot as plt

\# Sample text\
text = \[\'NLP\', \'loves\', \'AI\', \'and\', \'AI\', \'loves\',
\'Neelima\'\]

\# Create an empty graph\
G = nx.Graph()

\# Add edges between consecutive words\
for i in range(len(text)-1):\
G.add_edge(text\[i\], text\[i+1\])

\# Draw the graph\
plt.figure(figsize=(8,5))\
nx.draw_networkx(G, with_labels=True, node_color=\'skyblue\',
node_size=2000, font_size=15, edge_color=\'gray\')\
plt.title(\"Word Co-occurrence Graph\")\
plt.show()\
\
**output:**

![](media/image1.png){width="5.0in" height="3.125in"}

**📌 Key Graph Theory & Network Concepts That Help NLP**

  ------------------------------------------------------------------------
  **Graph Theory    **How It's Used in NLP**     **Example Applications**
  Concept**                                      
  ----------------- ---------------------------- -------------------------
  **Nodes           Represent **words,           Words in a text
  (Vertices)**      sentences, entities, or      co-occurrence graph
                    documents** in a graph       

  **Edges**         Represent relationships or   \'AI\' connected to
                    co-occurrences between nodes \'NLP\' if they appear in
                                                 same sentence

  **Edge Weights**  Capture the **strength or    Higher weight for more
                    frequency of relationship**  frequent word pairs
                    between nodes                

  **Degree of a     Number of connections a      Frequent words like
  Node**            word/entity has ---          \'the\', \'is\' have high
                    indicates importance         degree

  **Path**          Sequence of connected nodes  Finding relations between
                    in the graph                 distant entities in a
                                                 document

  **Shortest Path** Minimum number of edges      Text summarization
                    between two nodes            (TextRank uses this for
                                                 scoring sentences)

  **Centrality      Identify the most important  Detecting key terms or
  Measures**        nodes in a graph             influential words in text

  **Community       Grouping similar             Topic modeling, word
  Detection /       words/entities based on      sense disambiguation
  Clustering**      connectivity                 

  **Graph Traversal Exploring related terms or   Discovering context words
  (DFS, BFS)**      entities through the graph   or neighbor entities

  **PageRank /      Graph-based ranking          Text summarization,
  TextRank          algorithms                   keyword extraction
  Algorithm**                                    
  ------------------------------------------------------------------------

 

 

 

Calculus

 

**📌 What is Calculus in NLP?**

**Calculus** in NLP is mostly about:

- **Optimization**

- **Gradients (derivatives)**

- **Updating parameters in models**

It's the mathematical tool that helps models **learn by adjusting their
internal parameters** to minimize errors during training.

 

**📊 Where Does Calculus Help in NLP?**

  ---------------------------------------------------------------------
  **Concept**            **Where It's Used**
  ---------------------- ----------------------------------------------
  **Derivatives**        Measure how changing a model parameter affects
                         the output

  **Gradient Descent**   Optimization algorithm to minimize model
                         errors

  **Partial              When dealing with multi-parameter functions
  Derivatives**          (like neural networks)

  **Chain Rule**         When computing gradients through layers in
                         models (Backpropagation)

  **Jacobian & Hessian   In second-order optimization or understanding
  (Advanced)**           how model output changes
  ---------------------------------------------------------------------

 

 

**📌 Practical Use-Cases of Calculus in NLP**

✅ **Training Word2Vec / GloVe**

Use **gradient descent** to adjust word vector values to reduce loss

✅ **Fine-tuning BERT / Transformers**

Backpropagation (which uses chain rule and derivatives) updates millions
of parameters

✅ **Sentiment Analysis Classifier**

Use derivatives to calculate loss and update model weights to improve
predictions

✅ **Optimization in Language Models**

Minimizing cross-entropy loss via calculus-based optimization algorithms

 

**📖 Example: How Gradient Descent Works in NLP**

Let's say our loss function is:

 

ini

CopyEdit

Loss = (predicted - actual)\^2

To reduce this loss, we:

1.  Compute the derivative of loss with respect to the model's
    parameters

2.  Update parameters in the opposite direction of the gradient

**Formula:**

 

ini

CopyEdit

new_weight = old_weight - learning_rate × derivative

Do this repeatedly (called **iterations/epochs**) until the loss becomes
very small.

 

**📌 Visual Intuition:**

Imagine a valley (the loss curve) and you want to get to the lowest
point (minimum error).

Calculus tells you:

- Which direction to move (gradient)

- How big a step to take (learning rate × gradient value)

This is **how models learn.**

 

**📊 Where You See It in NLP Model Training**

  ----------------------------------------------------------------
  **NLP Task**             **What Calculus Does**
  ------------------------ ---------------------------------------
  Word2Vec Training        Adjusts word vectors to better predict
                           context words

  Sentiment Classifier     Optimizes model weights to classify
                           texts correctly

  Text Generation (RNNs,   Tunes parameters to improve sequence
  LSTMs)                   prediction

  Fine-tuning Transformers Updates millions of parameters using
                           gradients
  ----------------------------------------------------------------

 

 

**📌 Key Calculus Concepts to Know for NLP**

- Derivative

- Partial Derivative

- Gradient

- Gradient Descent

- Chain Rule

- Loss Function

- Backpropagation (uses derivatives)

 

**✅ In Summary:**

**Calculus powers the learning process in NLP.**

It helps models:

- **Understand how bad their predictions are (loss function)**

- **Figure out how to adjust themselves to improve**

- **Use gradients to make small changes toward better performance\**
  Without calculus, **no AI or NLP model could learn from data.**

 

 

Information Theory

 

**📌 What is Information Theory in NLP?**

**Information theory** deals with measuring:

- **Uncertainty**

- **Surprise**

- **Information content in data**

In NLP, it helps quantify **how much information a word carries**, how
predictable language is, and how efficient a model's predictions are.

 

**📖 Key Concepts in Information Theory (Used in NLP)**

  ----------------------------------------------------------------------------
  **Concept**         **What It Measures**          **Where It's Used in NLP**
  ------------------- ----------------------------- --------------------------
  **Entropy (H)**     Average uncertainty or        Language modeling,
                      surprise in a probability     measuring dataset
                      distribution                  complexity

  **Cross-Entropy**   Difference between true       Loss function for text
                      distribution and predicted    classification and
                      distribution                  language modeling

  **Perplexity**      How well a model predicts a   Evaluating language models
                      sample (lower is better)      (Word2Vec, GPT, BERT)

  **Mutual            How much knowing one word     Word association strength,
  Information (MI)**  reduces uncertainty about     feature selection in text
                      another                       tasks

  **KL Divergence**   How different two probability Comparing model
                      distributions are             predictions vs actual
                                                    distribution
  ----------------------------------------------------------------------------

 

 

**📊 Real-World Examples in NLP**

✅ **Entropy**

High entropy: unpredictable text

Low entropy: repetitive or formulaic text

✅ **Cross-Entropy Loss**

Used as a loss function in:

- Text classification

- Language generation

- Word embedding training

✅ **Perplexity**

Used to evaluate:

- GPT

- BERT

- LSTM Language models

A lower perplexity = better model predictions.

✅ **Mutual Information**

Used to:

- Measure strength between words

- Pick useful features for text classification

- Extract word pairs for collocations (e.g. \'New York\')

 

**📖 Simple Example: Entropy Calculation**

If you have a tiny corpus with word probabilities:

 

mathematica

CopyEdit

P(A) = 0.5\
P(B) = 0.25\
P(C) = 0.25

**Entropy (H)** is:

 

matlab

CopyEdit

H = - Σ p(x) \* log2(p(x))\
= -(0.5\*log2(0.5) + 0.25\*log2(0.25) + 0.25\*log2(0.25))\
= 1.5 bits

Meaning: on average, 1.5 bits of information needed per word from this
distribution.

 

**📖 Cross-Entropy Example (as used in NLP Loss Functions)**

If true distribution is P and predicted distribution is Q:

 

perl

CopyEdit

H(P, Q) = - Σ p(x) \* log2(q(x))

Lower values = better model predictions.

 

**📖 Perplexity Example**

Perplexity is just:

 

mathematica

CopyEdit

2\^Entropy

Lower perplexity = better model.

 

**📌 Where You See It in NLP**

  -----------------------------------------------------------------
  **Task**                     **How Information Theory Helps**
  ---------------------------- ------------------------------------
  Language Model Training      Cross-entropy loss guides model
  (GPT/BERT)                   updates

  Text Generation              Perplexity used to evaluate model
                               fluency

  Text Classification          Cross-entropy loss for optimization

  Feature Selection in Text    Mutual Information to select
  Mining                       high-value words

  Sentiment Analysis, NER      Optimize predictions by minimizing
                               cross-entropy
  -----------------------------------------------------------------

 

 

**✅ In Summary:**

**Information Theory in NLP helps models:**

- **Measure uncertainty and surprise**

- **Quantify information content**

- **Optimize predictions via loss functions**

- **Evaluate how good or bad a model's predictions are**

Without these measures, AI models wouldn't know how "good" or "bad"
their guesses are.
