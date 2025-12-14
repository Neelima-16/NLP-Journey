Bag of words

 

**📌 Topic: Bag of Words (BoW) --- Feature Extraction ✅**

**What is it?**

Bag of Words is a simple, commonly used method to convert text into
numerical feature vectors that a machine learning model can process.

 

**📖 How it works:**

1.  **Tokenize** the text (split into individual words).

2.  **Build a vocabulary** of unique words from all documents.

<!-- -->

3.  For each document, **count the occurrence** of each word from the
    vocabulary.

4.  Represent each document as a vector of these counts.

 

**📦 Example:**

Let\'s say you have:

- Doc 1: *\"I love NLP\"*

- Doc 2: *\"I love Python\"*

**Vocabulary:** \[\'I\', \'love\', \'NLP\', \'Python\'\]

  -------------------------------------------------
          **I**   **love**   **NLP**   **Python**
  ------- ------- ---------- --------- ------------
  Doc 1   1       1          1         0

  Doc 2   1       1          0         1
  -------------------------------------------------

 

 

**📊 Limitations:**

- Ignores word order and context.

- High dimensionality with large vocabularies.

- Sparse matrix (many zeros).

>  
>
>  

**📌 Manual Python Implementation of Bag of Words:**

\# Sample documents\
documents = \[\
\"I love NLP\",\
\"I love Python\"\
\]

\# Step 1: Tokenize and build vocabulary\
vocab = set()\
for doc in documents:\
words = doc.lower().split()\
vocab.update(words)

\# Sort vocabulary for consistent ordering\
vocab = sorted(list(vocab))

\# Step 2: Create BoW vector for each document\
bow_vectors = \[\]\
for doc in documents:\
vector = \[\]\
words = doc.lower().split()\
for word in vocab:\
vector.append(words.count(word))\
bow_vectors.append(vector)

\# Display\
print(\"Vocabulary:\", vocab)\
print(\"Bag of Words Vectors:\")\
for vec in bow_vectors:\
print(vec)

**✅ Output:**

 

Vocabulary: \[\'i\', \'love\', \'nlp\', \'python\'\]\
Bag of Words Vectors:\
\[1, 1, 1, 0\]\
\[1, 1, 0, 1\]

 

**📌 Using Scikit-learn's CountVectorizer (real-world)**

 

from sklearn.feature_extraction.text import CountVectorizer

\# Sample documents\
documents = \[\
\"I love NLP\",\
\"I love Python\"\
\]

\# Create a CountVectorizer instance\
vectorizer = CountVectorizer()

\# Fit and transform documents into BoW vectors\
bow_matrix = vectorizer.fit_transform(documents)

\# Get the vocabulary\
print(\"Vocabulary:\", vectorizer.get_feature_names_out())

\# Convert to array and display\
print(\"Bag of Words Matrix:\\n\", bow_matrix.toarray())

**✅ Output:**

 

Vocabulary: \[\'love\' \'nlp\' \'python\'\]\
Bag of Words Matrix:\
\[\[1 1 0\]\
\[1 0 1\]\]

(Note: CountVectorizer removes single-character words like 'I' by
default unless you change its parameters.)

 

 

TF-IDF (Term Frequency - Inverse Document Frequency)

 

**📖 What is TF-IDF?**

**TF-IDF** is a numerical statistic that reflects how important a word
is to a document **in a collection or corpus**.

It's one of the most popular techniques for **text feature extraction**
in NLP.\
\
 

👉 **TF-IDF** is a method to figure out which words in a document are
**important**.

Not all words are equally useful.

Words like *\"the\"*, *\"is\"*, *\"I\"*, *\"and\"* appear everywhere ---
they don't tell us much about a document's meaning.

**TF-IDF helps highlight words that are frequent in a particular
document but rare in other documents.\**
 

 

**📦 Example to Understand:**

Imagine we have 3 documents:

- **Doc 1:** *I love NLP*

- **Doc 2:** *I love Python*

- **Doc 3:** *Python is great*

 

**📌 Step 1: Term Frequency (TF)**

How many times a word appears in a document.

Example for **Doc 1: \"I love NLP\"**

- 'I' appears 1 time

- 'love' appears 1 time

- 'NLP' appears 1 time

If total words = 3, then:

TF(word)=number of times word appears in document​ / total no.of words in
the document\
 

So for 'I' in Doc 1:

TF(I)=1 / 3 = 0.33\
\
**📌 Step 2: Inverse Document Frequency (IDF)**

How **unique** or **rare** a word is across all documents.

If a word appears in **many documents**, its IDF is low.

If it appears in **few documents**, its IDF is high.

Formula:

IDF(word)=log(total documents/ no.of documents containg the word )

Example:

- Total documents = 3

- 'I' appears in 2 documents → IDF is lower

- 'NLP' appears in 1 document → IDF is higher

 

**📌 Step 3: Multiply TF × IDF**

This gives us **TF-IDF score** for each word in each document.

- **High score** = Word is frequent in this document and rare in other
  documents

- **Low score** = Word is either rare in this document or common in many
  documents

 

**📊 Summary Table (Illustration)**

  ------------------------------------------------
  **Word**   **Doc 1  **Doc 1 TF-IDF (if rare in
             TF**     others)**
  ---------- -------- ----------------------------
  I          0.33     Low (since it\'s common in
                      many docs)

  love       0.33     Low

  NLP        0.33     High (if it\'s rare
                      elsewhere)
  ------------------------------------------------

 

 

**📌 So What's The Point?**

👉 **TF-IDF helps us pick out the important words in each document and
ignore common, less useful words.**

For example:

- 'NLP' is important in **Doc 1**

- 'Python' is important in **Doc 2**

 

**📦 Quick Example in Code (With Printed Values)**

from sklearn.feature_extraction.text import TfidfVectorizer

docs = \[\"I love NLP\", \"I love Python\"\]

vectorizer = TfidfVectorizer()\
tfidf = vectorizer.fit_transform(docs)

print(\"Features (words):\", vectorizer.get_feature_names_out())\
print(\"TF-IDF Matrix:\\n\", tfidf.toarray())

**Output:**

 

Features (words): \[\'love\' \'nlp\' \'python\'\]\
TF-IDF Matrix:\
\[\[0.707 0.707 0. \]\
\[0.707 0. 0.707\]\]

Meaning:

- Doc 1: 'love' and 'NLP' are equally important

- Doc 2: 'love' and 'Python' are equally important

 

**📌 Final Thought:**

👉 **TF-IDF finds important words by combining:**

- **How often a word appears in a document (TF)**

- **How rare it is across all documents (IDF)**

This makes it much smarter than just counting words.

 

 

Word Embedding

z

 

**📌 Topic: Word Embedding --- Feature Extraction ✅**

**📖 What is it?**

**Word Embedding** is a technique to represent words in a continuous,
dense vector space where similar words have similar representations.

Unlike **Bag of Words**, which is sparse and context-free, word
embeddings capture **semantic relationships** between words.\
 

**📌 What is an Embedding in NLP?**

👉 **Embedding** is a **dense, numerical representation of data (like
words, sentences, or documents) in a lower-dimensional continuous vector
space**, where similar data points (like words with similar meaning)
have similar vector representations.

In short:

> 📌 It's a way to **convert text into numbers** --- but in a smart way
> that **captures meaning, context, and relationships** between words.\
>  
>
> **Embedding is a technique to map words (or other discrete data) into
> dense, meaningful numerical vectors in a continuous space where
> similar things are positioned closer together.**
>
>  

**📊 Why Use Word Embeddings?**

- It preserves the **meaning and context**.

- Words with similar meaning are mapped to nearby points in vector
  space.

- Reduces dimensionality while preserving useful information.

 

**📦 Popular Word Embedding Techniques:**

  ---------------------------------------------------------------------
  **Method**     **Description**
  -------------- ------------------------------------------------------
  **Word2Vec**   Predicts a word given surrounding words (CBOW) or vice
                 versa (Skip-gram)

  **GloVe**      Combines global co-occurrence and local context

  **FastText**   Improves Word2Vec by including subword (character
                 n-gram) information
  ---------------------------------------------------------------------

 

 

**📈 How it Works (Intuition):**

- Each word is assigned a vector of fixed size (say 100, 300
  dimensions).

- During training, the model learns to adjust these vectors such that
  **similar words** are **closer together** in vector space.

 

**📌 Example:**

Words like:

- *King → \[0.2, 0.7, -0.4, ...\]*

- *Queen → \[0.3, 0.8, -0.5, ...\]*

- *Man → \[0.1, 0.6, -0.3, ...\]*

- *Woman → \[0.25, 0.75, -0.45, ...\]*

And mathematically:

**King - Man + Woman ≈ Queen\**
You can see that **'King' is close to 'Queen'**, and **'Man' is close to
'Woman'** in this space.

This famous relationship was learned using **Word2Vec**.

 

**📌 Implementation Example (Word2Vec using Gensim):**

 

from gensim.models import Word2Vec

\# Sample sentences\
sentences = \[\
\[\'I\', \'love\', \'NLP\'\],\
\[\'I\', \'love\', \'Python\'\]\
\]

\# Train a Word2Vec model\
model = Word2Vec(sentences, vector_size=50, window=3, min_count=1, sg=1)

\# Get vector for a word\
print(model.wv\[\'Python\'\])

✅ It gives a 50-dimensional vector for \'Python\' that captures its
meaning relative to other words.

 

**📌 Summary:**

  -------------------------------------------------------
  **Bag of Words**       **Word Embedding**
  ---------------------- --------------------------------
  Sparse,                Dense, low-dimensional
  high-dimensional       

  No context, meaning    Captures meaning & relationships
  ignored                

  Simple, fast           Computationally heavier but more
                         powerful
  -------------------------------------------------------

 

 

Word2Vec

 

**📌 What is Word2Vec? (Plain & Clear Explanation)**

👉 **Word2Vec** is a technique to convert words into numbers (called
**word embeddings**) in such a way that **similar words have similar
numbers (vectors)**.

It was developed by Google in 2013.

**📦 Why Word2Vec?**

Previously, models used **one-hot encoding** or **count-based methods**
(like Bag of Words), but they:

- Didn't capture meaning or relationships between words

- Produced sparse, high-dimensional data

**Word2Vec solves this by learning word meanings based on context.**

 

**📖 How Does It Work?**

It uses a **neural network model** (a very tiny one) to learn from a
large collection of text and create dense, meaningful word vectors.

**📌 Two main approaches in Word2Vec:**

  ----------------------------------------------------------------------
  **Model Name**        **What it Does**                **Example**
  --------------------- ------------------------------- ----------------
  **CBOW (Continuous    Predicts a word based on        \"The cat \_\_\_
  Bag of Words)**       surrounding words               on the mat\"

  **Skip-gram**         Predicts surrounding words      \"cat\" →
                        based on the current word       predicts
                                                        \"The\", \"on\"
  ----------------------------------------------------------------------

 

 

**📊 Simple Example\
Sentence:** *I love natural language processing*

**CBOW:**

- Input: \[\"I\", \"natural\", \"language\", \"processing\"\]

- Predict: \"love\"

**Skip-gram:**

- Input: \"love\"

- Predicts: \"I\", \"natural\", \"language\", \"processing\"

 

**📦 What's Special About Word2Vec Vectors?**

👉 Words with similar meanings have similar vectors (they're close
together in the vector space).

**Example:**

  ---------------------------
  **Word**   **Vector
             (simplified)**
  ---------- ----------------
  king       \[0.52, 0.65,
             0.12, \...\]

  queen      \[0.51, 0.64,
             0.11, \...\]

  apple      \[-0.23, 0.55,
             0.99, \...\]
  ---------------------------

 

- **king** and **queen** will be closer to each other than to **apple**.

 

**📊 Famous Vector Arithmetic Example**

**king - man + woman ≈ queen**

Meaning:

- Take the vector of **king**

- Subtract the vector of **man**

- Add the vector of **woman**

- Result: A vector very close to **queen**

 

**📌 Word2Vec Implementation Example (using Gensim)**

 

from gensim.models import Word2Vec

\# Example sentences\
sentences = \[\
\[\'I\', \'love\', \'NLP\'\],\
\[\'I\', \'love\', \'Python\'\],\
\[\'Python\', \'is\', \'awesome\'\]\
\]

\# Train Word2Vec model\
model = Word2Vec(sentences, vector_size=50, window=3, min_count=1, sg=1)

\# Get vector for a word\
print(model.wv\[\'Python\'\])

\# Find most similar words\
print(model.wv.most_similar(\'Python\'))

✅ **Output:**

- A 50-dimensional vector for \'Python\'

- A list of words most similar to \'Python\'

 

**📌 Final Summary (1 Line)**

👉 **Word2Vec learns word meanings by looking at surrounding words and
converts words into vectors where similar words are close together.\
\**
 

**📌 📊 Simple Example Breakdown**

**Sentence:**

👉 *I love natural language processing*

Let's say we use a **window size of 2** --- which means, for every word,
we consider the **two words before and after it** as context.

 

**🔵 CBOW (Continuous Bag of Words)**

**Goal:**

**Predict the current word based on its surrounding context words.**

**Example:**

If our target word is **\"love\"**

And window size is 2 --- the context words around **\"love\"** are:

- \"I\" (before)

- \"natural\" (after)

- (if more words around, would take them too --- up to 2 words before
  and after)

**So CBOW takes these context words:**

 

\[\"I\", \"natural\", \"language\", \"processing\"\]

and tries to **predict the middle word: \"love\"**

 

**🔴 Skip-gram**

**Goal:**

**Predict the surrounding context words based on the current word.**

**Example:**

If the current word is **\"love\"**

And window size is 2 --- the context words it needs to predict are:

- \"I\"

- \"natural\"

- (and if there were more on either side)

**So Skip-gram takes the current word \"love\"**

and predicts the words around it:

 

\"I\", \"natural\", \"language\", \"processing\"

 

**📌 How It's Done Internally**

Under the hood:

- Both CBOW and Skip-gram pass words through a tiny neural network.

- The network has:

  - An **input layer** (one-hot encoded word(s))

  - A **hidden layer** (embedding layer --- this is what we want to
    learn)

  - An **output layer** (one-hot encoded prediction)

**Training happens by adjusting the word vector weights so that context
and target predictions improve.**

 

**📊 Visual Representation**

**CBOW**

 

Context words ➝ \[ I, natural, language, processing \]\
\|\
Neural Network\
\|\
Predicts: \"love\"

**Skip-gram**

 

Input word: \"love\"\
\|\
Neural Network\
\|\
Predicts: \[ I, natural, language, processing \]

 

**📌 Summary:**

  ------------------------------------------------------------
  **CBOW**                       **Skip-gram**
  ------------------------------ -----------------------------
  Predicts word based on context Predicts context words based
                                 on a word

  Faster to train, good for      Better for rare words
  frequent words                 

  Input: multiple context words  Input: single word

  Output: one target word        Output: multiple context
                                 words
  ------------------------------------------------------------

 

 

FastText

 

**📌 What is FastText? (In Plain English)**

👉 FastText is a way to convert words into numbers (called word
embeddings) **but in a smarter way than Word2Vec**.

- **Word2Vec**: Each word gets one vector.

- **FastText**: It also looks at the **smaller parts inside a word
  (called subwords)** --- for example, parts like \"love\", \"lov\",
  \"ove\", \"ve\" in the word **love**.

This means:

- If it finds a new or misspelled word like **\"lovely\"** or
  **\"loove\"** it can still guess a good vector because it knows the
  parts **\"lov\"**, **\"ove\"** from other words.\
   

**📊 Simple Example:**

Imagine the word:

**love**

Now break it into small pieces (called character n-grams).

If we use 3-letter chunks:

- \<lo

- lov

- ove

- ve\>

**FastText learns a small vector for each of these pieces.**

Then it adds them together to get the final word vector for **love**.\
\
 

**📌 Why is this useful?**

✅ Handles **new words** it never saw before

✅ Deals better with **spelling mistakes**

✅ Works great in languages where words change a lot (like **Hindi**,
**Telugu**, or **French**)

Example:

- Word2Vec → doesn't know what to do with **\"loove\"**

- FastText → understands it's close to **\"love\"** because the parts
  inside it are familiar

 

**📦 FastText vs Word2Vec (Simple Table)**

  ------------------------------------------------------------------
  **📌 Feature**                       **Word2Vec**   **FastText**
  ------------------------------------ -------------- --------------
  Learns one vector for each word      ✅             ✅

  Breaks words into parts and learns   ❌             ✅
  sub-parts too                                       

  Can handle new, unseen words         ❌             ✅
  ------------------------------------------------------------------

 

 

**📌 Real-life Example**

Imagine teaching a kid the meaning of "love".

Now if the kid hears "loved" or "lovely", they can still guess it's
about something good because it starts with **"lov"** --- the part they
already know.

FastText does this for computers.

 

**📌 Simple FastText Code Example**

 

from gensim.models import FastText

\# Example sentences\
sentences = \[\
\[\'I\', \'love\', \'NLP\'\],\
\[\'I\', \'love\', \'Python\'\]\
\]

\# Train FastText model\
model = FastText(sentences, vector_size=4, window=3, min_count=1)

\# See vector for \'love\'\
print(model.wv\[\'love\'\])

\# See vector for a new word \'lovely\'\
print(model.wv\[\'lovely\'\]) \# Word2Vec would give an error, FastText
works

✅ Even though we never gave it **'lovely'**, it'll give a vector
because it can guess based on the **\'lov\'**, **\'ove\'** it learned
earlier.

 

**📌 Final Summary (1 Line)**

👉 **FastText converts words into numbers by breaking them into small
parts and learning those parts too --- so it can understand new or
unknown words better.\
\**
 

**📊 FastText vs Word2Vec --- Visual Diagram**

**🔵 Word2Vec**

 

Word: \"love\"\
↓\
Learns a single vector\
↓\
\[0.32, -0.44, 0.12, \...\]

**Problem:** If a new word like **\"lovely\"** appears --- it gives an
error ❌ because no vector was learned for \"lovely\".

 

**🟢 FastText**

 

Word: \"love\"\
↓\
Break into character n-grams (example with 3 letters)\
↓\
\<lo lov ove ve\>\
↓\
Learn a vector for each part\
↓\
\[0.1, 0.2, 0.3, \...\] + \[0.2, 0.1, 0.0, \...\] + \...\
↓\
Combine them (sum or average)\
↓\
Final vector for \"love\"\
↓\
\[0.32, -0.44, 0.12, \...\]

Now --- new word: \*\*\"lovely\"\*\*\
↓\
Break into:\
\<lo, lov, ove, vel, ely, ly\>\
↓\
Use known n-gram vectors to build a vector\
↓\
\[vector for \"lovely\"\]

✅ So FastText can handle unseen words like **\"lovely\"**,
**\"loove\"**, or **\"lovin\"** by breaking them into sub-parts and
combining known pieces.

 

 

GloVe**(Global Vectors for Word Representation)**

**What is GloVe?**

👉 **GloVe** is another popular technique to convert words into vectors
(word embeddings), just like **Word2Vec** --- **but the way it learns
those vectors is different.**

📌 Word2Vec learns from **local context windows** (nearby words).

📌 **GloVe learns from the overall statistics of how frequently words
co-occur together in a text corpus** --- so it uses **global
information** about word relationships.

**GloVe was developed by Stanford NLP Group in 2014.**

 

**📦 Why Do We Need GloVe?**

- Word2Vec learns word meanings based on *neighboring words*.

- But what if we could learn better word relationships by using **how
  often words appear together in the entire corpus**?

- GloVe combines the power of **count-based methods** (like Bag of
  Words/Co-occurrence Matrix) and **predictive models** (like Word2Vec).

 

**📖 How Does GloVe Work?**

1️⃣ First, it builds a **word co-occurrence matrix**.

👉 Example: how often *word i* appears near *word j* in a corpus.

Example:

  -----------------------------------------------
          **cat**   **dog**   **mat**   **sat**
  ------- --------- --------- --------- ---------
  cat     0         5         3         2

  dog     5         0         1         4

  mat     3         1         0         7

  sat     2         4         7         0
  -----------------------------------------------

 

Each number shows how often two words appeared close to each other in
the text.

 

2️⃣ Then, it tries to learn word vectors in such a way that:

👉 The **dot product** of two word vectors equals the **log of the
number of times they co-occur**.

**Meaning:**

If two words co-occur frequently → their vectors should be similar (dot
product high)

If they rarely co-occur → vectors should be distant (dot product low)

 

**📊 Famous GloVe Formula**

**[\
wiT​⋅wj​+bi​+bj​=log(Xij​)]{.underline}**\
\
Where:

- wi,wjw_i, w_jwi​,wj​ → word vectors of words *i* and *j*

- bi,bjb_i, b_jbi​,bj​ → biases

- XijX\_{ij}Xij​ → number of times word *i* and *j* appeared together

It optimizes this using a **cost function** over the entire matrix.

 

**📌 Special Qualities of GloVe**

✅ Combines the **global co-occurrence information**

✅ Works well for large corpora

✅ Famous for preserving word relationships like:

**king - man + woman ≈ queen**

 

**📦 Word2Vec vs GloVe**

  ----------------------------------------------------------------------
  **Feature**           **Word2Vec**         **GloVe**
  --------------------- -------------------- ---------------------------
  Learns from           Local context        Global co-occurrence
                        windows              statistics

  Training type         Predictive neural    Count-based matrix
                        network              factorization

  Example models        CBOW, Skip-gram      Co-occurrence matrix +
                                             optimization

  Performance on small  Usually better       Needs large corpus
  data                                       
  ----------------------------------------------------------------------

 

 

**📌 Simple GloVe Example**

We usually use pre-trained GloVe vectors (like **glove.6B.300d.txt**)
which you can load into your models.

**Example using gensim to load pre-trained GloVe:**

 

from gensim.models.keyedvectors import KeyedVectors

\# Load GloVe vectors in word2vec format\
glove_file = \'glove.6B.100d.word2vec.txt\'\
model = KeyedVectors.load_word2vec_format(glove_file, binary=False)

\# Get vector for a word\
print(model\[\'king\'\])

\# Find most similar words\
print(model.most_similar(\'king\'))

 

**📌 Final 1-Line Summary**

👉 **GloVe learns word vectors by using the global word co-occurrence
counts in a corpus --- so similar words get similar vectors based on how
often they appear together.**
