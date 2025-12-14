Word level processing

 

**📖 Word Level Processing in NLU**

In **Natural Language Understanding (NLU)**, **word-level processing**
focuses on breaking down and analyzing individual words within a text to
understand their meaning and role. It's a crucial step before moving on
to sentence-level or document-level analysis.

**📌 Key Components:**

1.  **Tokenization**

    - **Splitting text into individual words called tokens.**

    - **Example: \"I love NLP\" → \[\"I\", \"love\", \"NLP\"\]**

2.  **Normalization**

    - **Standardizing words to a common format.**

    - **Includes:**

      - **Lowercasing: Hello → hello**

      - **Removing punctuation/numbers: 42 → \`\`**

      - **Stemming: playing, played → play**

      - **Lemmatization: running → run (using grammar rules)**

3.  **Stopword Removal**

    - **Removing common, less informative words.**

    - **Example: \"I have a car\" → \[\"car\"\] (if I, have, a are
      stopwords)**

4.  **Word Embeddings / Vectorization**

    - **Converting words into numerical form for machines.**

    - **Examples: Bag of Words (BoW), TF-IDF, Word2Vec, GloVe**

 

**📌 Why it matters:**

- Helps NLU models focus on meaningful content.

- Reduces noise.

- Converts unstructured text into structured, machine-readable formats.

 

**📝 Practice Exercise:**

Given the sentence:

**\"Cats are running faster than dogs!\"**

👉 Perform the following word-level processing steps manually:

1.  Tokenize the sentence.

2.  Lowercase all tokens.

3.  Remove stopwords (assume stopwords = are, than)

4.  Apply stemming (simplify running to run)\
    \
    \
     

> **Given sentence:**
>
> *\"Cats are running faster than dogs!\"*
>
> **Your processing:**
>
> 1️⃣ **Tokenization** ✅
>
> \[\"Cats\", \"are\", \"running\", \"faster\", \"than\", \"dogs!\"\]
> --- Correct!
>
> 2️⃣ **Lowercasing** ✅
>
> \[\"cats\", \"are\", \"running\", \"faster\", \"than\", \"dogs!\"\]
> --- Perfect!
>
> 3️⃣ **Stopword Removal** ✅
>
> Stopwords: are, than
>
> Result: \[\"cats\", \"running\", \"faster\", \"dogs!\"\] --- Great!
>
> 4️⃣ **Stemming** ✅

- running → run

- faster → fast

> Final result:
>
> \[\"cats\", \"run\", \"fast\", \"dogs!\"\]
>
>  
>
> 💡 **Optional Cleanup:**
>
> You might also consider removing punctuation at this stage:

- dogs! → dogs

> So clean final tokens:
>
> \[\"cats\", \"run\", \"fast\", \"dogs\"\]

 

 

Phrase level processing

 

**📖 Phrase Level Processing in NLU**

Once individual words are processed, the next step is to understand
**groups of words (phrases)** that act as meaningful units within a
sentence. This is essential because meaning often depends on word
combinations, not isolated words.

 

**📌 Key Components:**

1.  **Chunking (Shallow Parsing)**

    - **Identifies and groups words into phrases like:**

      - **Noun Phrases (NP): *\"the beautiful garden\"***

      - **Verb Phrases (VP): *\"is running fast\"***

      - **Prepositional Phrases (PP): *\"in the park\"***

    - **Example:\
      *\"The little boy is playing in the park.\"\*
      → \[NP The little boy\] \[VP is playing\] \[PP in the park\]**

2.  **Part-of-Speech (POS) Tagging**

    - **Tags each word with its grammatical category:**

      - **Noun (NN)**

      - **Verb (VB)**

      - **Adjective (JJ)**

      - **Preposition (IN)**

    - **Essential for identifying phrase boundaries.**

3.  **Collocation Detection**

    - **Detects commonly occurring word pairs/groups.**

    - **Example: *\"fast food\"*, *\"strong tea\"*, *\"make up\"***

 

**📌 Why it matters:**

- Phrases often carry more precise meaning than single words.

- Helps in syntactic parsing, sentiment analysis, question answering,
  and more.

- Reduces ambiguity by treating related words as a unit.

 

**📝 Practice Exercise:**

Given this sentence:

**\"The old man walked through the dark forest.\"**

👉 Perform these steps:

1.  **POS Tagging** (you can write tags like Noun, Verb, Adjective)

<!-- -->

2.  Identify:

    - Noun Phrases (NP)

    - Verb Phrases (VP)

    - Prepositional Phrases (PP)\
      \
       

> **Given sentence:**
>
> *\"The old man walked through the dark forest.\"*
>
> **✅ 1️⃣ POS Tagging (with common tags)**
>
> Let's assign POS tags for all words:

  ------------------------
  **Word**   **POS Tag**
  ---------- -------------
  The        Determiner
             (DT)

  old        Adjective
             (JJ)

  man        Noun (NN)

  walked     Verb (VBD)

  through    Preposition
             (IN)

  the        Determiner
             (DT)

  dark       Adjective
             (JJ)

  forest     Noun (NN)
  ------------------------

>  
>
>  
>
> **✅ 2️⃣ Phrase Identification**
>
> Now, based on POS tags:

- **\[NP The old man\]** --- Correct ✔️

- **\[VP walked\]** --- Ideally just the verb here ✔️

- **\[PP through the dark forest\]** --- The entire prepositional phrase
  starts with *through* and includes the noun phrase it points to.

> Within the PP:

- **\[NP the dark forest\]** --- another nested noun phrase.

> So cleanly:
>
>  
>
> \[NP The old man\] \[VP walked\] \[PP through \[NP the dark forest\]\]

 

 

Sentence level processing

 

**📖 Sentence Level Processing in NLU**

After processing words and phrases, the next step is to understand
**entire sentences**. This involves analyzing sentence structure and
meaning to determine the relationship between phrases, and to interpret
the sentence as a whole.

 

**📌 Key Components:**

1.  **Syntactic Parsing**

    - **Analyzes sentence grammar using parse trees or dependency
      parsing.**

    - **Determines how words and phrases relate structurally.**

    - **Example:\
      *\"The cat chased the mouse.\"\*
      → Subject: *The cat*, Verb: *chased*, Object: *the mouse***

2.  **Semantic Role Labeling (SRL)**

    - **Identifies roles played by sentence components:**

      - **Who did what to whom, when, where, how.**

    - **Example:\
      *\"John gave Mary a book.\"\*
      → Agent: *John*, Action: *gave*, Recipient: *Mary*, Theme:
      *book***

3.  **Sentence Boundary Detection**

    - **Recognizing where one sentence ends and another begins.**

    - **Important for parsing multi-sentence texts or documents.**

4.  **Sentiment and Intention Analysis**

    - **At sentence level: determining if a sentence expresses positive,
      negative, or neutral sentiment.**

    - **Example: *\"I love this book!\"* → Positive**

5.  **Sentence Type Identification**

    - **Classifying sentences as:**

      - **Declarative (statement)**

      - **Interrogative (question)**

      - **Imperative (command)**

      - **Exclamatory (strong emotion)**

 

**📌 Why it matters:**

- Gives context to how phrases and words work together.

- Enables tasks like question answering, summarization, and translation.

- Helps in intent recognition in applications like chatbots.

 

**📝 Practice Exercise:**

Given this sentence:

**\"Emma quickly gave her friend a thoughtful gift yesterday.\"**

👉 Perform these steps:

1.  Identify:

    - Subject

    - Verb

    - Object(s)

    - Adverbial phrases (if any)

2.  Label semantic roles:

    - Who did what to whom, when, how\
      \
      \
       

> **Given Sentence:**
>
> *\"Emma quickly gave her friend a thoughtful gift yesterday.\"*
>
> **✅ 1️⃣ Structural Identification**

  ------------------------------------------
  **Element**      **Value**
  ---------------- -------------------------
  **Subject**      Emma

  **Verb**         gave

  **Direct         a thoughtful gift
  Object**         

  **Indirect       her friend
  Object**         

  **Adverbial      quickly (manner),
  Modifiers**      yesterday (time)
  ------------------------------------------

>  
>
> 📌 *Note:*

- **"thoughtful"** is an adjective modifying *gift*, not an adverbial
  phrase.

- **"quickly"** is an adverb modifying *gave*.

- **"yesterday"** is an adverbial phrase of time.

>  
>
> **✅ 2️⃣ Semantic Role Labeling**

  -----------------------------
  **Role**        **Element**
  --------------- -------------
  **Agent**       Emma

  **Action**      gave

  **Recipient**   her friend

  **Theme**       a thoughtful
                  gift

  **Manner**      quickly

  **Time**        yesterday
  -----------------------------

>  
>
> Perfectly captured the core semantic roles --- well done!

 

 

Discourse level processing

 

**📖 Discourse Level Processing in NLU**

While sentence-level processing helps understand individual sentences,
**Discourse Level Processing** looks at **how multiple sentences relate
to one another to form coherent meaning in a conversation, paragraph, or
document**.

It focuses on context, continuity, reference resolution, and coherence
across larger text structures.

 

**📌 Key Components:**

1.  **Anaphora and Coreference Resolution**

    - **Identifying when different words refer to the same entity.**

    - **Example:\
      *\"Neelima loves coding. She practices every day.\"\*
      → *She* refers to *Neelima***

2.  **Discourse Relations**

    - **Understanding logical and semantic connections between
      sentences:**

      - **Cause-Effect: *\"It rained, so the match was cancelled.\"***

      - **Contrast: *\"I love tea, but I prefer coffee.\"***

      - **Temporal: *\"After finishing his work, he went home.\"***

3.  **Topic Segmentation**

    - **Dividing text into sections based on topic shifts or changes in
      focus.**

4.  **Dialogue Act Recognition**

    - **Classifying the function of each utterance:**

      - **Question, Answer, Greeting, Command, Apology, etc.**

 

**📌 Why it matters:**

- Ensures meaning isn't lost across multiple sentences.

- Critical for tasks like:

  - Summarization

  - Question answering

  - Chatbots and dialogue systems

  - Document classification

- Resolves ambiguities and references accurately.

 

**📝 Reflective Practice:**

Consider this mini text:

**\"Ravi bought a new car yesterday. He drove it to his office today.
The car was faster than he expected.\"**

👉 Identify:

1.  **Anaphora/Coreference Links**

2.  **Discourse Relations** between sentences\
    \
    \
    ** **

> **✅ Your Answers**
>
> **Given Text:**
>
> *\"Ravi bought a new car yesterday. He drove it to his office today.
> The car was faster than he expected.\"*
>
> **1️⃣ Anaphora/Coreference Resolution**

- **He** → *Ravi* ✅

- **It** → *the new car* ✅

- **The car** (in third sentence) → *the new car* ✅

- **He** (again) → *Ravi* ✅

> You caught the main one perfectly. ✅
>
>  
>
> **2️⃣ Discourse Relations**

- **Between Sentence 1 and 2:\
  Temporal Relation** → Action *drove* happens after *buying* ✅

- **Between Sentence 2 and 3:\
  Elaboration/Explanation** → Sentence 3 describes Ravi's experience
  with *the car he drove today*.

 

 

Semantic Role Labeling

 

**📖 Semantic Level Processing in NLU**

**Semantic Processing** deals with understanding the **meaning of words,
phrases, and sentences** in context.

While word, phrase, and sentence-level processing handle structure and
grammar, **semantic processing ensures that the system interprets the
actual intended meaning**.

 

**📌 Key Components:**

1.  **Word Sense Disambiguation (WSD)**

    - **Determining which meaning of a word is used in a given
      context.**

    - **Example:\
      *\"I went to the bank to withdraw money.\"\*
      → *bank* as a financial institution, not a river bank.**

2.  **Named Entity Recognition (NER)**

    - **Identifying and classifying proper names in text:**

      - **Person: *"Neelima"***

      - **Organization: *"OpenAI"***

      - **Location: *"Hyderabad"***

      - **Date: *"June 20"***

3.  **Semantic Role Labeling (SRL)**

    - **Assigning roles like who did what to whom, when, where (as we
      touched earlier in sentence-level processing)**

4.  **Meaning Representation**

    - **Structuring meaning in a form machines can reason with:**

      - **First-Order Logic**

      - **Semantic Networks**

      - **Conceptual Graphs**

5.  **Ontology-based Reasoning**

    - **Using structured knowledge bases (like WordNet, ConceptNet) to
      understand relationships:**

      - ***"A dog is a type of animal."***

      - ***"A cat is a pet."***

**Key Semantic Roles:**

- **Agent (Doer)** --- The entity performing the action

- **Action (Predicate)** --- The verb or main event

- **Theme (Patient)** --- The entity affected by the action

- **Recipient** --- The entity receiving something

- **Instrument** --- Means by which the action is performed

- **Location** --- Where the action happens

- **Time** --- When it happens

- **Manner** --- How it happens

 

**📌 Why it matters:**

- Ensures a system doesn\'t just \"parse\" text but **understands actual
  meaning**.

- Prevents misinterpretations, crucial for:

  - Chatbots

  - Virtual assistants

  - Machine translation

  - Question answering systems

 

**📝 Quick Practice:**

For each sentence, identify **the intended meaning of the ambiguous
word**:

1.  *\"The bat flew out of the cave.\"*

2.  *\"He swung the bat and hit a six.\"*

👉 What does *\"bat\"* mean in each case?\
\
 

Let's confirm:

  ---------------------------------------------------------------------
  **Sentence**         **Meaning of *bat***
  -------------------- ------------------------------------------------
  *\"The bat flew out  A flying nocturnal animal (mammal, not bird
  of the cave.\"*      technically --- but you got the idea right!)

  *\"He swung the bat  A wooden stick used in cricket/baseball
  and hit a six.\"*    
  ---------------------------------------------------------------------

 

✅ **Correct on both counts!**

Excellent application of **Word Sense Disambiguation**, which is a key
part of **Semantic Level Processing**.

 

 

Text summarization

 

**📖 What is Text Summarization in NLP?**

**Text Summarization** is the process of automatically generating a
shortened version of a text while preserving its key information and
meaning.

It helps in quickly understanding the essence of large documents, news
articles, research papers, or conversation logs.

 

**📌 Types of Text Summarization:**

**1️⃣ Extractive Summarization**

- Selects important sentences, phrases, or paragraphs directly from the
  original text.

- Simply picks and combines key parts without changing wording.

- Example:

  - Original: *\"Neelima loves coding. She practices Python daily. Her
    favorite topics are NLP and AI.\"*

  - Summary: *\"Neelima loves coding. Her favorite topics are NLP and
    AI.\"*

**2️⃣ Abstractive Summarization**

- Generates new sentences that paraphrase the meaning of the original
  text.

- It uses deep learning and language generation techniques.

- Example:

  - Original: *\"Neelima loves coding. She practices Python daily. Her
    favorite topics are NLP and AI.\"*

  - Summary: *\"Neelima is passionate about coding, especially in NLP
    and AI.\"*

 

**📌 Popular Techniques:**

- **Extractive**

  - TextRank (graph-based)

  - LexRank

  - Frequency-based sentence scoring

- **Abstractive**

  - Seq2Seq models (Encoder-Decoder)

  - Transformer-based models like **BERTSUM**, **T5**, **PEGASUS**

 

**📌 Why it matters:**

- Saves time by quickly conveying essential information.

- Used in:

  - News summarization

  - Research paper abstracts

  - Meeting minute generation

  - Customer review analysis

  - Social media post summarization

 

**📝 Quick Practice:**

👉 Given this short text:

*\"Artificial Intelligence is transforming industries. From healthcare
to finance, AI systems are improving efficiency and decision-making.
Experts believe AI will continue to shape the future in remarkable
ways.\"*

Write:

1.  **An Extractive Summary**

2.  **An Abstractive Summary\
    \
    \
     **

> **✅ 1️⃣ Extractive Summary**
>
> You wrote:
>
> *\"Artificial Intelligence is transforming industries. Experts believe
> AI will continue to shape the future in remarkable ways.\"*
>
> ✔️ **Perfect extractive summary** --- you've picked two of the most
> informative sentences directly from the text without changing them.
> Well done!
>
>  
>
> **✅ 2️⃣ Now --- for the Abstractive Summary\**
> \"AI is revolutionizing various industries by enhancing efficiency and
> decision-making, and experts predict it will have a significant impact
> on the future.\"
