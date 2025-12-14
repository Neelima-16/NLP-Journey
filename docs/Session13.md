Machine Translation

 

**📌 What is Machine Translation (MT)?**

**Machine Translation** is a subfield of **Natural Language Processing
(NLP)** where computers are programmed to automatically translate text
or speech from one language to another without human involvement.

A classic example you use every day: **Google Translate**.

 

 

Rule-Based Machine Translation

24 June 2025

11:28

 

**📌 What is Rule-Based Machine Translation (RBMT)?**

It's a type of **automatic translation system** where computers use
**grammar rules** and **word dictionaries** to translate sentences from
one language to another.

 

**📌 How does it work? (Imagine this like following a recipe)**

When you translate a sentence:

1.  **Look up each word** in a dictionary for its meaning in another
    language.

2.  **Apply grammar rules** to arrange the words properly in the new
    language's sentence format.

3.  **Combine them** into a correct sentence in the target language.

 

**📌 Example:**

Let's translate this:

**English:** I love you.

**To Hindi:** मैं तुमसे प्यार करता हूँ।

**Steps:**

- Word dictionary:

  - I → मैं

  - love → प्यार करना

  - you → तुमसे

- Grammar Rule:\
  In Hindi, the verb comes at the end.

**So final sentence:**

मैं तुमसे प्यार करता हूँ।

 

**📌 Key Points:**

- Uses **fixed rules and dictionaries**.

- Works well for **simple sentences**.

- **Hard to handle** jokes, emotions, idioms, and complex grammar.

- **Takes a lot of time** to write rules for every language pair.

 

**📌 Real-life Example:**

Old translation software like **Systran** (used by early Google
Translate) was rule-based.

 

 

Statistical Machine Translation

 

**📌 Statistical Machine Translation (SMT)**

Now that you know about Rule-Based systems, **Statistical Machine
Translation (SMT)** works differently --- it uses **math and
statistics** instead of hand-written rules.

 

**📖 What is SMT?**

**SMT is a machine translation method that learns how to translate by
analyzing large amounts of bilingual text data (called parallel
corpora).**

👉 It doesn\'t follow grammar rules but looks at how words and phrases
are usually translated based on probability.

 

**📌 How does SMT work?**

1.  **Collect a huge dataset** of the same text written in two
    languages.

2.  **Break it into words and phrases**.

<!-- -->

3.  Use **mathematical models to calculate the probability** of a phrase
    in one language being translated into a phrase in the other.

4.  Pick the translation that has the **highest probability**.

 

**📌 Example:**

If in your dataset:

- "Hello" → "नमस्ते" occurs **800 times**

- "Hello" → "सलाम" occurs **200 times**

The system will most likely translate **"Hello"** as **"नमस्ते"** because
it has a higher probability.

 

**📌 Types of SMT Models:**

- **Word-based SMT**: Translates one word at a time.

- **Phrase-based SMT**: Translates short phrases instead of individual
  words (better fluency).

- **Syntax-based SMT**: Uses some grammar structure for better accuracy.

 

**📌 Advantages:**

- Can automatically learn translations from data.

- Easier to build for many languages if large data is available.

**📌 Limitations:**

- Needs a huge amount of bilingual data.

- Sometimes produces awkward or grammatically incorrect sentences.

- Struggles with long-distance word dependencies.

 

**📌 Real-life Example:**

**Google Translate before 2016** was based on SMT.

 

 

Neural Machine Translation

 

**📖 What is Neural Machine Translation (NMT)?**

**Neural Machine Translation** is an advanced machine translation method
that uses **artificial neural networks** (deep learning models) to
translate entire sentences from one language to another.

It's the method used by today's best translation systems --- like
**Google Translate (after 2016)** and **DeepL**.

 

**📌 How does NMT work?**

Instead of translating one word or phrase at a time like SMT, NMT:

1.  **Reads the entire sentence at once**.

2.  **Understands the context and meaning**.

<!-- -->

3.  Translates it into the target language sentence in a fluent, natural
    way.

It uses a **deep neural network** model called **Encoder-Decoder** with
an **Attention Mechanism**.

 

**📌 Encoder-Decoder Architecture (in simple terms):**

- **Encoder**: Converts the input sentence into a numerical format
  (vector).

- **Attention Layer**: Focuses on important words during translation.

- **Decoder**: Uses this information to generate the translated
  sentence, word by word.

 

**📌 Example:**

**English:** \"I am learning Python.\"

**Hindi:** \"मैं पायथन सीख रहा हूँ।\"

NMT understands the **full sentence meaning** before translating,
ensuring better fluency and accuracy.

 

**📌 Why is NMT better?**

✅ Understands context and meaning

✅ Produces fluent, natural-sounding translations

✅ Learns from huge amounts of text data

 

**📌 Limitations:**

- Needs **very powerful computers** (GPUs) for training.

- Requires **a lot of data**.

- Sometimes makes small mistakes in rare or specialized phrases.

 

**📌 Real-life Example:**

- **Modern Google Translate** (since Nov 2016)

- **Microsoft Translator**

- **DeepL Translator**

 

**📌 Fun Fact:**

NMT models like **Transformers** (introduced by Google in 2017) made
Neural Machine Translation even faster and smarter. It became the
backbone for translation tools, chatbots, and NLP applications
worldwide.

 

 

Quality estimation

 

**📖 What is Quality Estimation (QE) in Machine Translation?**

**Quality Estimation (QE)** is a technique used to **predict the quality
of a machine-translated sentence without having a reference (human)
translation available**.

In simple words --- it answers:

👉 *"Is this translation good enough, or should it be revised?"*

Even when we don't have the correct translation to compare against.

 

**📌 Why is Quality Estimation Important?**

- **Saves time and cost** in translation projects.

- Helps decide whether:

  - A machine-translated sentence is fine as-is.

  - It needs light or heavy editing.

  - It should be re-translated by a human.

- Useful in **real-time translation apps**, customer support bots, and
  **localization industries**.

 

**📌 How does Quality Estimation work?**

- Uses **machine learning models** trained on examples of:

  - Good translations

  - Bad translations

  - And how humans rated them

- The model predicts a **quality score** for each translation.

 

**📌 Example:**

  ---------------------------------------------------
  **Source       **Machine         **Predicted
  Sentence**     Translation**     Quality Score**
  -------------- ----------------- ------------------
  \"I love       \"मैं कोडिंग करता    0.90 (Good)
  coding.\"      हूं।\"              

  \"Good         \"अच्छा सुबह!\"     0.30 (Bad)
  morning!\"                       
  ---------------------------------------------------

 

*Score is between 0 and 1 (1 being perfect translation)*

 

**📌 Types of Quality Estimation:**

1.  **Sentence-level QE**: Scores an entire sentence.

2.  **Word-level QE**: Marks good/bad words inside a translation.

3.  **Document-level QE**: Scores quality for the whole document.

 

**📌 How it's different from Evaluation Metrics (like BLEU, METEOR):**

- **Evaluation metrics need a reference (human) translation to compare
  with**.

- **Quality Estimation predicts quality without one**.

 

**📌 Applications:**

- **Translation agencies** to automate quality checks.

- **Post-editing workflows**.

- **AI-powered translation platforms** like Google Translate, Amazon
  Translate.

 

 

Machine Translation evaluation

 

**📖 What is Machine Translation Evaluation?**

**Machine Translation Evaluation** is the process of **measuring how
good or accurate a machine-translated sentence is compared to a correct,
human-translated sentence**.

 

**📌 Why is Evaluation Important?**

- To **compare different translation systems**.

- To **track the progress of translation models**.

- To know whether a translation is **good enough for production or needs
  improvement**.

 

**📌 Types of Machine Translation Evaluation:**

**🔹 1️⃣️⃣ Human Evaluation**

- **Humans manually check** translations and rate them based on:

  - Fluency (how natural it sounds)

  - Adequacy (how well the meaning is preserved)

- **Very accurate**, but **time-consuming and costly**.

**📌 Example:**

Rate translation on a scale of 1--5:

- 5 = Perfect translation

- 1 = Completely wrong

 

**🔹 2️⃣️⃣ Automatic Evaluation**

- Uses **mathematical formulas** to compare machine translations with
  human reference translations.

 

**📌 Common Automatic Evaluation Metrics:**

**📍 BLEU (Bilingual Evaluation Understudy)**

- Most popular automatic metric.

- Compares machine translation to one or more reference translations by
  matching overlapping **n-grams** (groups of consecutive words).

**Score range:** 0 to 1 (1 means perfect match)

 

**📍 METEOR**

- Considers synonyms and stemming (word roots) too.

- More language-friendly than BLEU.

 

**📍 TER (Translation Edit Rate)**

- Measures how many edits (insert, delete, replace, shift) are needed to
  turn a machine translation into the reference translation.

- Lower TER = better translation.

 

**📍 chrF**

- Works at the character level (good for languages with rich morphology
  like German, Finnish).

 

**📌 Example:**

  ---------------------------------------------------
  **Machine         **Reference        **BLEU Score**
  Translation**     Translation**      
  ----------------- ------------------ --------------
  \"I love          \"I love coding.\" 1.0 (perfect)
  coding.\"                            

  \"I like          \"I love coding.\" 0.4 (partial
  programming.\"                       match)
  ---------------------------------------------------

 

 

**📌 Summary:**

  --------------------------------------------------------------------
  **Type**        **Who Does    **Accuracy**   **Speed**   **Cost**
                  It?**                                    
  --------------- ------------- -------------- ----------- -----------
  Human           Humans        Very High      Slow        Expensive
  Evaluation                                               

  Automatic       Computer      High           Very Fast   Cheap
                  Tools                                    
  --------------------------------------------------------------------

 

 

Low Resource Machine translation

 

**📖 What is Low-Resource Machine Translation?**

**Low-Resource Machine Translation** refers to the challenge of building
good machine translation systems for **languages that have very little
bilingual training data available**.

 

**📌 Why is it a Problem?**

Modern machine translation systems like **Neural Machine Translation
(NMT)** require **huge amounts of parallel sentences** (same sentence in
two languages) to learn how to translate well.

But for many languages --- especially regional, tribal, or minority
languages --- such large bilingual datasets don't exist.

 

**📌 Example:**

- **English--French**: Huge data available → Easy to build MT system.

- **English--Telugu**, **English--Kannada** or **English--Tibetan**:
  Very little data → Hard to build.

 

**📌 How is Low-Resource Machine Translation handled?**

**🔹 1️⃣️⃣ Transfer Learning**

Use knowledge from a **high-resource language pair** (like
English-French) to improve translation in a **low-resource pair** (like
English-Telugu).

 

**🔹 2️⃣️⃣ Multilingual NMT**

Train a single model on multiple languages together.

The model can share patterns learned from rich languages to help
low-resource languages.

 

**🔹 3️⃣️⃣ Back-Translation**

Use existing MT systems to:

- Translate monolingual text in the target language into the source
  language.

- Create artificial bilingual data.

 

**🔹 4️⃣️⃣ Unsupervised MT**

Build translation systems **without any parallel data** --- using only
monolingual texts from both languages and clever alignment techniques.

 

**📌 Real-life Example:**

- **Google Translate** has made efforts to add translation for
  low-resource Indian languages like **Maithili, Dogri, Sanskrit, and
  Bhojpuri** using multilingual and transfer learning techniques.

 

**📌 Summary:**

  -----------------------------------------------------
  **Challenge**           **Solution**
  ----------------------- -----------------------------
  Very little parallel    Transfer learning,
  data                    multilingual models

  Poor translation        Back-translation,
  quality                 unsupervised MT

  Difficult grammar, rare Data augmentation, shared
  words                   embeddings
  -----------------------------------------------------

 
