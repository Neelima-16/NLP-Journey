Text classification

 

**📖 What is Text Classification?**

Text classification is the process of assigning predefined categories or
labels to text data based on its content.

In **Sentiment Analysis**, it typically involves classifying a given
text into sentiment categories like **positive**, **negative**, or
**neutral**.

For example:

- \"I love this movie!\" → **Positive**

- \"This product is terrible.\" → **Negative**

 

**📌 Why is Text Classification Important in Sentiment Analysis?**

Because sentiment analysis relies on identifying the *emotional tone*
behind a body of text --- and classification is how we map that tone
into specific, interpretable categories.

 

**📊 How Does Text Classification Work?**

It usually follows these steps:

1.  **Data Collection** → gather text data (product reviews, tweets,
    etc.)

2.  **Text Preprocessing** → clean, tokenize, and vectorize the text

3.  **Feature Extraction** → convert text to numerical features (e.g.,
    using Bag of Words, TF-IDF, or word embeddings)

4.  **Model Training** → train a classification model (like Logistic
    Regression, Naive Bayes, SVM, or Neural Networks)

5.  **Prediction & Evaluation** → predict sentiment category and measure
    accuracy\
    \
    ** **

 

**📖 How It's Done (Simple Steps)**

1️⃣ **Collect Text Data**

Example:

\"I love this\" → Positive

\"I hate this\" → Negative

2️⃣ **Convert Text to Numbers**

A computer understands only numbers.

We convert words to numbers using techniques like **TF-IDF** or **word
count**.

Example:

\"I love this\" → \[0.3, 0.7, 0.5\]

3️⃣ **Give It to a Machine Learning Model**

We use a program (called a model) that learns from examples --- like:

- **Logistic Regression**

- **Naive Bayes**

These models learn from past examples like:

- \"Good movie\" → Positive

- \"Bad movie\" → Negative

4️⃣ **Test It**

Once the model learns, you can give it a new sentence like:

- \"Fantastic product!\"

And it will predict: **Positive**

 

**🛠️ Common Text Classification Models for Sentiment Analysis:**

- **Logistic Regression**

- **Naive Bayes Classifier**

- **Support Vector Machines (SVM)**

- **Random Forest**

- **Neural Networks (LSTM, CNN, BERT)**

**✅ Summary:**

- Text Classification assigns labels to text

- In Sentiment Analysis → label = sentiment category

- Involves preprocessing, feature extraction, model training

- Common algorithms: Logistic Regression, Naive Bayes, SVM, Neural
  Networks

 

 

 

**Aspect-Based Sentiment Analysis**

 

**📖 What is Aspect-Based Sentiment Analysis?**

👉 In regular sentiment analysis, we classify the *overall sentiment* of
a sentence or review as **positive**, **negative**, or **neutral**.

👉 But sometimes, one sentence talks about multiple things (aspects),
each with its own sentiment.

**Aspect-Based Sentiment Analysis (ABSA)** identifies:

1.  **What is being talked about** (called an **aspect**)

2.  **What is the sentiment towards each aspect**

 

**📦 Example:**

👉 **Sentence:**

**\"The battery life is great, but the screen is dull.\"**

- Aspect 1: **battery life** → Positive

- Aspect 2: **screen** → Negative

A normal sentiment analysis system might just say **neutral or mixed**
--- but **ABSA breaks it down for each aspect**.

 

**📌 Why Do We Need ABSA?**

In product reviews, customer feedback, and restaurant reviews --- people
often talk about multiple parts of a product/service in one sentence.

**ABSA helps companies understand exactly what people like or dislike.**

Example:

- In a restaurant review:

  - **Food**: Positive

  - **Service**: Negative

This allows businesses to improve specific areas.

 

**📖 How Does ABSA Work?**

1️⃣ **Aspect Extraction**

→ Find the aspects (topics) being talked about.

2️⃣ **Sentiment Classification for Each Aspect**

→ Find if the sentiment for each aspect is **positive**, **negative**,
or **neutral**.

 

**📊 Techniques Used in ABSA**

- **Rule-based methods**: Using sentiment lexicons (like positive and
  negative word lists)

- **Machine Learning models**: Logistic Regression, Naive Bayes

- **Deep Learning models**: LSTM, BERT (pretrained transformers)

 

**📦 Example Output:**

**Input:**

*\"The camera is amazing, but the battery life is poor.\"*

**ABSA Output:**

- Aspect: **camera** → Positive

- Aspect: **battery life** → Negative

 

**✅ Simple Summary:**

  -----------------------------------------------------------------------
  **Task**        **Regular Sentiment    **ABSA**
                  Analysis**             
  --------------- ---------------------- --------------------------------
  What it does    Overall sentiment      Sentiment per aspect

  Example result  Positive               Battery: Good, Screen: Bad

  Helpful for     General mood tracking  Product improvement,
                                         fine-grained feedback
  -----------------------------------------------------------------------

 

 

Opinion Mining

 

**📖 What is Opinion Mining?**

👉 **Opinion Mining** is another name for **Sentiment Analysis** --- but
it focuses on identifying and extracting **opinions, emotions, and
attitudes** from text data.

**It answers:**

- What is someone's opinion?

- Is it positive, negative, or neutral?

- About what topic or aspect is this opinion given?

**Example:**

*\"The hotel room was clean but too small.\"*

- **Opinion about room cleanliness**: Positive

- **Opinion about room size**: Negative

 

**📌 How is it Different from Sentiment Analysis?**

- **Sentiment Analysis**: Mainly detects whether a text is positive,
  negative, or neutral.

- **Opinion Mining**: Goes deeper to identify the exact *opinion
  expressions*, *opinion holders* (who said it), and *opinion targets*
  (about what).

 

**📦 Example:**

**Sentence:**

*\"I absolutely love the camera on this phone.\"*

**Opinion Mining Result:**

- **Opinion Holder**: I

- **Opinion Target**: camera

- **Opinion Expression**: absolutely love

- **Polarity**: Positive

 

**📊 How Does Opinion Mining Work?**

1️⃣ **Text Preprocessing** → Clean text

2️⃣ **Opinion Expression Extraction** → Find words expressing opinions
(like *love*, *hate*, *good*, *bad*)

3️⃣ **Target Identification** → Find the subject or object about which
the opinion is expressed

4️⃣ **Sentiment Classification** → Identify whether the opinion is
positive, negative, or neutral

 

**📌 Techniques Used:**

- **Rule-based methods**

- **Machine Learning algorithms**

- **Deep Learning models (like BERT, RoBERTa)**

- **Lexicon-based approaches** (using predefined sentiment word
  dictionaries)

 

**✅ Simple Summary:**

  ----------------------------------------------------------
  **Concept**     **What it does**
  --------------- ------------------------------------------
  Sentiment       Classifies text as positive, negative, or
  Analysis        neutral

  Opinion Mining  Detects **who** expressed **what opinion**
                  about **which aspect**
  ----------------------------------------------------------

**📖 Why is Opinion Mining Important?**

It helps:

- Companies understand what people **like or dislike** about their
  products.

- Track public opinions on **social media**, **product reviews**, and
  **surveys**.

- Political analysis of public sentiment towards parties or candidates.

**📊 📌 Opinion Mining vs. Aspect-Based Sentiment Analysis (ABSA)**

  -----------------------------------------------------------------------------
  **🔍 Point**      **Opinion Mining**            **Aspect-Based Sentiment
                                                  Analysis (ABSA)**
  ----------------- ----------------------------- -----------------------------
  **What it does**  Identifies **opinions,        Focuses on finding **aspects
                    opinions holders (who), and   (features/topics)** and
                    targets (what)**              sentiment about each aspect

  **Focus**         Extracting opinion            Classifying sentiment for
                    expressions and their         multiple aspects within a
                    targets, and who said them    text

  **Example Task**  Find out that *\"I absolutely Classify *\"The camera is
                    love the camera\"* expresses  amazing but battery life is
                    a positive opinion about      poor\"* → Camera: Positive,
                    **camera** by **I**           Battery: Negative

  **Opinion Holder  ✅ Yes (Who is expressing the ❌ Usually no (it doesn't
  Detection**       opinion?)                     care who said it, just what
                                                  was said about what)

  **Granularity**   More focused on **opinion     Focused on **aspects and
                    expressions and sources**     their associated sentiments**

  **Use case        Track who thinks what about a Know how people feel about
  example**         political party or product    **each feature** of a product
                    feature                       or service

  **Relation**      Broader concept that may      Specific type of opinion
                    include ABSA as a part        mining, focusing on
                                                  per-aspect sentiment
                                                  classification
  -----------------------------------------------------------------------------

 

 

**📦 Quick Example:**

**Sentence:**

*\"I love the battery life but hate the camera on my new phone.\"*

- **Opinion Mining Result**:

  - Opinion Holder: **I**

  - Opinion Targets: **battery life**, **camera**

  - Opinion Expressions: **love**, **hate**

  - Polarity: **Positive** for battery life, **Negative** for camera

- **ABSA Result**:

  - Aspect: **battery life** → Positive

  - Aspect: **camera** → Negative

👉 See?

**Opinion Mining** also captures *who said it*, while **ABSA** is more
about *what is being said about which aspect*.

 

**✅ Simple Summary:**

- **Opinion Mining** = Who, what, and how (about an opinion)

- **ABSA** = What and how (per aspect sentiment)

 

 

 

Emotion Recognition

 

**📖 What is Emotion Recognition in NLP?**

👉 **Emotion Recognition** is the task of identifying the **specific
emotions** expressed in a piece of text.

Where basic **Sentiment Analysis** classifies text as **Positive,
Negative, or Neutral**,

**Emotion Recognition** goes deeper to detect emotions like:

- **Happy**

- **Angry**

- **Sad**

- **Fear**

- **Surprise**

- **Disgust**

- etc.

 

**📦 Example:**

**Text:**

*\"I am thrilled with my results!\"*

**Emotion Recognition Output:**

→ **Emotion**: **Joy**

 

**📊 How is Emotion Recognition Different from Regular Sentiment
Analysis?**

  -------------------------------------------------------------------
  **Sentiment Analysis**          **Emotion Recognition**
  ------------------------------- -----------------------------------
  Classifies into Positive,       Detects specific emotions like
  Negative, Neutral               Happy, Sad, Angry

  Coarse-grained (general mood)   Fine-grained (specific feelings)
  -------------------------------------------------------------------

 

 

**📌 Why is Emotion Recognition Important?**

- In **product reviews** → Know exactly what emotions customers feel.

- In **social media monitoring** → Track public mood and reactions.

- In **mental health apps** → Detect signs of stress, sadness, etc.

- In **chatbots and virtual assistants** → Respond empathetically.

 

**📖 How Does Emotion Recognition Work?**

1️⃣ **Text Preprocessing**

Clean and prepare text.

2️⃣ **Feature Extraction**

Convert words to numerical form (like TF-IDF, word embeddings).

3️⃣ **Model Training**

Use models like:

- **Naive Bayes**

- **Logistic Regression**

- **LSTM**

- **BERT**

Trained on labeled emotion datasets (like **Emotion Dataset by
CrowdFlower**, **GoEmotions by Google**)

4️⃣ **Emotion Prediction**

The model predicts the emotion label for new text.

 

**📦 Example Emotion Categories:**

  -----------------------------------
  **Emotion**   **Example words**
  ------------- ---------------------
  Joy           happy, thrilled,
                delighted

  Anger         furious, irritated,
                annoyed

  Sadness       upset, heartbroken,
                gloomy

  Fear          scared, anxious,
                frightened

  Surprise      amazed, shocked

  Disgust       repulsed, sickened
  -----------------------------------

 

 

**✅ Simple Summary:**

**Emotion Recognition** = Detect **which emotion** a person is
expressing in their words

**Sentiment Analysis** = Detect **whether the emotion is positive,
negative, or neutral**

 

**📦 Quick Example Output:**

**Input Text:**

*\"I\'m so scared about the results.\"*

**Sentiment Analysis Result:** Negative

**Emotion Recognition Result:** Fear

 

 

Weakly Supervised Learning

 

**📖 What is Weakly Supervised Learning?**

👉 In traditional **Supervised Learning**, you need a large set of
**labeled data** (where every example has a correct label like
Positive/Negative).

👉 But labeling data manually is **time-consuming, expensive, and not
always possible** --- especially for huge datasets.

👉 **Weakly Supervised Learning** is a technique where models are
trained using:

- **Partially labeled data**

- **Noisy labels** (some labels might be wrong)

- **Inaccurate supervision signals**

So, the model learns with **less reliable or incomplete supervision**
compared to fully labeled data.

 

**📦 Example:**

Imagine you have:

- 1000 product reviews

- But only 50 reviews are labeled Positive/Negative

**Weakly supervised methods** can use those 50, combined with some
clever tricks (like labeling based on keyword matches or heuristic
rules), to train a usable sentiment analysis model.

 

**📊 Why Use Weakly Supervised Learning?**

- **Reduces need for costly manual annotation**

- **Works with noisy labels or automatically generated labels**

- Helps when **labeled data is scarce**

 

**📌 How is it Done?**

Some popular weak supervision strategies:

1️⃣ **Using heuristic labeling functions**

→ Example: If a review contains *"love"*, label it Positive.

2️⃣ **Distant supervision**

→ Automatically assign labels by linking data to an external source.

For example: if a tweet contains 😊 → mark it Positive.

3️⃣ **Data programming**

→ Combine multiple weak labeling functions and resolve conflicts.

4️⃣ **Semi-supervised learning**

→ Use a small labeled dataset and a large unlabeled dataset. Train
initial model on labeled data, predict labels for the rest, and
iteratively improve.

 

**📖 Example Use Case in Sentiment Analysis:**

**You don't have labeled reviews** --- but:

- Reviews with emojis like 😢 → Negative

- Reviews with 😊 → Positive

Use these as noisy labels to train a weakly supervised model.

 

**✅ Simple Summary:**

  ------------------------------------------------------------
  **Supervised Learning**  **Weakly Supervised Learning**
  ------------------------ -----------------------------------
  Needs large, clean       Uses small, incomplete, or noisy
  labeled data             labeled data

  Accurate, but costly to  Cheaper, faster, handles massive
  prepare                  unlabeled data
  ------------------------------------------------------------

 

 

**📌 Why It Matters in NLP**

Most real-world NLP tasks --- like analyzing millions of social media
posts, customer reviews, or news articles --- lack enough labeled data.

**Weakly supervised methods make it possible to build models without
huge annotation budgets.**

 

 

Sentiment Lexicons

 

**📖 What are Sentiment Lexicons?**

👉 A **Sentiment Lexicon** is basically a **predefined list of words (or
phrases)** where each word is associated with a **sentiment value** ---
like **positive**, **negative**, or **neutral**.

You can think of it like a **dictionary of emotional words**.

 

**📦 Example:**

  --------------------------
  **Word**   **Sentiment**
  ---------- ---------------
  love       Positive

  happy      Positive

  terrible   Negative

  bad        Negative

  okay       Neutral
  --------------------------

 

 

**📌 How is it Used in Sentiment Analysis?**

When analyzing a sentence, a program can:

1.  **Break the text into words**

2.  **Check each word in the lexicon**

3.  **Sum up or average the sentiment scores**

Example:

*\"The movie was fantastic and thrilling!\"*

- **fantastic** → Positive

- **thrilling** → Positive

**Overall Sentiment**: Positive

 

**📊 Why Use Sentiment Lexicons?**

✅ Simple to implement

✅ No need for labeled training data

✅ Good for rule-based and weakly supervised systems

✅ Useful for analyzing **small text pieces** like tweets, comments,
reviews

 

**📖 Popular Sentiment Lexicons in NLP:**

  ----------------------------------------------------------------------------------
  **Lexicon Name**   **Languages**   **Type**                      **Notes**
  ------------------ --------------- ----------------------------- -----------------
  **SentiWordNet**   English         Numeric polarity scores       WordNet-based

  **AFINN**          English         Numeric scores (-5 to +5)     Simple wordlist

  **VADER**          English         Positive, Negative, Neutral,  Great for social
                                     Compound                      media

  **LIWC (paid)**    Multiple        Categorized lexicon           Psychological
                                                                   insights
  ----------------------------------------------------------------------------------

 

 

**📌 How is it Different from Machine Learning Models?**

  -----------------------------------------
  **Lexicon-Based**     **Machine
                        Learning**
  --------------------- -------------------
  Uses predefined word  Learns from labeled
  lists                 data

  No training needed    Needs training data

  Easy to set up        Requires model
                        building
  -----------------------------------------

 

 

**✅ Simple Summary:**

- A **Sentiment Lexicon** is a ready-made list of emotional words and
  their sentiment.

- Helps quickly determine if a text is **positive**, **negative**, or
  **neutral** by checking word-by-word.
