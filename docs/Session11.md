**Natural Language Generation (NLG)**

 

**📌 What is Natural Language Generation (NLG)?**

**Natural Language Generation (NLG)** is a subfield of Natural Language
Processing (NLP) that focuses on **automatically producing human-like
text from structured or unstructured data**.

👉 In simpler terms:

- It takes some kind of **input data (numerical, text, structured, or
  unstructured)**

- And generates **coherent, meaningful, and natural-sounding human
  language** as output.

 

**📌 Where does NLG fit in NLP?**

While NLP as a whole deals with understanding and processing human
language, NLG is about **creating language**.

It's often seen as the last step in a typical NLP pipeline:

**Text Input → Text Processing → Understanding → Text Generation (NLG)**

 

**📌 Example of NLG in Everyday Use:**

  -----------------------------------------------------------------------
  **Scenario**     **Input**            **NLG Output**
  ---------------- -------------------- ---------------------------------
  Weather report   Weather data (temp,  "Today it's sunny with a high of
                   humidity, etc.)      30°C and clear skies."

  Chatbot response User's message       "Sure, I can help you with that.
                                        Could you provide more details?"

  Financial report Stock market data    "The market closed higher today,
  summary                               led by gains in tech stocks."
  -----------------------------------------------------------------------

 

 

**📌 Why is NLG Important?**

- Makes **data accessible and understandable** to humans

- Enables **conversational AI**: chatbots, virtual assistants

- Automates report writing, summaries, content generation

- Key in fields like journalism (e.g., sports, weather), healthcare
  (medical reports), business (automated insights)

 

**📌 5 Core Stages of NLG Systems:**

1.  **Content Determination:\**
    Decide what information should be conveyed.

2.  **Document Structuring:\**
    Organize selected content into a logical order.

3.  **Sentence Aggregation:\**
    Combine related information into sentences.

4.  **Lexicalization:\**
    Choose appropriate words and phrases.

5.  **Surface Realization:\**
    Apply grammar and syntax rules to generate natural text.

 

**📌 Quick Difference: NLU vs NLG**

  ---------------------------------------------------------------------
  **NLU (Natural Language                **NLG (Natural Language
  Understanding)**                       Generation)**
  -------------------------------------- ------------------------------
  Understands and interprets text        Produces human-like text

  Converts language into structured data Converts structured data into
                                         language

  Examples: Chatbot understanding,       Chatbot responses, automated
  sentiment analysis                     summaries
  ---------------------------------------------------------------------

 

 

Template Based Generation

**📌 What is Template-Based Generation in NLG?**

**Template-Based Generation** is the simplest and most traditional
approach to generating natural language text.

👉 It works by:

- **Predefining templates (fixed sentence structures)** with
  placeholders for dynamic data.

- **Filling those placeholders** with the relevant data at runtime.

 

**📌 How Does It Work?**

You prepare a sentence template like:

> 📄 **\"Today, the weather in {city} is {condition} with a high of
> {temperature}°C.\"**

At generation time:

- {city} gets replaced by **Hyderabad**

- {condition} by **sunny**

- {temperature} by **30**

And you get:

> **\"Today, the weather in Hyderabad is sunny with a high of 30°C.\"**

 

**📌 Example Applications**

  ----------------------------------------------------------------------------
  **Application**   **Example Template**           **Output**
  ----------------- ------------------------------ ---------------------------
  **Weather App**   \"It is currently              \"It is currently 27°C with
                    {temperature}°C with           clear skies.\"
                    {condition}.\"                 

  **Chatbot**       \"Hello {name}, how can I      \"Hello Neelima, how can I
                    assist you today?\"            assist you today?\"

  **Financial       \"{company} shares rose by     \"ABC Corp shares rose by
  Report**          {percentage}% today.\"         5.3% today.\"
  ----------------------------------------------------------------------------

 

 

**📌 Advantages of Template-Based Generation**

✅ Very **simple to implement**

✅ **Controlled and predictable** outputs

✅ Works well for **repetitive, structured content**

 

**📌 Limitations**

❌ **Lacks flexibility** --- can't handle complex or creative sentences

❌ Becomes cumbersome when dealing with **diverse or unstructured data**

❌ May sound **robotic or repetitive**

 

**📌 Where It's Still Useful:**

- Weather reports

- Stock market summaries

- Customer support chatbots (for fixed queries)

- Notification messages

 

**✅ Summary:**

  -------------------------------------------------
  **Feature**     **Template-Based Generation**
  --------------- ---------------------------------
  Technique       Pre-written text templates with
                  placeholders

  Data Handling   Inserts data into placeholders

  Pros            Simple, fast, predictable

  Cons            Limited variety and flexibility

  Common Uses     Reports, alerts, canned chatbot
                  responses
  -------------------------------------------------

 

 

Statistical Generation

 

**📌 What is Statistical NLG?**

**Statistical NLG** is a data-driven approach where language generation
is based on **probabilities derived from large corpora (text
datasets)**.

Instead of using fixed templates, it uses statistical models to
determine which words and phrases are most likely to appear in a given
context.

 

**📌 How Does It Work?**

- It learns **patterns, word sequences, and probabilities** from a
  training dataset.

- When generating text, it chooses the next word or phrase based on
  **likelihood (probability)** computed from those patterns.

 

**📌 Example:**

Imagine a weather report generation system.

In template-based NLG:

> "Today it is {condition} with a high of {temperature}°C."

In statistical NLG:

- It learns from a dataset of past weather reports:

  - "Today it's sunny..." occurred 500 times

  - "It is cloudy..." occurred 300 times

  - "Expect heavy rain..." occurred 200 times

Then when generating, it picks phrases based on their **probabilities**.

 

**📌 Common Statistical Techniques Used:**

- **N-gram models** (like bigram, trigram)

- **Hidden Markov Models (HMMs)**

- **Maximum Entropy Models**

These models predict the next word in a sentence based on the preceding
word(s).

 

**📌 Example --- Bigram Model:**

If we have:

- "it is" → 60%

- "it was" → 30%

- "it has" → 10%

When generating after "it", it picks **"is"** 60% of the time.

 

**📌 Advantages**

✅ More **flexible than templates**

✅ Can generate varied outputs

✅ Learns from data --- doesn't require manually creating templates

 

**📌 Limitations**

❌ Requires **large, clean datasets**

❌ May generate **grammatically awkward sentences**

❌ Struggles with **long-range dependencies (context over multiple
sentences)**

 

**📌 Where It's Used:**

- Early chatbots

- Text completion tools

- Simple report generators

 

**✅ Summary:**

  ---------------------------------------------
  **Feature**   **Statistical NLG**
  ------------- -------------------------------
  Technique     Uses statistical models trained
                on data

  Data Handling Generates text based on
                probability

  Pros          More variety and learning from
                data

  Cons          Needs large datasets, sometimes
                awkward

  Common Uses   Early chatbots, text suggestion
                tools
  ---------------------------------------------

 

**📌 Does the Model Always Pick the Highest Probability Option?**

**No --- not necessarily.**

In **probabilistic models like n-grams, it makes a random choice based
on the probability distribution.**

Meaning:

- "it is" has a 60% chance

- "it was" has a 30% chance

- "it has" has a 10% chance

When generating the next word after **"it"**, the model will **randomly
sample** from these options according to their probabilities.

 

**📌 How Does This Sampling Work?**

Let's imagine a dice roll analogy 🎲:

- Assign a range of numbers to each option based on its probability.

  - 1--60: "is"

  - 61--90: "was"

  - 91--100: "has"

Now when the system needs to pick the next word after **"it"**, it rolls
a random number between 1 and 100.

- If it rolls 45 → **"is"**

- If it rolls 75 → **"was"**

- If it rolls 95 → **"has"**

So even though **"it has"** is low probability (10%), there's still a
chance it will be chosen occasionally.

 

**📌 What If We Always Pick the Highest Probability?**

If a model always picks the most probable word (**greedy selection**):

- Text will become **very repetitive**

- It won't reflect natural language's variety

- It might even get stuck in loops or overly predictable sequences

👉 That's why most NLG systems use **random sampling based on
probability distribution** (called **stochastic sampling**) or
techniques like **top-k sampling** or **top-p (nucleus) sampling** to
balance predictability and creativity.

 

**📌 Summary:**

  --------------------------------------------------------
  **Picking Strategy**           **Behavior**
  ------------------------------ -------------------------
  Always pick highest            Predictable, repetitive
  probability (greedy)           

  Random sampling based on       Natural, varied text
  probabilities                  

  Advanced sampling (top-k,      Balanced --- avoids both
  nucleus)                       extremes
  --------------------------------------------------------

 

 

Controlled Generation

 

**📌 What is Controlled Generation in NLG?**

**Controlled Natural Language Generation** refers to generating text
**while controlling certain aspects of the output** --- such as tone,
style, length, sentiment, or specific content inclusion.

👉 In other words:

Not only do we want the system to generate text, but we also want to
**influence or steer how that text turns out**.

 

**📌 Why Do We Need Controlled Generation?**

Uncontrolled text generation (like standard statistical or neural NLG)
might:

- Go off-topic

- Produce undesirable sentiment

- Be too long/too short

- Include irrelevant or biased content

To make NLG outputs more **useful, safe, and relevant**, controlled
generation is crucial.

 

**📌 How Is Controlled Generation Achieved?**

When we ask a model to **generate text**, by default, it might say
anything that seems probable based on its training.

But in **controlled generation**, we want to guide the model --- like
giving it instructions on **how it should talk**.

For example:

- "Speak politely."

- "Sound cheerful."

- "Talk only about cricket."

- "Keep it short."

We can guide the model in **three popular ways**:

 

**1️⃣ Control Tokens (Special Tags)**

👉 **What it means:**

We add a **special word or tag** before the input to tell the model how
to respond.

**Example:**

If we want a positive review:

- **Input:** \<positive\> This phone has a large battery.

- **Output:** "This phone has an amazing battery life and it lasts all
  day!"

If we want a negative review:

- **Input:** \<negative\> This phone has a large battery.

- **Output:** "The battery is big but drains quickly and is unreliable."

The **control token (\<positive\> or \<negative\>) signals the model to
adjust its tone**.

 

**2️⃣ Conditional Generation**

👉 **What it means:**

We give the model both:

- The **data it needs to talk about**

- And a **label or control value** to decide how to talk about it

**Example:**

Generate a chatbot reply based on:

- **Input:** "My phone is slow."

- **Sentiment Control:** positive

The model is trained to respond in a **positive way even to
complaints**:

- **Output:** "We're sorry to hear that! Our latest update makes devices
  faster and more efficient --- check it out!"

This is called **conditional generation** --- the model learns to
generate differently based on a given condition.

 

**3️⃣ Parameter Tuning (Sampling Techniques)**

👉 **What it means:**

When a model picks the next word while generating a sentence, it usually
picks from a list of possible words with different probabilities.

By adjusting certain settings, we can control the **style and
diversity** of the text.

**Key parameters:**

- **Temperature:**

  - High temperature (like 1.0) → more random and creative text

  - Low temperature (like 0.2) → more predictable and safe text

**Example:**

At **low temperature**:

- "The food is good."

At **high temperature**:

- "The food is absolutely delightful, with a magical aroma."

 

- **Top-k sampling:**

  - Only choose from the top **k most likely words**

- **Top-p (nucleus) sampling:**

  - Only choose from the most likely words whose **total probability
    adds up to p (like 90%)**

This controls whether the model sticks to safe choices or tries creative
words.

 

**📌 In Short:**

  -----------------------------------------------------------------------
  **Method**      **What It Does**           **Example**
  --------------- -------------------------- ----------------------------
  Control Tokens  Adds a special word/tag to \<positive\> \"The service
                  instruct the model         was great.\"

  Conditional     Provides a control value   Complaint + positive =
  Generation      along with input           \"We're working to
                                             improve!\"

  Parameter       Adjusts how                Temperature 1.0 = more
  Tuning          random/creative the model  surprising sentences
                  is                         
  -----------------------------------------------------------------------

 

**📌 Example:**

  --------------------------------------------------------------------------
  **Type of       **Effect**            **Example**
  Control**                             
  --------------- --------------------- ------------------------------------
  **Tone**        Polite, formal,       \"Dear customer, thank you for your
                  friendly              feedback.\"

  **Length**      Short or detailed     \"It's sunny.\" vs \"The weather
                  responses             today is sunny, with clear skies.\"

  **Sentiment**   Positive, neutral,    \"This phone is great!\" vs \"This
                  negative              phone is terrible.\"

  **Topic         Talk only about a     Generate only sports news content
  Restriction**   specific subject      
  --------------------------------------------------------------------------

 

 

**📌 Where Is Controlled Generation Used?**

✅ Chatbots (formal vs casual replies)

✅ Content creation platforms (positive reviews only)

✅ Personal assistants (length & politeness control)

✅ Social media content moderation tools

 

**📌 Advantages**

✅ Ensures **consistency and appropriateness**

✅ Allows **tailoring content to different audiences**

✅ Reduces risk of **bias, offensive or irrelevant text**

 

**📌 Limitations**

❌ Requires extra **design, labeling, or model training**

❌ Sometimes control signals might conflict with natural text flow

❌ More complex than template-based or plain statistical NLG

 

**✅ Summary:**

  -------------------------------------------------
  **Feature**   **Controlled Generation**
  ------------- -----------------------------------
  Technique     Directs generation based on control
                parameters

  Control Types Tone, style, topic, sentiment,
                length, etc.

  Pros          Tailored, appropriate, on-topic
                outputs

  Cons          Adds complexity, needs careful
                tuning

  Common Uses   Chatbots, AI writing tools,
                automated reports
  -------------------------------------------------

 

 

Evaluating Generated Language

 

**📌 What is *Evaluating Generated Language* in NLG?**

When a system generates text (like a chatbot reply, a summary, or a
product review), we need to check:

- **How good is the text?**

- **Is it meaningful, fluent, and relevant?**

- **Does it match human-like quality?**

This process is called **evaluation of generated language**.

It's an important step to ensure our NLG system is actually producing
useful and natural text.

 

**📌 How is Generated Language Evaluated?**

There are **two main ways to evaluate generated text:**

**1️⃣ Automatic Evaluation**

Using predefined mathematical metrics to compare the generated text
with:

- One or more **reference (ideal) texts**

- Or to measure qualities like diversity, fluency, and relevance

**2️⃣ Human Evaluation**

Asking real people to rate or judge the text based on:

- Fluency

- Coherence

- Relevance

- Creativity

- Overall naturalness

 

**📌 Common Automatic Evaluation Metrics:**

  -----------------------------------------------------------------------
  **Metric**              **What it measures**   **How it works**
  ----------------------- ---------------------- ------------------------
  **BLEU (Bilingual       Measures similarity to Compares overlapping
  Evaluation              reference text         n-grams (word sequences)
  Understudy)**                                  

  **ROUGE                 Measures how much of   Counts overlapping
  (Recall-Oriented        the reference text is  words/phrases
  Understudy for Gisting  captured               
  Evaluation)**                                  

  **METEOR**              Considers synonyms,    Scores based on
                          word stems, and exact  alignment between
                          matches                generated and reference
                                                 text

  **CIDEr**               Measures consensus     Scores higher if
                          among multiple         generated text matches
                          human-written texts    multiple references

  **Perplexity**          Measures how well a    Lower perplexity =
                          language model         better model prediction
                          predicts a sample      
  -----------------------------------------------------------------------

 

 

**📌 Example:**

**Reference Text:**

*"The food was delicious and the service was excellent."*

**Generated Text A:**

*"The food was tasty and the service was great."*

**Generated Text B:**

*"Food good. Service fine."*

**Evaluation:**

- **BLEU score** for Text A will be higher (more overlap with reference
  n-grams)

- **Text A** will likely be rated better by human judges for fluency and
  coherence

 

**📌 When Do We Use Human Evaluation?**

When:

- Automatic metrics don't capture nuances like tone, creativity, or
  emotional appropriateness

- You're evaluating chatbot conversations, poems, story generation, or
  creative writing

- For final validation in critical applications (like medical or legal
  report generation)

**Common Human Evaluation Criteria:**

- **Fluency:** Is the sentence grammatically correct?

- **Coherence:** Does it make sense in context?

- **Relevance:** Is it on-topic and appropriate?

- **Naturalness:** Does it sound human-written?

- **Style/Tone:** Does it match the desired style (formal, polite,
  casual)?

 

**📌 Summary:**

  --------------------------------------------------------------------------
  **Evaluation    **Examples**                     **When to Use**
  Type**                                           
  --------------- -------------------------------- -------------------------
  **Automatic**   BLEU, ROUGE, METEOR, CIDEr,      Fast, large-scale,
                  Perplexity                       objective scoring

  **Human**       Ratings on fluency, coherence,   When text quality is
                  relevance, naturalness           subjective or nuanced
  --------------------------------------------------------------------------

 

 

Narrative and Story Generation

 

**📌 What is Narrative and Story Generation? (Super Simple)**

👉 Think of it like this:

You tell a computer:

**\"Make up a story about a boy and a dragon.\"**

And the computer **writes a story for you** --- with characters, events,
a beginning, a problem, and an ending.

That's **Narrative and Story Generation** --- a part of Natural Language
Generation (NLG) where we teach computers to **write stories just like
humans do**.\
\
 

**📌 How Does It Work? (Step by Step)**

When a human writes a story:

- First, they think of **what will happen** (events)

- Then, they decide **in what order things will happen** (structure)

- Finally, they **write nice sentences**

A computer story system does the same:

**1️⃣ Decide what the story is about**

(Who are the characters? What is the problem?)

👉 Example:

- Characters: **Riya** (hero), **Kira** (monster)

- Problem: Riya wants to find a treasure

- Ending: Riya defeats Kira

 

**2️⃣ Decide the order of events**

(What happens first, next, last?)

👉 Example:

- First: Riya hears about a treasure

- Then: She meets Kira guarding the treasure

- Finally: She defeats Kira and finds the treasure

 

**3️⃣ Write proper sentences for these events**

👉 Example:

- *Riya was a brave girl living in a village.*

- *One day, she heard about a treasure hidden in the forest.*

- *When she went there, she met a terrible monster named Kira.*

- *After a long fight, Riya defeated Kira and found the treasure.*

Now, this becomes a **generated story**.

 

**📌 Why is This Hard for Computers?**

Because:

- The computer must remember what happened before (to avoid mistakes)

- Keep the characters behaving like themselves (the hero should stay
  brave)

- Events must make sense (she can't find the treasure before meeting the
  monster)

That's why **story generation is harder than normal sentence
generation**.\
\
 

**📌 How is a Story Generated by a Computer?**

There are **two main approaches** computers use to generate stories:

 

**📖 1️⃣️⃣ Traditional Rule-Based or Template-Based Story Generation**

This was the older way before neural networks.

**🔸 How it works:**

- You write **story rules or templates**

- The system picks characters, places, and events based on those rules

- It fills those into sentence templates to build a story

**Example Template:**

\"{character} went to the {place} to find the {object}. {event}
happened. In the end, {character} {result}.\"

**System randomly fills:**

- character = \"Riya\"

- place = \"forest\"

- object = \"treasure\"

- event = \"She met a dragon\"

- result = \"defeated the dragon\"

And outputs:

> "Riya went to the forest to find the treasure. She met a dragon. In
> the end, Riya defeated the dragon."

This method is simple but lacks creativity and variety.

 

**📖 2️⃣️⃣ Modern Neural Network-Based Story Generation (like GPT)**

This is how GPT or modern AI models generate stories.

**🔸 How it works:**

1.  **The model is trained on thousands or millions of stories\**
    It reads storybooks, articles, novels --- learning patterns like:

- How a story begins

- What characters do

- What usually happens next

- How to end a story

2.  **When you give it a prompt, it predicts the next word one by one**

👉 Example:

**Prompt:** "Once upon a time, there was a brave knight named Aryan."

**The model predicts:**

- Next word: *He*

- Next: *lived*

- Next: *in*

- Next: *a*

- Next: *small*

- Next: *village.*

- And so on...

**It keeps generating word by word** based on what it has learned, using
probability --- predicting what makes sense to follow next.

1.  **It uses probabilities to pick the next word**

- After *"He lived in a"*, possible words:

  - *castle (60%)*

  - *village (30%)*

  - *forest (10%)*

It randomly selects one based on these chances.

1.  **It remembers context**

- If it said \"village\" before, it tries not to contradict itself
  later.

- It tries to keep characters and events consistent.

 

**📌 How Does the Model Learn Story Patterns?**

During training, the model sees examples like:

> *"A boy found a cave. Inside the cave was a treasure. The boy took the
> treasure and left."*

It learns:

- Stories start by introducing a character and setting.

- Then something happens.

- Then a result.

So when you give it a new character, it can create similar new events by
predicting what would logically follow.

 

**📌 Visual Flow:**

**Prompt** → **Predict next word** → **Update context** → **Predict next
word** → ...

**Until story is complete**

 

**📌 Summary**

  ------------------------------------------------------------
  **Traditional Way**             **Modern Neural Way (like
                                  GPT)**
  ------------------------------- ----------------------------
  Uses fixed templates and random Learns from reading millions
  events                          of stories

  Very predictable                Can generate new, creative
                                  stories

  Limited variety                 More natural, human-like
                                  storytelling

  No learning                     Predicts next word based on
                                  context
  ------------------------------------------------------------

 

 

**✅ Example (From GPT):**

**Prompt:** *"In a faraway kingdom, there was a kind queen named
Elara."*

**Generated:**

*She ruled her people with wisdom and grace. One day, a mysterious
traveler arrived at the castle gates, carrying a box covered in ancient
symbols...*

 

**📌 That's how story generation works --- either by:**

- Filling pre-made templates (old)

- Or predicting next words one by one, learning from examples (modern
  GPT-style)

 

 

Text Summarization

 

**What is Text Summarization in NLP/NLG?**

**Text Summarization** is the process of taking a long piece of text and
generating a **shorter, meaningful, and coherent version** while
preserving the **important information and overall meaning**.

👉 In simple words:

It's like reading a long news article and then writing a **summary in
your own words** --- but letting a computer do it automatically.

 

**📌 Types of Text Summarization**

There are **two main types**:

**1️⃣ Extractive Summarization**

- Picks and extracts **important sentences or phrases directly from the
  original text**.

- No new sentences are created.

**Example:**

**Original Text:**

*\"Artificial Intelligence is transforming industries. Experts believe
AI will continue to shape the future. AI-driven tools are already making
an impact in healthcare, finance, and education.\"*

**Extractive Summary:**

*\"Artificial Intelligence is transforming industries. AI-driven tools
are already making an impact in healthcare, finance, and education.\"*

👉 Notice: It just picked key sentences as-is.

 

**2️⃣ Abstractive Summarization**

- **Generates new sentences in its own words** while keeping the meaning
  intact.

- Like how humans write a summary.

**Example:**

**Original Text:**

*\"Artificial Intelligence is transforming industries. Experts believe
AI will continue to shape the future. AI-driven tools are already making
an impact in healthcare, finance, and education.\"*

**Abstractive Summary:**

*\"AI is revolutionizing industries and impacting fields like
healthcare, finance, and education.\"*

👉 Notice: New sentence, same meaning.

 

**📌 How Does It Work?**

**For Extractive Summarization:**

- Rank sentences based on importance

- Pick top-ranked ones for the summary

**For Abstractive Summarization:**

- Understand text meaning (context)

- Rephrase and generate new summary sentences using models like:

  - **Seq2Seq models**

  - **Transformer models (like BART, T5, GPT)**

 

**📌 Applications of Text Summarization**

✅ News article summaries

✅ Headline generation

✅ Academic paper abstracts

✅ Customer review summarization

✅ Legal/financial report summarization

✅ Meeting note summarization

 

**📌 Advantages**

✅ Saves time for readers

✅ Makes large documents digestible

✅ Useful for quick decision-making

 

**📌 Challenges**

❌ Hard to capture context in long documents

❌ Abstractive summaries may lose details or introduce errors

❌ Maintaining factual accuracy

 

**📌 Summary:**

  -------------------------------------------------------------
  **Feature**   **Extractive            **Abstractive
                Summarization**         Summarization**
  ------------- ----------------------- -----------------------
  How it works  Picks key sentences     Generates new sentences
                as-is                   

  Pros          Simple, factual         Human-like, fluent

  Cons          May be choppy,          Complex, risk of errors
                unconnected             

  Common uses   News digests, meeting   AI-generated article
                notes                   summaries
  -------------------------------------------------------------
