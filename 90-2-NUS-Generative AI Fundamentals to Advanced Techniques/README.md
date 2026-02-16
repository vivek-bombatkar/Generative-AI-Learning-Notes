# Generative AI & Deep Learning Notes
### My Learning Notes from [NUS generative-ai-fundamentals-to-advanced-techniques-programme](https://nus.comp.emeritus.org/generative-ai-fundamentals-to-advanced-techniques-programme)

## Table of Contents
- [Learning Paradigms in Machine Learning](#learning-paradigms-in-machine-learning)
- [From Brains to Artificial Neural Networks](#from-brains-to-artificial-neural-networks)
- [Convolutional Neural Networks](#convolutional-neural-networks)
- [Transformers and Attention Mechanisms](#transformers-and-attention-mechanisms)
- [Transformer Model Families](#transformer-model-families)
- [Alignment, Reliability, and Knowledge Grounding](#alignment-reliability-and-knowledge-grounding)
- [Multimodal and Generalist Models](#multimodal-and-generalist-models)
- [Useful Links](#useful-links)



## Learning Paradigms in Machine Learning

### Reinforcement Learning vs Supervised Learning

- **Supervised Learning:** A learning paradigm where a model is trained on labelled input-output pairs and receives direct feedback on errors. citeturn20search0  
- **Reinforcement Learning (RL):** A learning paradigm where an agent learns by interacting with an environment using rewards and penalties. citeturn5search0  

#### Supervised Learning
1. Collect labelled data. citeturn20search0˙  
2. Split into training and validation sets. citeturn20search0  
3. Train model to minimize loss. citeturn20search0  
4. Evaluate performance. citeturn20search0  

#### Reinforcement Learning
1. Initialize agent and environment. citeturn5search0  
2. Perform actions based on current policy. citeturn5search0  
3. Receive reward feedback. citeturn5search0  
4. Update policy using reward signal. citeturn5search0  
5. Repeat until convergence. citeturn5search0  

#### Examples
- **Supervised Learning:** Image classification, spam detection, price prediction. citeturn20search0  
- **Reinforcement Learning:** Game playing, robotics control, recommendation strategies with delayed rewards. citeturn5search0turn2search3  

### Unsupervised Learning

- **Unsupervised Learning:** A branch of machine learning where models learn patterns and relationships from unlabelled data without predefined outputs. citeturn20search1  

#### Examples

##### Clustering Problems
- Customer segmentation citeturn20search1  
- Image segmentation citeturn20search1  

##### Dimensionality Reduction Problems
- Principal Component Analysis (PCA) citeturn4search12  
- t-SNE citeturn5search1  

##### Anomaly Detection
- Fraud detection citeturn20search7  

##### Generative Models (partly unsupervised)
- Autoencoders citeturn5search3  
- GANs (Generative Adversarial Networks) citeturn5search2  

##### Market Basket Analysis
- Identifying frequently co-occurring items citeturn20search6turn20search22  

## From Brains to Artificial Neural Networks

### The Human Brain and Neural Complexity

#### Overview
The human brain is an incredibly complex organ and one of nature’s greatest engineering marvels.  
The **neocortex** plays a key role in higher cognitive functions such as:
- 🧠 Reasoning  
- 👁️ Perception  
- 🤔 Decision-making  
- 🗣️ Language  
Its massive connectivity gives the brain extraordinary computational power. citeturn4search5turn7search6  

#### Neurons and Synapses
- The brain contains around **100 billion neurons** 🧩 citeturn4search5  
- Each neuron connects to **1,000–10,000** other neurons through **synapses** citeturn7search6turn7search2  
- Synapses transmit information via ⚡ electrical and 🧪 chemical signals citeturn7search2  
- The neocortex alone is estimated to have about **500 trillion synapses**, forming a massive biological network capable of learning and thought citeturn7search2turn8search0  

> Note (clarification added): Published estimates vary by method and definition; widely cited totals include ~86 billion neurons for the whole brain and on the order of 10^14 synapses overall, with neocortex synapse counts reported in the ~10^14 range as well. citeturn4search5turn7search2  

#### Neural Wiring and Efficiency
- The neocortex contains roughly **300 million feet (≈91,440 km)** of neural wiring citeturn8search0  
- This wiring is compacted into a volume of about **1.5 quarts (≈1.4 liters)** 🤯 citeturn8search0  
- Such efficiency is achieved through several biological optimizations: citeturn8search0turn4search5  

##### Folding of the Cortex
- The brain surface is folded into **gyri** (ridges) and **sulci** (grooves)  
- Folding increases surface area without increasing overall volume citeturn8search0  

##### Myelination
- Axons are coated with **myelin**, a fatty insulating layer  
- Myelin speeds up signal transmission and reduces energy usage citeturn4search15  

##### Specialized Networks
- The brain is organized into specialized functional areas, such as:
  - 👀 Visual cortex
  - ✋ Motor cortex
  - 🧠 Prefrontal cortex
- Specialization minimizes unnecessary wiring and improves processing speed citeturn8search0turn4search5  

### Neural Networks and Artificial Neural Networks

- Neural networks are inspired by biological brains. citeturn5search0  
- Artificial neurons approximate real neurons. citeturn5search0  
- ANNs are networks of artificial neurons. citeturn5search0  
- ANNs are simplified models of brain functionality. citeturn5search0  
- Practically, ANNs are parallel computational systems. citeturn5search0  

#### Definitions
- **Neural Networks (NNs):** Networks of neurons similar to those found in biological brains. citeturn5search0  
- **Artificial Neurons:** Crude approximations of biological neurons, implemented as mathematical or software constructs. citeturn5search0  
- **Artificial Neural Networks (ANNs):** Networks of artificial neurons that approximate certain functions of real brains. citeturn5search0  

### Biological vs Artificial Neurons

#### Biological Neurons
- Biological neurons have **synaptic gaps** of varying strengths 🔗 citeturn7search2  
- These synapses connect to the **soma (cell body)** 🧠 citeturn7search2  
- Signal strength depends on synaptic weight and connectivity citeturn7search2  
- Information flows via:
  - 🌿 Dendrites (input)
  - 🧠 Cell body (integration)
  - ⚡ Axon (signal transmission)
  - 🔚 Axon terminals (output) citeturn7search2  

#### Artificial Neurons
- Artificial neurons replace synapses with **numerical inputs** citeturn5search0  
- Inputs can come from:
  - Other neurons
  - Sensors
  - Data features
  - Variables citeturn5search0  
- Core operations include:
  - ➕ Weighted sum (Σ)
  - 📉 Activation / threshold function citeturn5search0  

#### Computational Power
- The neocortex contains about **500 trillion synapses** operating **in parallel** citeturn8search0turn7search2  
- Enables massive information processing and storage simultaneously citeturn7search2  
- The human brain operates on roughly **20 watts of power** 💡 citeturn7search1turn4search8  
- This is far more **energy-efficient** than modern supercomputers citeturn7search1  

#### Implications for Intelligence and Learning
- High neuron density and interconnectivity enable human intelligence citeturn4search5  
- **Plasticity** allows neural connections to reorganize with:
  - 📚 Learning
  - 🧠 Experience
  - 🩹 Recovery from injury citeturn4search5  
- This adaptability is central to skill acquisition and cognition citeturn4search5  

#### Artificial Neural Networks (ANNs)
- ANNs are inspired by biological neural systems citeturn5search0  
- They aim to approximate learning and decision-making citeturn5search0  
- While powerful, they are far less energy-efficient than the human brain citeturn7search1  
- The neocortex remains a benchmark for efficient computation and learning citeturn7search1turn4search5  

### Artificial Neural Networks. Why?

- 🧮 **Extremely powerful computational devices**  
  - Turing-equivalent universal computers citeturn5search0  
- ⚡ **Massive parallelism**  
  - Many simple units operate simultaneously, making computation efficient citeturn5search0  
- 📚 **Learning and generalization**  
  - Learn directly from training data  
  - No need for carefully handcrafted rules or designs citeturn20search0  
- 🛡️ **Fault-tolerant and noise-tolerant**  
  - Performance degrades gracefully even with imperfect data or failures citeturn5search0  
- 🧠 **Beyond symbolic systems**  
  - Can do everything a symbolic or logic-based system can, and more citeturn5search0  
- 📊 **Excellent with unstructured data**  
  - Particularly strong with:
    - 📝 Text  
    - 🖼️ Images  
    - 🔊 Audio  
    - Other semi-structured data citeturn5search0  

## Convolutional Neural Networks

### Deep Convolutional Neural Networks (CNNs)

#### Definition
**Deep Convolutional Neural Networks (CNNs)** are a specialized type of neural network designed to process **structured, grid-like data**, especially **images** 🖼️. citeturn4search1  

#### Key characteristics
- Designed for **spatial data** arranged in grids citeturn4search1  
- Combine:
  - 🧠 **Convolutional layers** for feature extraction
  - 🔗 **Fully connected layers** for decision-making citeturn4search1  
- Inspired by the **human visual system** citeturn4search1  
- Highly effective when **spatial hierarchies** matter citeturn4search1  

#### Core building blocks
- 🟨 **Input**  
- 🟩 **Convolutional Layer**. Extracts local features (edges, textures)
- 🟦 **Pooling Layer**. Reduces spatial size and computation
- 🧪 **Activation Function**. Adds non-linearity
- 🔵 **Fully Connected Layer**. Performs classification or prediction
- 🎯 **Output** citeturn4search1  

#### Why CNNs work well
- Learn **hierarchical features**. from simple edges to complex shapes citeturn4search1  
- Preserve **spatial relationships** in data citeturn4search1  
- Particularly strong for **vision-based tasks** citeturn4search1  

#### Typical use cases
- Image classification
- Object detection
- Image segmentation
- Visual pattern recognition citeturn4search1  

#### Advantages of CNNs
- ⚙️ **Automatic Feature Extraction**  
  - CNNs learn features directly from raw data  
  - No manual feature engineering required citeturn4search1  
- 🔁 **Parameter Sharing**  
  - Same filters are reused across the image  
  - Fewer parameters than fully connected networks citeturn4search1  
- 📍 **Translation Invariance**  
  - Recognize patterns (edges, shapes, objects) regardless of position in the image citeturn4search1  
- 📐 **Efficient for High-Dimensional Data**  
  - Scales well to large images and datasets citeturn4search1  
- 🏆 **State-of-the-Art Performance**  
  - Top results in:
    - Image classification
    - Object detection
    - Image segmentation citeturn4search1  
- 🌍 **Adaptability to Diverse Domains**  
  - Can be applied to:
    - 🖼️ Images
    - 🔊 Audio spectrograms
    - ⏱️ Time-series data  
  - Requires minimal architectural changes citeturn4search1  

#### Limitations of CNNs
- 💻 **Computationally Intensive**  
  - Training requires powerful hardware (GPUs, TPUs) citeturn4search1  
- 📊 **Data Hungry**  
  - Needs large labelled datasets for good performance  
  - Data collection and annotation can be expensive citeturn4search1  
- 🔍 **Lack of Interpretability**  
  - Acts as a black-box model  
  - Difficult to understand or debug decisions citeturn4search1  
- 📉 **Overfitting Risk**  
  - Without proper regularisation, models may memorise training data citeturn4search1  
- 🎛️ **Sensitivity to Hyperparameters**  
  - Performance depends heavily on:
    - Architecture
    - Learning rate
    - Other tuning choices citeturn4search1  

## Transformers and Attention Mechanisms

### Attention Mechanism in Transformers

#### Attention
At the core of modern NLP lies **attention**.  
It allows models to **focus**, not memorise.  
Instead of treating all words equally, attention helps the model decide **what matters most** in context. citeturn15view0  

A commonly used formulation is scaled dot-product attention:  
**Attention(Q, K, V) = softmax(QKᵀ / √dₖ) · V**. citeturn15view0  

#### Query, Key, and Value (QKV)

To understand attention, everything revolves around three components:

- 🔍 **Query (Q)**  
  - Represents the **current word or token**
  - Think of it as a word *asking* for relevant context citeturn15view0  

- 🗝️ **Key (K)**  
  - Represents **all words in the input sequence**
  - Each word has a key used to measure relevance to the query citeturn15view0  

- 📦 **Value (V)**  
  - Contains the **actual information** (embeddings)
  - This is the content passed forward once relevance is determined citeturn15view0  

#### How Attention Works (Step-by-Step)

- Compute **attention scores** using dot products between Query and Keys. citeturn15view0  
- Scale scores by **√dₖ** for stability. citeturn15view0  
- Apply **softmax** to get a probability distribution over tokens. citeturn15view0  
- Take the **weighted sum of Values** to produce a context-aware representation. citeturn15view0  

#### Why Attention Matters
- Focuses on **relevant words**, regardless of position citeturn15view0  
- Handles **long-range dependencies** citeturn15view0  
- Essential for understanding meaning in complex sentences citeturn15view0  

### Multi-Head Attention

#### What is Multi-Head Attention?
- Instead of one attention mechanism, the model uses **multiple heads**
- Each head attends to **different aspects** of the sequence citeturn15view0  

#### How It Works
- Input is split into multiple Q, K, V sets  
- Each head computes attention **independently**  
- Outputs are:
  - Concatenated
  - Passed through a linear layer  
➡️ Result. richer and more expressive representations citeturn15view0  

### Feedforward Networks (FFN)

#### Role of Feedforward Networks
- Applied **after multi-head attention**
- Operates **independently on each position** citeturn15view0  

#### Structure
- Linear layer
- ReLU activation
- Linear layer citeturn15view0  

#### Purpose
- Captures **more abstract patterns**
- Refines representations beyond word-to-word relationships
- Output is passed to the next transformer layer citeturn15view0  

### Transformer Architecture. Big Picture

- 🔡 Embeddings citeturn15view0  
- 🔁 Self-attention (QKV) citeturn15view0  
- 🔀 Multi-head attention citeturn15view0  
- 🧪 Feedforward networks citeturn15view0  
- 🔄 Stacked layers for depth citeturn15view0  

Together, these components allow transformers to:
- Understand context deeply
- Scale efficiently
- Generate high-quality, human-like text citeturn15view0  

### Training a Transformer
- 🔡 Data preprocessing. tokenisation and embeddings citeturn15view0turn2search4  
- 📍 Positional encodings added for sequence order citeturn15view0  
- 🎲 Random weight initialisation citeturn15view0  
- 🔁 Training via **backpropagation** citeturn15view0  
- 🎛️ Hyperparameters tuned:
  - Learning rate
  - Batch size citeturn15view0turn2search4  
- Goal. minimise loss and optimise performance citeturn15view0  

### Computational Challenges
- Large models are **computationally intensive** citeturn15view0turn1search7  
- Example:
  - GPT-3 has **175 billion parameters** citeturn1search7  
- CPUs are insufficient for training
- Requires:
  - ⚡ GPUs
  - 🚀 TPUs citeturn1search7turn17view0  
- Computational needs have grown **exponentially**
- Hardware advances are critical for progress citeturn12search7  

### Transformer Variants
- Transformers excel at sequential data
- Variants optimise performance for specific tasks citeturn3search10turn4search0  

#### Hybrid Architectures

##### CNN + Transformer
- Common in computer vision
- CNN captures **local spatial features**
- Transformer captures **long-range dependencies** citeturn4search1turn15view0  
- Used in:
  - Vision Transformers (ViTs)
  - ResNet hybrids citeturn4search1  

##### RNN + Transformer
- Useful for:
  - 🗣️ Speech recognition
  - ⏱️ Time-series forecasting citeturn15view0  
- RNN handles local sequences
- Transformer handles global context citeturn15view0  

##### Benefits
- Reduced computational cost
- Better efficiency and scalability
- Strong performance on complex tasks citeturn3search10turn4search0  

#### Efficient Transformers (Long Sequences)
- Standard transformers scale **quadratically** with sequence length due to self-attention citeturn3search10turn3search3turn4search0  
- Problematic for long documents
- Sparse and efficient variants address this citeturn3search10turn4search0  

##### Longformer
- Uses attention patterns combining local + global information
- Scales more linearly for long documents citeturn3search10turn3search2  
- Ideal for:
  - Document-level QA
  - Summarisation citeturn3search10  

##### Linformer
- Approximates attention via low-rank structure
- Reduces memory and inference cost for long sequences citeturn3search3  

##### Reformer
- Optimised for memory efficiency
- Uses locality-sensitive hashing (LSH) attention and reversible layers citeturn4search0  

## Transformer Model Families

### BERT vs GPT vs T5

#### BERT
- **Bidirectional Encoder Representations from Transformers**
- Understands text in **both directions** citeturn17view0  
- Best suited for:
  - 😊 Sentiment analysis
  - 🏷️ Text classification
  - ❓ Question answering citeturn17view0  

#### GPT
- **Generative Pre-trained Transformer**
- **Autoregressive**. predicts the next word in a sequence citeturn15view0turn1search7  
- Ideal for:
  - 💬 Conversational AI
  - ✍️ Content generation
  - 💻 Code generation citeturn1search7  

#### T5
- **Text-to-Text Transfer Transformer**
- Treats **all NLP tasks as text-to-text** citeturn2search4  
- Highly versatile for:
  - 🌍 Translation
  - 🧾 Summarisation
  - ❓ Question answering citeturn2search4  

### BERT limitations and common variants

#### Limitations of BERT (encoder-only)
- **Context window limit:** BERT is commonly pre-trained with an input context window up to **512 tokens**. citeturn18view0turn0search13  
- **Dimensionality cost:** common configurations include hidden sizes like **768** (BERT-base), which affects memory and compute. citeturn0search13turn19search15  
- **No true free-form generation:** BERT is trained with a masked prediction objective (and NSP in the original version), so it excels at understanding/representation learning rather than left-to-right generation. citeturn17view0turn18view0  

#### Variants mentioned in the notes and discussion
- **RoBERTa:** improved BERT pretraining by changing key training choices (for example, removing NSP and using dynamic masking and more data/compute). citeturn0search6turn0search10  
- **ELECTRA:** replaces masked-token prediction with **replaced token detection** (discriminator predicts whether a token was replaced by a generator). citeturn0search3turn0search7  
- **DistilBERT:** uses knowledge distillation to create a smaller, faster model while preserving much of BERT’s performance. citeturn11search0  
- **ALBERT:** reduces parameters (e.g., factorized embeddings and parameter sharing) to improve efficiency. citeturn11search5  
- **SpanBERT:** masks contiguous spans and trains objectives tailored to span representations. citeturn11search6  
- **CodeBERT:** pre-trained for natural language + programming language tasks (bimodal NL–PL). citeturn11search3  

### GPT models and how ChatGPT works

#### Model Scale Comparison

| Model   | Number of Parameters |
|--------|----------------------|
| GPT-1  | 117M |
| GPT-2  | 1.5B |
| GPT-3  | 175B |
| GPT-3.5 | GPT-3 + ~6B |
| GPT-4  | ~1.7T |

Notes/clarifications added:
- GPT-2 (1.5B) and GPT-3 (175B) parameter counts are documented in their respective technical reports. citeturn1search0turn1search7  
- Some commonly repeated values for “GPT-3.5” and “GPT-4 parameter count” are **not officially disclosed**. The GPT-4 technical report explicitly states it does not provide details such as model size. citeturn14view0  

#### How ChatGPT Works (End-to-End Flow)

➡️ **Pre-training**  
- Trained on massive internet text
- Learns to predict the **next token**
- Captures grammar, facts, and language patterns citeturn1search7turn13view0  

➡️ **Fine-tuning**  
- Uses datasets reviewed by **human trainers**
- Learns to generate safer and more helpful responses
- Generalises from human feedback citeturn1search1turn1search5  

➡️ **Input Processing**  
- User input is **tokenised** into words or subwords
- Tokens are fed into the transformer model citeturn2search4turn13view0  

➡️ **Contextual Understanding**  
- Maintains conversation history 🧠
- Transformer architecture models long-range dependencies
- Enables context-aware responses citeturn15view0turn13view0  

➡️ **Response Generation**  
- Predicts next tokens based on learned patterns
- Produces coherent, human-like text citeturn15view0turn13view0  

➡️ **Sampling and Optimisation**  
- Uses probabilistic sampling 🎲
- Adds controlled randomness for natural responses
- Safety techniques reduce harmful outputs citeturn6search0turn1search1turn13view0  

➡️ **Post-processing**  
- Removes special tokens and formatting
- Final response is shown to the user 💬 citeturn13view0  

### Encoder–decoder models and key examples

#### Encoder-decoder models
Encoder–decoder models are a fundamental architecture in modern deep learning. These models bring together an encoder and a decoder, enabling efficient processing of input data while generating meaningful output. citeturn15view0turn2search4  

- Start with the encoder, which takes the input and processes it to capture its contextual meaning. It transforms the data into a structured representation that the model can understand. citeturn15view0  
- Then comes the decoder, which uses that structured information to generate text that is natural and coherent. Unlike simple text generation models that produce output sequentially, encoder-decoder models maintain logical consistency by referencing the complete context provided by the encoder. citeturn15view0  

#### Applications of encoder–decoder models
Encoder–decoder models power a variety of real-world applications including:
- Machine translation to convert text from one language to another citeturn15view0turn2search4  
- Text summarisation, extracting the key points while preserving the meaning citeturn2search4turn2search2  
- Caption generation, generating textual descriptions for images or videos citeturn3search1  

#### BART
BART (Bidirectional and Auto-Regressive Transformer) is a hybrid model that integrates the strengths of both BERT and GPT, making it highly effective for tasks requiring text reconstruction and controlled generation. citeturn2search1  

- It employs a bidirectional encoding process (similar to BERT) for comprehensive contextual understanding. citeturn2search1  
- It uses denoising objectives such as span corruption / text infilling (replacing spans with a single mask token), forcing reconstruction of phrases while maintaining coherence. citeturn2search1  
- On the decoding side, BART adopts autoregressive generation (similar to GPT), where tokens are generated one at a time while conditioning on the encoded input. citeturn2search1turn15view0  

##### BART’s Denoising Process
BART employs a denoising autoencoder approach, where input data is deliberately corrupted before being passed through the model. citeturn2search1  

Noise-insertion techniques described in the notes:
- Token masking replaces random words with a special [MASK] token. citeturn2search1  
- Token deletion removes entire words from the sequence. citeturn2search1  
- Text infilling (span corruption) replaces entire spans of text with a single [MASK] token. citeturn2search1  

This training strategy makes BART robust to noisy or incomplete inputs (e.g., imperfect formatting or missing spans) and supports tasks like summarisation, paraphrasing, translation, and text completion. citeturn2search1  

#### PEGASUS
PEGASUS (Pre-training with Extracted Gap-sentences for Abstractive Summarisation) is specifically designed to enhance text summarisation by masking entire sentences rather than individual tokens during pretraining. citeturn2search2turn2search10  

Key points from the notes:
- Entire sentences (instead of tokens) are masked during pretraining. citeturn2search2  
- Sentences may be randomly identified, though the method focuses on removing “important” sentences (gap sentence generation). citeturn2search2  
- These are identified to be ones with high similarity to the rest of the document (intuitively encouraging summary-like targets). citeturn2search2  

PEGASUS achieves strong summarisation performance and can require minimal fine-tuning for high-quality abstractive summaries. citeturn2search2turn2search10  

#### T5
T5 (Text-to-Text Transfer Transformer) reformulates all NLP tasks into a text-to-text format, using an encoder-decoder architecture for tasks like translation, summarisation, and Q&A. citeturn2search4  

Text to Text
- Indicative of the types of input and output to be expected citeturn2search4  
- Encoder model used to get information for input text citeturn2search4  
- Decoder model used to generate output text citeturn2search4  

Transfer Transformer
- Transformer capable of employing transfer learning citeturn2search4  
- Allows for multiple NLP tasks to be accomplished by the model:
  - Translation
  - Summarisation
  - Q&A citeturn2search4  

T5 uses task prefixes (instructions) to unify workflows across tasks (e.g., “translate …”, “summarize …”). citeturn2search4  

## Alignment, Reliability, and Knowledge Grounding

### GPT and Reinforcement Learning

GPT models, built on deep learning, have revolutionised language understanding and generation by predicting text patterns with remarkable fluency. Reinforcement Learning (RL), on the other hand, empowers systems to learn through trial and error, optimising actions for long-term rewards. Together, they unlock new frontiers in adaptive, intelligent decision-making and human-like interactions. citeturn5search0turn1search7  

### Reinforcement Learning from Human Feedback (RLHF)

Reinforcement Learning from Human Feedback (RLHF) enhances GPT’s ability to generate not just human-like text but also reliable and contextually appropriate responses. citeturn1search1turn13view0  

Why RLHF is used (as captured in the notes and discussion):
- GPT can produce fluent language, but a pure Transformer architecture does not *inherently* verify factual accuracy or suitability of outputs. citeturn13view0turn1search1  
- Example (preserved): “Under Augustus, the Roman Empire came to [MASK]” — GPT alone may not “know” which completion is historically correct without grounding or reliable internal knowledge. citeturn13view0turn1search1  

Typical RLHF-style pipeline:
- A model is fine-tuned using a **reward model** (often a transformer trained on human preference rankings) that prioritizes more useful outputs. citeturn1search1turn1search5  
- Human reviewers evaluate and correct a subset of responses, reinforcing high-quality and informative text generation. citeturn1search1turn1search5  

### Retrieval-Augmented Generation (RAG)

Retrieval-Augmented Generation (RAG) enhances GPT by integrating external knowledge sources, addressing its limitations with fixed training data. citeturn1search2turn1search6  

Key points (preserved and clarified):
- ChatGPT generates responses off fixed training data. citeturn13view0  
- Without external sources, it struggles with real-time updates, niche topics, or retrieving specific factual details. citeturn1search2turn13view0  
- RAG retrieves relevant documents or data (e.g., knowledge bases, web snapshots, or indexed corpora), feeding them into the model as additional context for response generation. citeturn1search2turn1search6  
- This approach improves accuracy, reduces hallucinations, and enables up-to-date, domain-specific answers. citeturn1search2turn13view0  
- Combining GPT-style models with retrieval creates more reliable, informed, and adaptable AI systems for business and research applications. citeturn1search2turn1search6  

### Zero-Shot Learning (ZSL)

Zero-Shot Learning (ZSL) enables GPT to perform tasks without additional training, relying solely on its extensive pretraining. Instead of requiring labeled examples or fine-tuning: citeturn1search7  

- GPT leverages its extensive pre-training to perform tasks without additional training, generating relevant outputs directly from prompt instruction. citeturn1search7  
- Enhanced flexibility: allows adaptation to new tasks without extra fine-tuning. citeturn1search7  
- Streamlined workflow: reduces the need for task-specific fine-tuning. citeturn1search7  
- Can be more efficient for real-time applications—enhancing overall productivity. citeturn1search7  

### Model Temperature

Temperature is a parameter that controls the randomness of token selection during inference (commonly described as scaling logits before softmax). citeturn6search0  

Notes (preserved):
- Future tokens picked via probability distribution citeturn6search0  
- Higher probability = higher chance of selection citeturn6search0  
- Temperature scaling adjusts randomness in word selection. citeturn6search0  
- Tuning temperature balances precision and creativity in generated text. citeturn6search0  

### Challenges in Text Generation

#### Hallucinations
- Generates plausible but incorrect information
- Caused by missing or weak training signals citeturn13view0  

#### Bias
- Models inherit biases from training data
- Can reflect:
  - Gender bias
  - Cultural bias
  - Racial bias citeturn13view0  

#### Ethics
- Risks include:
  - Misinformation
  - Plagiarism
  - Copyright issues citeturn13view0  
- Potential misuse:
  - Fake news
  - Opinion manipulation citeturn13view0  
- Requires:
  - Better data curation
  - Bias mitigation
  - Strong ethical guidelines citeturn13view0  

## Multimodal and Generalist Models

### Multimodal models

Multimodal models are designed to process multiple types of data, moving beyond the traditional text based inputs. These models integrate different modalities such as text, image, audio, and video, allowing AI to understand and generate more contextually rich outputs. citeturn3search0turn3search1  

Instead of relying on a single data type, multimodal models combine multiple inputs to enhance their decision making and interpretation. To achieve this, separate architectures are often used for different data types. citeturn3search0turn3search1  

Examples from the notes:
- ViLBERT – 2 separate models for text and videos citeturn3search0  
- Show & Tell – CNN based model for images, LSTM for text captioning citeturn3search1  

Clarification (added to preserve intent while tightening accuracy):
- ViLBERT is a **two-stream** vision-and-language model with separate visual and textual streams that interact via co-attention; it is primarily presented for image+text settings but the same design pattern is often discussed in broader multimodal contexts. citeturn3search0turn3search4  

Looking at the illustration (as described in the notes), different components work together:
- CNN extracts features from an image
- A separate model such as BiLSTM processes corresponding textual data
- Outputs are then pooled or chained to form a final structured response, ensuring both image and text contribute to overall understanding citeturn3search1turn3search0  

This fusion enables tasks such as automatic captioning, video analysis, and even speech to text with contextual awareness. Multimodal learning enhances AI's ability to interpret the world more like humans by integrating multiple sensory inputs. citeturn3search0turn3search1  

Applications mentioned:
- Autonomous systems
- Accessibility tools
- Interactive AI assistance citeturn3search0turn3search1  

### Gato

GATO, developed by DeepMind in 2022, is a generalist deep neural network capable of handling text, images, video, and robotic control within a single Transformer architecture. citeturn2search3turn2search7  

Unlike traditional multimodal models, GATO does not use separate CNNs or LSTMs; instead, it tokenises all inputs into a shared format, treating different modalities as a sequence. citeturn2search3turn2search7  

This unified approach allows the model to handle diverse tasks without needing specialised architecture for each modality. citeturn2search3turn2search7  

GATO has been trained across a wide range of applications, from chat bots and gaming to robotic control, demonstrating its adaptability. citeturn2search3turn2search7  

Its versatility represents a major shift from specialised AI systems towards scalable generalist AI models that can efficiently operate across multiple domains. citeturn2search3turn2search7  


## Useful Links
- https://paperswithcode.com/method/gpt
- https://30dayscoding.com/blog/understanding-the-architecture-of-gpt-models
- https://arxiv.org/abs/1810.04805
- https://huggingface.co/docs/transformers/en/model_doc/distilbert
- https://github.com/huggingface/transformers
- https://huggingface.co/docs/transformers/en/model_doc/bart

### Visualise Deep Learning Models

```text
https://projector.tensorflow.org/
https://adamharley.com/nn_vis/cnn/3d.html
```
