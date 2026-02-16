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

| Aspect | Supervised Learning | Reinforcement Learning (RL) |
|---|---|---|
| Definition | A learning paradigm where a model is trained on labelled input-output pairs and receives direct feedback on errors. | A learning paradigm where an agent learns by interacting with an environment using rewards and penalties. |
| Training steps | 1. Collect labelled data.<br>2. Split into training and validation sets.<br>3. Train model to minimize loss.<br>4. Evaluate performance. | 1. Initialize agent and environment.<br>2. Perform actions based on current policy.<br>3. Receive reward feedback.<br>4. Update policy using reward signal.<br>5. Repeat until convergence. |
| Examples | Image classification, spam detection, price prediction. | Game playing, robotics control, recommendation strategies with delayed rewards. |


### Unsupervised Learning

- **Unsupervised Learning:** A branch of machine learning where models learn patterns and relationships from unlabelled data without predefined outputs.

#### Examples

##### Clustering Problems
- Customer segmentation
- Image segmentation

##### Dimensionality Reduction Problems
- Principal Component Analysis (PCA)
- t-SNE

##### Anomaly Detection
- Fraud detection

##### Generative Models (partly unsupervised)
- Autoencoders
- GANs (Generative Adversarial Networks)

##### Market Basket Analysis
- Identifying frequently co-occurring items

## From Brains to Artificial Neural Networks

### The Human Brain and Neural Complexity

#### Overview
The human brain is an incredibly complex organ and one of nature's greatest engineering marvels.  
The **neocortex** plays a key role in higher cognitive functions such as:
- Reasoning
- Perception
- Decision-making
- Language
Its massive connectivity gives the brain extraordinary computational power.

#### Neurons and Synapses
- The brain contains around **100 billion neurons**
- Each neuron connects to **1,000-10,000** other neurons through **synapses**
- Synapses transmit information via electrical and chemical signals
- The neocortex alone is estimated to have about **500 trillion synapses**, forming a massive biological network capable of learning and thought

> Note (clarification added): Published estimates vary by method and definition; widely cited totals include ~86 billion neurons for the whole brain and on the order of 10^14 synapses overall, with neocortex synapse counts reported in the ~10^14 range as well.

#### Neural Wiring and Efficiency
- The neocortex contains roughly **300 million feet (~91,440 km)** of neural wiring
- This wiring is compacted into a volume of about **1.5 quarts (~1.4 liters)**
- Such efficiency is achieved through several biological optimizations:

##### Folding of the Cortex
- The brain surface is folded into **gyri** (ridges) and **sulci** (grooves)
- Folding increases surface area without increasing overall volume

##### Myelination
- Axons are coated with **myelin**, a fatty insulating layer
- Myelin speeds up signal transmission and reduces energy usage

##### Specialized Networks
- The brain is organized into specialized functional areas, such as:
  - Visual cortex
  - Motor cortex
  - Prefrontal cortex
- Specialization minimizes unnecessary wiring and improves processing speed

### Neural Networks and Artificial Neural Networks

- Neural networks are inspired by biological brains.
- Artificial neurons approximate real neurons.
- ANNs are networks of artificial neurons.
- ANNs are simplified models of brain functionality.
- Practically, ANNs are parallel computational systems.

#### Definitions
- **Neural Networks (NNs):** Networks of neurons similar to those found in biological brains.
- **Artificial Neurons:** Crude approximations of biological neurons, implemented as mathematical or software constructs.
- **Artificial Neural Networks (ANNs):** Networks of artificial neurons that approximate certain functions of real brains.

### Biological vs Artificial Neurons

#### Biological Neurons
- Biological neurons have **synaptic gaps** of varying strengths
- These synapses connect to the **soma (cell body)**
- Signal strength depends on synaptic weight and connectivity
- Information flows via:
  - Dendrites (input)
  - Cell body (integration)
  - Axon (signal transmission)
  - Axon terminals (output)

#### Artificial Neurons
- Artificial neurons replace synapses with **numerical inputs**
- Inputs can come from:
  - Other neurons
  - Sensors
  - Data features
  - Variables
- Core operations include:
  - Weighted sum (Sigma)
  - Activation / threshold function

#### Computational Power
- The neocortex contains about **500 trillion synapses** operating **in parallel**
- Enables massive information processing and storage simultaneously
- The human brain operates on roughly **20 watts of power**
- This is far more **energy-efficient** than modern supercomputers

#### Implications for Intelligence and Learning
- High neuron density and interconnectivity enable human intelligence
- **Plasticity** allows neural connections to reorganize with:
  - Learning
  - Experience
  - Recovery from injury
- This adaptability is central to skill acquisition and cognition

#### Artificial Neural Networks (ANNs)
- ANNs are inspired by biological neural systems
- They aim to approximate learning and decision-making
- While powerful, they are far less energy-efficient than the human brain
- The neocortex remains a benchmark for efficient computation and learning

### Artificial Neural Networks. Why?

- **Extremely powerful computational devices**  
  - Turing-equivalent universal computers
- **Massive parallelism**  
  - Many simple units operate simultaneously, making computation efficient
- **Learning and generalization**  
  - Learn directly from training data  
  - No need for carefully handcrafted rules or designs
- **Fault-tolerant and noise-tolerant**  
  - Performance degrades gracefully even with imperfect data or failures
- **Beyond symbolic systems**  
  - Can do everything a symbolic or logic-based system can, and more
- **Excellent with unstructured data**  
  - Particularly strong with:
    - Text
    - Images
    - Audio
    - Other semi-structured data

## Convolutional Neural Networks

### Deep Convolutional Neural Networks (CNNs)

#### Definition
**Deep Convolutional Neural Networks (CNNs)** are a specialized type of neural network designed to process **structured, grid-like data**, especially **images**.

#### Key characteristics
- Designed for **spatial data** arranged in grids
- Combine:
  - **Convolutional layers** for feature extraction
  - **Fully connected layers** for decision-making
- Inspired by the **human visual system**
- Highly effective when **spatial hierarchies** matter

#### Core building blocks
- **Input**
- **Convolutional Layer**. Extracts local features (edges, textures)
- **Pooling Layer**. Reduces spatial size and computation
- **Activation Function**. Adds non-linearity
- **Fully Connected Layer**. Performs classification or prediction
- **Output**

#### Why CNNs work well
- Learn **hierarchical features**. from simple edges to complex shapes
- Preserve **spatial relationships** in data
- Particularly strong for **vision-based tasks**

#### Typical use cases
- Image classification
- Object detection
- Image segmentation
- Visual pattern recognition

#### Advantages of CNNs
- **Automatic Feature Extraction**  
  - CNNs learn features directly from raw data  
  - No manual feature engineering required
- **Parameter Sharing**  
  - Same filters are reused across the image  
  - Fewer parameters than fully connected networks
- **Translation Invariance**  
  - Recognize patterns (edges, shapes, objects) regardless of position in the image
- **Efficient for High-Dimensional Data**  
  - Scales well to large images and datasets
- **State-of-the-Art Performance**  
  - Top results in:
    - Image classification
    - Object detection
    - Image segmentation
- **Adaptability to Diverse Domains**  
  - Can be applied to:
    - Images
    - Audio spectrograms
    - Time-series data  
  - Requires minimal architectural changes

#### Limitations of CNNs
- **Computationally Intensive**  
  - Training requires powerful hardware (GPUs, TPUs)
- **Data Hungry**  
  - Needs large labelled datasets for good performance  
  - Data collection and annotation can be expensive
- **Lack of Interpretability**  
  - Acts as a black-box model  
  - Difficult to understand or debug decisions
- **Overfitting Risk**  
  - Without proper regularisation, models may memorise training data
- **Sensitivity to Hyperparameters**  
  - Performance depends heavily on:
    - Architecture
    - Learning rate
    - Other tuning choices

## Transformers and Attention Mechanisms

### Attention Mechanism in Transformers

#### Attention
At the core of modern NLP lies **attention**.  
It allows models to **focus**, not memorise.  
Instead of treating all words equally, attention helps the model decide **what matters most** in context.

A commonly used formulation is scaled dot-product attention:  
**Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V**.

#### Query, Key, and Value (QKV)

To understand attention, everything revolves around three components:

- **Query (Q)**  
  - Represents the **current word or token**
  - Think of it as a word *asking* for relevant context

- **Key (K)**  
  - Represents **all words in the input sequence**
  - Each word has a key used to measure relevance to the query

- **Value (V)**  
  - Contains the **actual information** (embeddings)
  - This is the content passed forward once relevance is determined

#### How Attention Works (Step-by-Step)

- Compute **attention scores** using dot products between Query and Keys.
- Scale scores by **sqrt(d_k)** for stability.
- Apply **softmax** to get a probability distribution over tokens.
- Take the **weighted sum of Values** to produce a context-aware representation.

#### Why Attention Matters
- Focuses on **relevant words**, regardless of position
- Handles **long-range dependencies**
- Essential for understanding meaning in complex sentences

### Multi-Head Attention

#### What is Multi-Head Attention?
- Instead of one attention mechanism, the model uses **multiple heads**
- Each head attends to **different aspects** of the sequence

#### How It Works
- Input is split into multiple Q, K, V sets
- Each head computes attention **independently**
- Outputs are:
  - Concatenated
  - Passed through a linear layer  
Result. richer and more expressive representations

### Feedforward Networks (FFN)

#### Role of Feedforward Networks
- Applied **after multi-head attention**
- Operates **independently on each position**

#### Structure
- Linear layer
- ReLU activation
- Linear layer

#### Purpose
- Captures **more abstract patterns**
- Refines representations beyond word-to-word relationships
- Output is passed to the next transformer layer

### Transformer Architecture. Big Picture

- Embeddings
- Self-attention (QKV)
- Multi-head attention
- Feedforward networks
- Stacked layers for depth

Together, these components allow transformers to:
- Understand context deeply
- Scale efficiently
- Generate high-quality, human-like text

### Training a Transformer
- Data preprocessing. tokenisation and embeddings
- Positional encodings added for sequence order
- Random weight initialisation
- Training via **backpropagation**
- Hyperparameters tuned:
  - Learning rate
  - Batch size
- Goal. minimise loss and optimise performance

### Computational Challenges
- Large models are **computationally intensive**
- Example:
  - GPT-3 has **175 billion parameters**
- CPUs are insufficient for training
- Requires:
  - GPUs
  - TPUs
- Computational needs have grown **exponentially**
- Hardware advances are critical for progress

### Transformer Variants
- Transformers excel at sequential data
- Variants optimise performance for specific tasks

#### Hybrid Architectures

##### CNN + Transformer
- Common in computer vision
- CNN captures **local spatial features**
- Transformer captures **long-range dependencies**
- Used in:
  - Vision Transformers (ViTs)
  - ResNet hybrids

##### RNN + Transformer
- Useful for:
  - Speech recognition
  - Time-series forecasting
- RNN handles local sequences
- Transformer handles global context

##### Benefits
- Reduced computational cost
- Better efficiency and scalability
- Strong performance on complex tasks

#### Efficient Transformers (Long Sequences)
- Standard transformers scale **quadratically** with sequence length due to self-attention
- Problematic for long documents
- Sparse and efficient variants address this

##### Longformer
- Uses attention patterns combining local + global information
- Scales more linearly for long documents
- Ideal for:
  - Document-level QA
  - Summarisation

##### Linformer
- Approximates attention via low-rank structure
- Reduces memory and inference cost for long sequences

##### Reformer
- Optimised for memory efficiency
- Uses locality-sensitive hashing (LSH) attention and reversible layers

## Transformer Model Families

### BERT vs GPT vs T5

#### BERT
- **Bidirectional Encoder Representations from Transformers**
- Understands text in **both directions**
- Best suited for:
  - Sentiment analysis
  - Text classification
  - Question answering

#### GPT
- **Generative Pre-trained Transformer**
- **Autoregressive**. predicts the next word in a sequence
- Ideal for:
  - Conversational AI
  - Content generation
  - Code generation

#### T5
- **Text-to-Text Transfer Transformer**
- Treats **all NLP tasks as text-to-text**
- Highly versatile for:
  - Translation
  - Summarisation
  - Question answering

### BERT limitations and common variants

#### Limitations of BERT (encoder-only)
- **Context window limit:** BERT is commonly pre-trained with an input context window up to **512 tokens**.
- **Dimensionality cost:** common configurations include hidden sizes like **768** (BERT-base), which affects memory and compute.
- **No true free-form generation:** BERT is trained with a masked prediction objective (and NSP in the original version), so it excels at understanding/representation learning rather than left-to-right generation.

#### Variants mentioned in the notes and discussion
- **RoBERTa:** improved BERT pretraining by changing key training choices (for example, removing NSP and using dynamic masking and more data/compute).
- **ELECTRA:** replaces masked-token prediction with **replaced token detection** (discriminator predicts whether a token was replaced by a generator).
- **DistilBERT:** uses knowledge distillation to create a smaller, faster model while preserving much of BERT’s performance.
- **ALBERT:** reduces parameters (e.g., factorized embeddings and parameter sharing) to improve efficiency.
- **SpanBERT:** masks contiguous spans and trains objectives tailored to span representations.
- **CodeBERT:** pre-trained for natural language + programming language tasks (bimodal NL-PL).

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
- GPT-2 (1.5B) and GPT-3 (175B) parameter counts are documented in their respective technical reports.
- Some commonly repeated values for “GPT-3.5” and “GPT-4 parameter count” are **not officially disclosed**. The GPT-4 technical report explicitly states it does not provide details such as model size.

#### How ChatGPT Works (End-to-End Flow)

**Pre-training**  
- Trained on massive internet text
- Learns to predict the **next token**
- Captures grammar, facts, and language patterns

**Fine-tuning**  
- Uses datasets reviewed by **human trainers**
- Learns to generate safer and more helpful responses
- Generalises from human feedback

**Input Processing**  
- User input is **tokenised** into words or subwords
- Tokens are fed into the transformer model

**Contextual Understanding**  
- Maintains conversation history
- Transformer architecture models long-range dependencies
- Enables context-aware responses

**Response Generation**  
- Predicts next tokens based on learned patterns
- Produces coherent, human-like text

**Sampling and Optimisation**  
- Uses probabilistic sampling
- Adds controlled randomness for natural responses
- Safety techniques reduce harmful outputs

**Post-processing**  
- Removes special tokens and formatting
- Final response is shown to the user

### Encoder-decoder models and key examples

#### Encoder-decoder models
Encoder-decoder models are a fundamental architecture in modern deep learning. These models bring together an encoder and a decoder, enabling efficient processing of input data while generating meaningful output.

- Start with the encoder, which takes the input and processes it to capture its contextual meaning. It transforms the data into a structured representation that the model can understand.
- Then comes the decoder, which uses that structured information to generate text that is natural and coherent. Unlike simple text generation models that produce output sequentially, encoder-decoder models maintain logical consistency by referencing the complete context provided by the encoder.

#### Applications of encoder-decoder models
Encoder-decoder models power a variety of real-world applications including:
- Machine translation to convert text from one language to another
- Text summarisation, extracting the key points while preserving the meaning
- Caption generation, generating textual descriptions for images or videos

#### BART
BART (Bidirectional and Auto-Regressive Transformer) is a hybrid model that integrates the strengths of both BERT and GPT, making it highly effective for tasks requiring text reconstruction and controlled generation.

- It employs a bidirectional encoding process (similar to BERT) for comprehensive contextual understanding.
- It uses denoising objectives such as span corruption / text infilling (replacing spans with a single mask token), forcing reconstruction of phrases while maintaining coherence.
- On the decoding side, BART adopts autoregressive generation (similar to GPT), where tokens are generated one at a time while conditioning on the encoded input.

##### BART’s Denoising Process
BART employs a denoising autoencoder approach, where input data is deliberately corrupted before being passed through the model.

Noise-insertion techniques described in the notes:
- Token masking replaces random words with a special [MASK] token.
- Token deletion removes entire words from the sequence.
- Text infilling (span corruption) replaces entire spans of text with a single [MASK] token.

This training strategy makes BART robust to noisy or incomplete inputs (e.g., imperfect formatting or missing spans) and supports tasks like summarisation, paraphrasing, translation, and text completion.

#### PEGASUS
PEGASUS (Pre-training with Extracted Gap-sentences for Abstractive Summarisation) is specifically designed to enhance text summarisation by masking entire sentences rather than individual tokens during pretraining.

Key points from the notes:
- Entire sentences (instead of tokens) are masked during pretraining.
- Sentences may be randomly identified, though the method focuses on removing “important” sentences (gap sentence generation).
- These are identified to be ones with high similarity to the rest of the document (intuitively encouraging summary-like targets).

PEGASUS achieves strong summarisation performance and can require minimal fine-tuning for high-quality abstractive summaries.

#### T5
T5 (Text-to-Text Transfer Transformer) reformulates all NLP tasks into a text-to-text format, using an encoder-decoder architecture for tasks like translation, summarisation, and Q&A.

Text to Text
- Indicative of the types of input and output to be expected
- Encoder model used to get information for input text
- Decoder model used to generate output text

Transfer Transformer
- Transformer capable of employing transfer learning
- Allows for multiple NLP tasks to be accomplished by the model:
  - Translation
  - Summarisation
  - Q&A

T5 uses task prefixes (instructions) to unify workflows across tasks (e.g., “translate ...”, “summarize ...”).

## Alignment, Reliability, and Knowledge Grounding

### GPT and Reinforcement Learning

GPT models, built on deep learning, have revolutionised language understanding and generation by predicting text patterns with remarkable fluency. Reinforcement Learning (RL), on the other hand, empowers systems to learn through trial and error, optimising actions for long-term rewards. Together, they unlock new frontiers in adaptive, intelligent decision-making and human-like interactions.

### Reinforcement Learning from Human Feedback (RLHF)

Reinforcement Learning from Human Feedback (RLHF) enhances GPT’s ability to generate not just human-like text but also reliable and contextually appropriate responses.

Why RLHF is used (as captured in the notes and discussion):
- GPT can produce fluent language, but a pure Transformer architecture does not *inherently* verify factual accuracy or suitability of outputs.
- Example (preserved): “Under Augustus, the Roman Empire came to [MASK]” — GPT alone may not “know” which completion is historically correct without grounding or reliable internal knowledge.

Typical RLHF-style pipeline:
- A model is fine-tuned using a **reward model** (often a transformer trained on human preference rankings) that prioritizes more useful outputs.
- Human reviewers evaluate and correct a subset of responses, reinforcing high-quality and informative text generation.

### Retrieval-Augmented Generation (RAG)

Retrieval-Augmented Generation (RAG) enhances GPT by integrating external knowledge sources, addressing its limitations with fixed training data.

Key points (preserved and clarified):
- ChatGPT generates responses off fixed training data.
- Without external sources, it struggles with real-time updates, niche topics, or retrieving specific factual details.
- RAG retrieves relevant documents or data (e.g., knowledge bases, web snapshots, or indexed corpora), feeding them into the model as additional context for response generation.
- This approach improves accuracy, reduces hallucinations, and enables up-to-date, domain-specific answers.
- Combining GPT-style models with retrieval creates more reliable, informed, and adaptable AI systems for business and research applications.

### Zero-Shot Learning (ZSL)

Zero-Shot Learning (ZSL) enables GPT to perform tasks without additional training, relying solely on its extensive pretraining. Instead of requiring labeled examples or fine-tuning:

- GPT leverages its extensive pre-training to perform tasks without additional training, generating relevant outputs directly from prompt instruction.
- Enhanced flexibility: allows adaptation to new tasks without extra fine-tuning.
- Streamlined workflow: reduces the need for task-specific fine-tuning.
- Can be more efficient for real-time applications—enhancing overall productivity.

### Model Temperature

Temperature is a parameter that controls the randomness of token selection during inference (commonly described as scaling logits before softmax).

Notes (preserved):
- Future tokens picked via probability distribution
- Higher probability = higher chance of selection
- Temperature scaling adjusts randomness in word selection.
- Tuning temperature balances precision and creativity in generated text.

### Challenges in Text Generation

#### Hallucinations
- Generates plausible but incorrect information
- Caused by missing or weak training signals

#### Bias
- Models inherit biases from training data
- Can reflect:
  - Gender bias
  - Cultural bias
  - Racial bias

#### Ethics
- Risks include:
  - Misinformation
  - Plagiarism
  - Copyright issues
- Potential misuse:
  - Fake news
  - Opinion manipulation
- Requires:
  - Better data curation
  - Bias mitigation
  - Strong ethical guidelines

## Multimodal and Generalist Models

### Multimodal models

Multimodal models are designed to process multiple types of data, moving beyond the traditional text based inputs. These models integrate different modalities such as text, image, audio, and video, allowing AI to understand and generate more contextually rich outputs.

Instead of relying on a single data type, multimodal models combine multiple inputs to enhance their decision making and interpretation. To achieve this, separate architectures are often used for different data types.

Examples from the notes:
- ViLBERT - 2 separate models for text and videos
- Show & Tell - CNN based model for images, LSTM for text captioning

Clarification (added to preserve intent while tightening accuracy):
- ViLBERT is a **two-stream** vision-and-language model with separate visual and textual streams that interact via co-attention; it is primarily presented for image+text settings but the same design pattern is often discussed in broader multimodal contexts.

Looking at the illustration (as described in the notes), different components work together:
- CNN extracts features from an image
- A separate model such as BiLSTM processes corresponding textual data
- Outputs are then pooled or chained to form a final structured response, ensuring both image and text contribute to overall understanding

This fusion enables tasks such as automatic captioning, video analysis, and even speech to text with contextual awareness. Multimodal learning enhances AI's ability to interpret the world more like humans by integrating multiple sensory inputs.

Applications mentioned:
- Autonomous systems
- Accessibility tools
- Interactive AI assistance

### Gato

GATO, developed by DeepMind in 2022, is a generalist deep neural network capable of handling text, images, video, and robotic control within a single Transformer architecture.

Unlike traditional multimodal models, GATO does not use separate CNNs or LSTMs; instead, it tokenises all inputs into a shared format, treating different modalities as a sequence.

This unified approach allows the model to handle diverse tasks without needing specialised architecture for each modality.

GATO has been trained across a wide range of applications, from chat bots and gaming to robotic control, demonstrating its adaptability.

Its versatility represents a major shift from specialised AI systems towards scalable generalist AI models that can efficiently operate across multiple domains.


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
