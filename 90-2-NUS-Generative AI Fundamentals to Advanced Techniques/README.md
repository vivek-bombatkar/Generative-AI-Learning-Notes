# Generative AI Learning Notes
### My Learning Notes from [National University of Singapore generative-ai-fundamentals-to-advanced-techniques-programme](https://nus.comp.emeritus.org/generative-ai-fundamentals-to-advanced-techniques-programme)

## Table of Contents
- [Learning Paradigms in Machine Learning](#learning-paradigms-in-machine-learning)
- [From Brains to Artificial Neural Networks](#from-brains-to-artificial-neural-networks)
- [Convolutional Neural Networks](#convolutional-neural-networks)
- [Transformers and Attention Mechanisms](#transformers-and-attention-mechanisms)
- [Transformer Model Families](#transformer-model-families)
- [Alignment, Reliability, and Knowledge Grounding](#alignment-reliability-and-knowledge-grounding)
- [Multimodal and Generalist Models](#multimodal-and-generalist-models)
- [Useful Links](#useful-links)



# Learning Paradigms in Machine Learning

## Supervised vs Reinforcement vs Unsupervised Learning

| Aspect | Supervised Learning | Reinforcement Learning (RL) | Unsupervised Learning |
|--------|--------------------|-----------------------------|-----------------------|
| Definition | Learns from labelled input-output pairs with direct error feedback | Agent learns by interacting with an environment using rewards and penalties | Learns patterns and structures from unlabelled data |
| Data Type | Labelled data | Interaction data (state, action, reward) | Unlabelled data |
| Objective | Minimize prediction error (loss function) | Maximize cumulative long-term reward | Discover hidden structure or relationships |
| Feedback Signal | Immediate and explicit (true label known) | Delayed and reward-based | No explicit feedback |
| Training Process | Train model on labelled data and optimize loss | Agent acts, receives reward, updates policy iteratively | Model identifies clusters, representations, or distributions |
| Decision Nature | Predictive | Sequential decision-making | Exploratory pattern discovery |
| Typical Algorithms | Linear Regression, Logistic Regression, Neural Networks | Q-Learning, Policy Gradient, Deep RL | K-Means, PCA, Autoencoders, GANs |
| Example Use Cases | Image classification, Spam detection, Price prediction | Game playing, Robotics control, Recommendation with delayed rewards | Customer segmentation, Fraud detection, Dimensionality reduction |


# From Brains to Artificial Neural Networks

## The Human Brain and Neural Complexity

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

## Neural Networks and Artificial Neural Networks

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
| Aspect | Biological Neurons | Artificial Neurons |
|--------|-------------------|--------------------|
| Synapses / Inputs | Biological neurons have **synaptic gaps** of varying strengths. | Artificial neurons replace synapses with **numerical inputs**. |
| Connection Target | Synapses connect to the **soma (cell body)**. | Inputs can come from other neurons, sensors, data features, or variables. |
| Signal Processing | Signal strength depends on synaptic weight and connectivity. | Core operations include weighted sum (Sigma) and activation or threshold function. |
| Information Flow | Dendrites (input) → Cell body (integration) → Axon (signal transmission) → Axon terminals (output). | Inputs are aggregated using weighted sum, then passed through an activation function to produce output. |


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

## Artificial Neural Networks. Why?

| Feature | Description |
|----------|-------------|
| Extremely powerful computational devices | Turing-equivalent universal computers |
| Massive parallelism | Many simple units operate simultaneously, making computation efficient |
| Learning and generalization | Learn directly from training data. No need for carefully handcrafted rules or designs |
| Fault-tolerant and noise-tolerant | Performance degrades gracefully even with imperfect data or failures |
| Beyond symbolic systems | Can do everything a symbolic or logic-based system can, and more |
| Excellent with unstructured data | Particularly strong with text, images, audio, and other semi-structured data |


# Convolutional Neural Networks

### Deep Convolutional Neural Networks (CNNs)

| Section | Description |
|----------|------------|
| **Definition** | Specialized neural networks designed for structured, grid-like data such as images |
| **Primary Strength** | Efficient processing of spatial data with preserved local relationships |
| **Inspiration** | Modeled after the human visual system |


### Key Characteristics

| Characteristic | Explanation |
|----------------|-------------|
| Spatial Awareness | Designed for data arranged in 2D or multi-dimensional grids |
| Feature Extraction | Uses convolutional layers to automatically learn patterns |
| Decision Making | Fully connected layers perform classification or regression |
| Hierarchical Learning | Learns features from simple edges to complex structures |


### Core Building Blocks

| Component | Role |
|------------|------|
| Input | Raw image or grid-structured data |
| Convolutional Layer | Extracts local features such as edges and textures |
| Pooling Layer | Reduces spatial dimensions and computational cost |
| Activation Function | Introduces non-linearity (e.g., ReLU) |
| Fully Connected Layer | Combines features for final prediction |
| Output | Final classification or regression result |


### Why CNNs Work Well

| Reason | Explanation |
|---------|-------------|
| Hierarchical Feature Learning | Builds complex representations from simple patterns |
| Spatial Relationship Preservation | Maintains locality information within images |
| Parameter Efficiency | Shares weights across spatial regions |


### Typical Use Cases

| Application Area |
|------------------|
| Image classification |
| Object detection |
| Image segmentation |
| Visual pattern recognition |


#### Advantages of CNNs
| Advantage | Description |
|------------|-------------|
| Automatic Feature Extraction | CNNs learn features directly from raw data. No manual feature engineering required. |
| Parameter Sharing | Same filters are reused across the image. Fewer parameters than fully connected networks. |
| Translation Invariance | Recognize patterns such as edges, shapes, and objects regardless of position in the image. |
| Efficient for High-Dimensional Data | Scales well to large images and datasets. |
| State-of-the-Art Performance | Top results in image classification, object detection, and image segmentation. |
| Adaptability to Diverse Domains | Can be applied to images, audio spectrograms, and time-series data. Requires minimal architectural changes. |


#### Limitations of CNNs
| Limitation | Description |
|-------------|-------------|
| Computationally Intensive | Training requires powerful hardware such as GPUs or TPUs. |
| Data Hungry | Needs large labelled datasets for good performance. Data collection and annotation can be expensive. |
| Lack of Interpretability | Acts as a black-box model. Difficult to understand or debug decisions. |
| Overfitting Risk | Without proper regularisation, models may memorise training data. |
| Sensitivity to Hyperparameters | Performance depends heavily on architecture, learning rate, and other tuning choices. |
``


# Transformers and Attention Mechanisms

### Attention Mechanism in Transformers

#### Attention
At the core of modern NLP lies **attention**.  
It allows models to **focus**, not memorise.  
Instead of treating all words equally, attention helps the model decide **what matters most** in context.

A commonly used formulation is scaled dot-product attention:  
**Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V**.

#### Query, Key, and Value (QKV)

To understand attention, everything revolves around three components:

| Component | Description |
|------------|-------------|
| Query (Q) | Represents the current word or token. Think of it as a word asking for relevant context. |
| Key (K) | Represents all words in the input sequence. Each word has a key used to measure relevance to the query. |
| Value (V) | Contains the actual information or embeddings. This is the content passed forward once relevance is determined. |


#### How Attention Works (Step-by-Step)

| Step | Operation | What Happens | Purpose |
|------|-----------|--------------|----------|
| 1 | Compute Attention Scores | Dot product between **Query (Q)** and all **Keys (K)** | Measure relevance between current token and others |
| 2 | Scale Scores | Divide scores by **√dₖ** | Maintain numerical stability and prevent large gradients |
| 3 | Apply Softmax | Convert scaled scores into probabilities | Create attention weights that sum to 1 |
| 4 | Weighted Sum of Values | Multiply attention weights with **Values (V)** and sum | Produce context-aware representation |


#### Why Attention Matters
- Focuses on **relevant words**, regardless of position
- Handles **long-range dependencies**
- Essential for understanding meaning in complex sentences

### Multi-Head Attention Overview

| Section | Description |
|----------|------------|
| **What It Is** | Uses multiple parallel attention heads instead of a single attention mechanism |
| **Core Idea** | Each head attends to different aspects or relationships within the sequence |
| **Input Processing** | Input embeddings are projected into multiple sets of **Q, K, V** vectors |
| **Attention Computation** | Each head computes scaled dot-product attention independently |
| **Output Aggregation** | Head outputs are concatenated and passed through a final linear layer |
| **Result** | Produces richer, more expressive, and context-aware representations |

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

## Transformer Architecture. Big Picture

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

## Transformer Variants Comparison

### Hybrid Architectures

| Variant | Core Idea | What Each Component Does | Benefits | Common Applications |
|----------|-----------|--------------------------|-----------|--------------------|
| **CNN + Transformer** | Combine local feature extraction with global context modeling | CNN captures local spatial features. Transformer captures long-range dependencies. | Strong performance in vision tasks. Efficient feature hierarchy learning. | Vision Transformers (ViTs), ResNet hybrids |
| **RNN + Transformer** | Combine sequential memory with global attention | RNN handles local sequential dependencies. Transformer models global context. | Improved modeling of temporal data. Better long-context reasoning. | Speech recognition, Time-series forecasting |

**Overall Benefits of Hybrid Models**
- Reduced computational cost  
- Better efficiency and scalability  
- Strong performance on complex tasks  

---

### Efficient Transformers for Long Sequences

| Model | Core Technique | How It Improves Efficiency | Best Use Cases |
|--------|----------------|---------------------------|----------------|
| **Longformer** | Local + global sparse attention | Reduces quadratic attention cost for long documents | Document-level QA, Summarisation |
| **Linformer** | Low-rank attention approximation | Projects keys and values to lower dimensions, reducing memory and compute | Long-sequence NLP tasks |
| **Reformer** | LSH attention + reversible layers | Uses locality-sensitive hashing for sparse attention and memory-efficient backpropagation | Memory-constrained long-sequence processing |


# Transformer Model Families

### BERT vs GPT vs T5 Comparison

| Model | Full Form | Architecture Type | Directionality | Core Training Objective | Best Suited For |
|-------|------------|------------------|----------------|------------------------|----------------|
| **BERT** | Bidirectional Encoder Representations from Transformers | Encoder-only | Bidirectional | Masked Language Modeling | Sentiment analysis, Text classification, Question answering |
| **GPT** | Generative Pre-trained Transformer | Decoder-only | Unidirectional (Autoregressive) | Next-token prediction | Conversational AI, Content generation, Code generation |
| **T5** | Text-to-Text Transfer Transformer | Encoder-Decoder | Bidirectional (Encoder) + Autoregressive (Decoder) | Text-to-text transformation | Translation, Summarisation, Question answering |


### BERT limitations and common variants

#### Limitations of BERT (encoder-only)
- **Context window limit:** BERT is commonly pre-trained with an input context window up to **512 tokens**.
- **Dimensionality cost:** common configurations include hidden sizes like **768** (BERT-base), which affects memory and compute.
- **No true free-form generation:** BERT is trained with a masked prediction objective (and NSP in the original version), so it excels at understanding/representation learning rather than left-to-right generation.

### BERT Variants Comparison

| Model | Key Idea | Training Modification | Primary Benefit | Typical Use Cases |
|--------|----------|----------------------|------------------|-------------------|
| **RoBERTa** | Robustly Optimized BERT Approach | Removes NSP, uses dynamic masking, larger data and compute | Stronger performance through improved pretraining strategy | Text classification, QA, NLP benchmarks |
| **ELECTRA** | Efficient pretraining via discriminator | Replaced Token Detection instead of masked-token prediction | More sample-efficient training | General NLP tasks with lower compute cost |
| **DistilBERT** | Smaller distilled BERT | Knowledge distillation from larger BERT model | Faster, lighter, reduced memory usage | Real-time applications, edge deployment |
| **ALBERT** | Parameter-efficient BERT | Factorized embeddings, parameter sharing | Fewer parameters, improved efficiency | Large-scale NLP tasks with limited memory |
| **SpanBERT** | Span-focused masking | Masks contiguous spans and trains span-level objectives | Better span representations | Question answering, coreference resolution |
| **CodeBERT** | Bimodal NL-PL model | Pretrained on natural language and programming language data | Strong understanding of code + text | Code search, code generation, documentation tasks |


# GPT models and how ChatGPT works

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

| Stage | What Happens | Purpose |
|-------|-------------|----------|
| **Pre-training** | Trained on massive internet-scale text data. Learns to predict the **next token** and captures grammar, facts, and language patterns. | Build foundational language understanding and general knowledge. |
| **Fine-tuning** | Further trained using datasets reviewed by **human trainers** and feedback signals. | Improve helpfulness, safety, and alignment with human intent. |
| **Input Processing** | User input is **tokenised** into tokens (words or subwords) and converted into embeddings. | Convert raw text into numerical format the model can process. |
| **Contextual Understanding** | Transformer processes tokens using self-attention. Conversation history is included to model long-range dependencies. | Generate context-aware understanding of the input. |
| **Response Generation** | Model predicts next tokens sequentially based on learned probability distributions. | Produce coherent and relevant output text. |
| **Sampling and Optimisation** | Uses probabilistic sampling methods (e.g., temperature, top-k, top-p). Adds controlled randomness while applying safety filters. | Balance creativity, coherence, and safety. |
| **Post-processing** | Removes special tokens, applies formatting rules, and prepares final output. | Deliver clean, readable response to the user. |


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

### BART vs PEGASUS vs T5 Comparison

| Model | Full Form | Architecture | Pre-training Strategy | Masking / Corruption Method | Strengths | Best Use Cases |
|--------|------------|--------------|----------------------|-----------------------------|------------|----------------|
| **BART** | Bidirectional and Auto-Regressive Transformer | Encoder-Decoder | Denoising Autoencoder | Token masking, Token deletion, Span corruption (text infilling) | Combines bidirectional understanding with autoregressive generation. Robust to noisy or incomplete inputs. | Summarisation, Paraphrasing, Translation, Text completion |
| **PEGASUS** | Pre-training with Extracted Gap-sentences for Abstractive Summarisation | Encoder-Decoder | Gap Sentence Generation (GSG) | Masks entire important sentences instead of tokens | Optimised specifically for abstractive summarisation. Minimal fine-tuning needed. | Document summarisation |
| **T5** | Text-to-Text Transfer Transformer | Encoder-Decoder | Text-to-text unified framework | Span corruption with task prefixes | Unified approach for multiple NLP tasks via instruction-style prompts. Strong transfer learning capability. | Translation, Summarisation, Question Answering |


# Alignment, Reliability, and Knowledge Grounding

### GPT, RLHF, RAG, ZSL, and Temperature Comparison

| Concept | What It Is | Core Mechanism | Why It Matters | Limitations |
|----------|------------|---------------|----------------|-------------|
| **Reinforcement Learning (RL)** | Learning paradigm based on trial and error with rewards | Agent takes actions, receives rewards, optimises long-term return | Enables adaptive and goal-oriented behaviour | Requires well-defined reward signals |
| **RLHF** | Reinforcement Learning from Human Feedback applied to GPT | Uses a reward model trained on human preference rankings to fine-tune outputs | Improves helpfulness, safety, and alignment with human expectations | Still limited by training data and human bias |
| **RAG** | Retrieval-Augmented Generation | Retrieves relevant external documents and injects them into model context | Improves factual accuracy, reduces hallucinations, enables up-to-date answers | Depends on retrieval quality and document relevance |
| **Zero-Shot Learning (ZSL)** | Performing tasks without task-specific fine-tuning | Relies on pretraining knowledge and prompt instructions | Enables flexibility and fast deployment for new tasks | May lack task-specific precision |
| **Temperature** | Inference-time parameter controlling randomness | Scales logits before softmax to adjust probability distribution | Balances creativity vs determinism in generated text | Too high reduces coherence, too low reduces diversity |


### Challenges in Text Generation

| Challenge | Description | Causes | Risks | Mitigation |
|------------|------------|--------|-------|------------|
| **Hallucinations** | Model generates plausible but factually incorrect information | Weak or missing training signals, probabilistic next-token prediction | Misinformation, reduced trust, incorrect decisions | Better training data, retrieval grounding (RAG), human feedback, fact-checking layers |
| **Bias** | Model reflects biases present in training data | Skewed datasets, historical or societal imbalances | Gender bias, Cultural bias, Racial bias, unfair outcomes | Bias-aware data curation, fairness evaluation, alignment tuning |
| **Ethics** | Broader societal and legal concerns around model outputs | Scale of deployment, content generation capability | Misinformation, Plagiarism, Copyright violations, Fake news, Opinion manipulation | Strong ethical guidelines, safety alignment, policy enforcement, responsible AI governance |


# Multimodal and Generalist Models

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
- https://projector.tensorflow.org/
- https://adamharley.com/nn_vis/cnn/3d.html
