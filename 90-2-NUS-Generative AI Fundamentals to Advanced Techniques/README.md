# Generative AI Learning Notes
### My Learning Notes from [National University of Singapore | Generative AI: Fundamentals to Advanced Techniques](https://nus.comp.emeritus.org/generative-ai-fundamentals-to-advanced-techniques-programme)

---
## Table of Contents
- [Learning Paradigms in Machine Learning](#learning-paradigms-in-machine-learning)
- [From Brains to Artificial Neural Networks](#from-brains-to-artificial-neural-networks)
- [Convolutional Neural Networks](#convolutional-neural-networks)
- [Transformers and Attention Mechanisms](#transformers-and-attention-mechanisms)
- [Transformer Model Families](#transformer-model-families)
- [Alignment, Reliability, and Knowledge Grounding](#alignment-reliability-and-knowledge-grounding)
- [Multimodal and Generalist Models](#multimodal-and-generalist-models)
- [AI Learning Techniques](#ai-learning-techniques)
- [Reinforcement Learning for Generative AI](#Reinforcement-Learning-for-Generative-AI)
- 
- [Useful Links](#useful-links)
---

# Learning Paradigms in Machine Learning

## Supervised vs Unsupervised vs Reinforcement Learning

| Aspect | Supervised Learning | Unsupervised Learning | Reinforcement Learning (RL) |
|--------|--------------------|-----------------------|-----------------------------|
| Definition | Learns from labelled input-output pairs with direct error feedback | Learns patterns and structures from unlabelled data | Agent learns by interacting with an environment using rewards and penalties |
| Data Type | Labelled data | Unlabelled data | Interaction data (state, action, reward) |
| Objective | Minimize prediction error (loss function) | Discover hidden structure or relationships | Maximize cumulative long-term reward |
| Feedback Signal | Immediate and explicit (true label known) | No explicit feedback | Delayed and reward-based |
| Training Process | Train model on labelled data and optimize loss | Model identifies clusters, representations, or distributions | Agent acts, receives reward, updates policy iteratively |
| Decision Nature | Predictive | Exploratory pattern discovery | Sequential decision-making |
| Typical Algorithms | Linear Regression, Logistic Regression, Neural Networks | K-Means, PCA, Autoencoders, GANs | Q-Learning, Policy Gradient, Deep RL |
| Example Use Cases | Image classification, Spam detection, Price prediction | Customer segmentation, Fraud detection, Dimensionality reduction | Game playing, Robotics control, Recommendation with delayed rewards |


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

### Links:
https://pytorch.org/
https://machinelearningmastery.com/regression-tutorial-keras-deep-learning-library-python/
https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
https://keras.io/
https://www.tensorflow.org/tutorials/images/cnn
https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

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



### Links:
https://huggingface.co/docs/huggingface_hub/v0.23.1/quick-start
https://discuss.huggingface.co/t/google-colab-hub-login/21853
https://huggingface.co/docs/transformers/en/model_doc/bert
https://www.tensorflow.org/text/tutorials/classify_text_with_bert
https://www.geeksforgeeks.org/explanation-of-bert-model-nlp/
https://bert-embedding.readthedocs.io/en/latest/
https://platform.openai.com/docs/api-reference/introduction
https://platform.openai.com/docs/concepts
https://docs.gptr.dev/docs/examples/examples


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


### Link:
https://www.geeksforgeeks.org/explanation-of-bert-model-nlp/
https://paperswithcode.com/method/gpt
https://30dayscoding.com/blog/understanding-the-architecture-of-gpt-models
https://arxiv.org/abs/1810.04805
https://huggingface.co/docs/transformers/en/model_doc/distilbert
https://github.com/huggingface/transformers
https://huggingface.co/docs/transformers/en/model_doc/bart


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
## Multimodal Models Overview

| Aspect | Description |
|--------|------------|
| Definition | Models designed to process and integrate multiple data modalities such as text, images, audio, and video |
| Core Idea | Combine different input types to enhance contextual understanding and decision-making |
| Architecture Pattern | Often uses separate architectures for different modalities with a fusion mechanism |
| Goal | Produce richer, context-aware outputs by leveraging multiple sensory inputs |

---

## Architectural Design Pattern

| Component | Role |
|------------|------|
| CNN (Image Stream) | Extracts visual features from images |
| Text Model (e.g., BiLSTM) | Processes textual data and captures sequential context |
| Fusion Layer | Pools, concatenates, or applies co-attention across modalities |
| Final Output Layer | Generates structured output using combined multimodal features |

---

## Example Models

| Model | Architecture Style | Key Characteristic |
|--------|--------------------|--------------------|
| **ViLBERT** | Two-stream vision-language model | Separate visual and textual streams interacting via co-attention |
| **Show & Tell** | CNN + LSTM | CNN extracts image features, LSTM generates captions |

---

## Capabilities Enabled by Fusion

| Capability | Description |
|-------------|------------|
| Automatic Captioning | Generate text descriptions from images |
| Video Analysis | Interpret visual sequences with contextual reasoning |
| Speech-to-Text with Context | Combine audio input with language understanding |
| Cross-Modal Reasoning | Link information across image and text domains |

---

## Application Areas

| Domain | Examples |
|--------|----------|
| Autonomous Systems | Perception + decision integration |
| Accessibility Tools | Image description for visually impaired users |
| Interactive AI Assistants | Context-aware multimodal interactions |


### Gato

GATO, developed by DeepMind in 2022, is a generalist deep neural network capable of handling text, images, video, and robotic control within a single Transformer architecture.

Unlike traditional multimodal models, GATO does not use separate CNNs or LSTMs; instead, it tokenises all inputs into a shared format, treating different modalities as a sequence.

This unified approach allows the model to handle diverse tasks without needing specialised architecture for each modality.

GATO has been trained across a wide range of applications, from chat bots and gaming to robotic control, demonstrating its adaptability.

Its versatility represents a major shift from specialised AI systems towards scalable generalist AI models that can efficiently operate across multiple domains.


# AI Learning Techniques

## AI Learning Techniques Overview

| Category | Learning Type | Core Idea | Example Techniques | Typical Applications |
|-----------|--------------|-----------|--------------------|----------------------|
| **AI Fundamentals** | Supervised Learning | Learn from labelled examples (X, Y) | Classification, Regression | Spam detection, Price prediction, Image classification |
| **AI Fundamentals** | Unsupervised Learning | Learn structure without labels | Clustering, Anomaly Detection, Recommender Systems | Customer segmentation, Fraud detection |
| **AI Fundamentals** | Reinforcement Learning (RL) | Learn by interacting with environment using rewards | Policy Learning, Reward Optimization | Robotics, Autonomous systems, Game-playing AI |
| **Deep Learning** | Pre-Training | Large-scale training on broad datasets | Transformer Pretraining | Foundation models |
| **Deep Learning** | Post-Training (RLHF, Fine-tuning) | Improve alignment and performance | RLHF, Instruction Tuning | Chatbots, GenAI systems |
| **Generative AI (GenAI)** | Generative Modeling | Generate new content from learned patterns | GPT-style models | Content generation, Conversational AI |
| **Agentic AI** | High-Autonomy Systems | AI systems that act independently | Planning + Tool use + Memory | Autonomous agents, Decision systems |


## Learning Paradigms Comparison

| Learning Paradigm | Learns From | Feedback Type | Autonomy Level | Key Characteristic |
|-------------------|------------|---------------|----------------|--------------------|
| Supervised Learning | Labelled data (X, Y) | Direct error feedback | Low | Predictive modeling |
| Unsupervised Learning | Unlabelled data | No explicit feedback | Low | Pattern discovery |
| Reinforcement Learning | Environment interaction | Reward and penalty | High | Sequential decision-making |

## Generative AI and LLM Training Methods

| Aspect | Description |
|--------|------------|
| Generative AI Category | Does not fit into a single learning paradigm. Learns probability distributions underlying training data. |
| Learning Nature | Often considered unsupervised at core, as it models data distributions without explicit labels for every output. |
| LLM Pre-Training | Relies primarily on supervised-style next-token prediction over large datasets. |
| LLM Post-Training | Uses reinforcement learning techniques (e.g., RLHF) to improve alignment and helpfulness. |
| Alignment Objective | Post-training integrates Generative AI and RL to address safety, factuality, and alignment challenges. |
| Broader Goal | Moves AI systems toward more reliable, aligned, and general-purpose intelligence. |


## Reinforcement Learning from Human Feedback - RLHF and Reward Model in LLM Alignment

### Reward Model Concept

| Component | Description |
|------------|------------|
| Prompt | User input given to the model (e.g., “What color is milk?”) |
| Model Output | Generated response (e.g., “Milk is white.”) |
| Reward Model | Evaluates how good the answer is relative to the prompt |
| Score | Numerical value representing output quality and alignment |

---

## Why Post-Training is Needed

| Stage | Explanation |
|--------|------------|
| Pre-Training | Produces a base LLM trained on large datasets using next-token prediction |
| Limitation | Model lacks refinement, alignment, and may produce hallucinations |
| Traditional Fine-Tuning | Updates model weights using supervised datasets but is expensive for large models |
| Persistent Challenge | Even after fine-tuning, hallucinations and misalignment can remain |

---

## Reinforcement Learning from Human Feedback (RLHF)

| Aspect | Description |
|---------|------------|
| Objective | Align pre-trained model outputs with human values |
| Method | Human reviewers rank multiple model outputs |
| Reward Model Training | A separate model learns to predict human preference scores |
| Policy Optimization | Main LLM is updated to maximize reward scores |
| Outcome | More helpful, safer, and better-aligned responses |

---

# Reinforcement Learning for Generative AI


## Markov Decision Process (MDP) and Optimal Policy – Summary

An MDP is a mathematical framework for **sequential decision-making** where actions influence both immediate and future rewards.

| Concept | Role in RL |
|----------|-----------|
| MDP | Defines environment dynamics |
| Policy | Defines agent behavior |
| Bellman Equation | Links present and future values |
| Bootstrapping | Computes values iteratively |
| Optimal Policy | Maximizes long-term rewards |



Core Components of an MDP

| Component | Description |
|------------|------------|
| States (S) | Possible situations the agent can be in (e.g., Hotel, Area1–Area4 in the ski example) |
| Actions (A) | Choices available in each state (e.g., ski, take lift) |
| Rewards (R) | Immediate reward received after transitioning to a new state |
| Transition Probabilities (Pss’) | Probability of moving from state s to state s’ |
| Discount Factor (γ) | Controls importance of future rewards |

##  Policy

| Concept | Explanation |
|----------|------------|
| Policy (π) | A mapping from states to actions |
| Deterministic Policy | Always chooses the same action in a state |
| Stochastic Policy | Chooses actions with certain probabilities |

## Model-Based vs Model-Free Learning 

| Approach | Assumption | How It Works | Limitation |
|-----------|------------|--------------|------------|
| Model-Based (Dynamic Programming) | Full knowledge of transition probabilities (Pss’) | Computes optimal policy analytically | Unrealistic in real-world settings |
| Model-Free | No knowledge of transition probabilities | Learns directly through interaction | Requires exploration and experience |

Dynamic Programming assumes the environment is fully known (Page 1).  
Model-Free methods learn via trial and error.

## MC vs TD Comparison

| Feature | Monte Carlo | Temporal Difference |
|----------|-------------|--------------------|
| Requires Terminal State | Yes | No |
| Bootstrapping | No | Yes |
| Update Timing | After full episode | After every step |
| Bias | Low | Slightly higher |
| Variance | High | Lower |
| Practical Use | Episodic tasks | Continuous environments |

| Concept | Role in RL |
|----------|------------|
| Monte Carlo | Learn from full experience |
| Temporal Difference | Learn from partial experience |
| Bootstrapping | Core mechanism in TD |
| Exploration | Necessary for optimal learning |

## On-Policy vs Off-Policy Learning

| Concept | On-Policy | Off-Policy |
|-----------|------------|------------|
| Target Policy (π) | Policy being evaluated and improved | Policy being learned |
| Behaviour Policy (b) | Same as target policy | Different from target policy |
| Exploration | Limited to current policy | Can explore using separate exploratory policy |
| Example | SARSA | Q-Learning |

Q-Learning is **off-policy**.  
It learns the optimal policy regardless of how actions are chosen.

| Concept | Explanation |
|----------|------------|
| Q(s,a) | Value of taking action a in state s |
| Goal | Learn optimal Q-values |
| Policy Derived | Deterministic optimal policy |
| Key Feature | Uses max Q-value of next state |

Q-learning updates Q-values using:
- Current estimate
- Observed reward
- Best future estimated reward

| Algorithm | Type | Requires Model? |
|------------|------|----------------|
| Dynamic Programming | Model-based | Yes |
| Monte Carlo | Model-free | No |
| TD Learning | Model-free | No |
| Q-Learning | Model-free, Off-policy | No |

## RLHF in One Table

| Stage | What Happens |
|--------|-------------|
| Pre-training | Learn language patterns |
| Instruction tuning | Learn task-following behaviour |
| Reward model training | Learn human preference ranking |
| PPO fine-tuning | Align model to reward model |
| User feedback | Continuous improvement |

### RLHF =  Human preferences → Reward model → PPO updates → Aligned LLM

## Alignment Workflow Comparison

| Stage | RLHF | PPO | DPO | Constitutional AI |
|--------|------|------|------|------------------|
| 1. Pre-training | Base LLM trained | Same | Same | Same |
| 2. Instruction Tuning | Yes | Yes | Yes | Yes |
| 3. Human Feedback | Rank responses | Rank responses | Rank responses | Draft constitution |
| 4. Reward Model | Trained | Used | Not required | Not required |
| 5. Optimisation | RL with PPO | Clipped PPO objective | Direct preference loss | Self-evaluation against rules |
| 6. Policy Update | Via RL loop | Via RL loop | Via gradient descent | Via rule-guided refinement |

## Conceptual Differences

| Method | Core Philosophy |
|---------|----------------|
| RLHF | Learn human preferences through reward modelling + reinforcement learning |
| PPO | Stabilise policy updates during RL training |
| DPO | Directly optimise preferred outputs without RL |
| Constitutional AI | Align model using explicit ethical principles |

## When to Use What?

| Scenario | Recommended Method |
|------------|------------------|
| Large-scale production LLM | RLHF + PPO |
| Lower-cost alignment | DPO |
| Safety-first systems | Constitutional AI |
| Research on stable optimisation | PPO |
| Simplified alignment pipeline | DPO |

| Method | Strength | Weakness |
|---------|----------|-----------|
| RLHF | Strong human alignment | Expensive & complex |
| PPO | Stable RL updates | Still RL-dependent |
| DPO | Simpler & efficient | Still needs preference data |
| Constitutional AI | Safety-oriented | Limited flexibility |

### Links 
- https://www.ibm.com/docs/en/dbaoc?topic=flow-adding-generative-ai-task-service
- https://www.superannotate.com/blog/rlhf-for-llm
- https://www.turing.com/resources/rlhf-in-llms
- https://www.superannotate.com/blog/llm-fine-tuning
- https://aws.amazon.com/blogs/machine-learning/llm-continuous-self-instruct-fine-tuning-framework-powered-by-a-compound-ai-system-on-amazon-sagemaker/


---
# Useful Links
- https://paperswithcode.com/method/gpt
- https://30dayscoding.com/blog/understanding-the-architecture-of-gpt-models
- https://arxiv.org/abs/1810.04805
- https://huggingface.co/docs/transformers/en/model_doc/distilbert
- https://github.com/huggingface/transformers
- https://huggingface.co/docs/transformers/en/model_doc/bart

### Visualise Deep Learning Models
- https://projector.tensorflow.org/
- https://adamharley.com/nn_vis/cnn/3d.html
