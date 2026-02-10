# My Learning Notes from [NUS generative-ai-fundamentals-to-advanced-techniques-programme](https://nus.comp.emeritus.org/generative-ai-fundamentals-to-advanced-techniques-programme)

---

# Reinforcement Learning vs Supervised Learning
- **Supervised Learning:** A learning paradigm where a model is trained on labelled input-output pairs and receives direct feedback on errors.
- **Reinforcement Learning (RL):** A learning paradigm where an agent learns by interacting with an environment using rewards and penalties.

### Supervised Learning
1. Collect labelled data.
2. Split into training and validation sets.
3. Train model to minimize loss.
4. Evaluate performance.

### Reinforcement Learning
1. Initialize agent and environment.
2. Perform actions based on current policy.
3. Receive reward feedback.
4. Update policy using reward signal.
5. Repeat until convergence.

## Examples
- **Supervised Learning:** Image classification, spam detection, price prediction.
- **Reinforcement Learning:** Game playing, robotics control, recommendation strategies with delayed rewards.

# Unsupervised Learning
- **Unsupervised Learning:** A branch of machine learning where models learn patterns and relationships from unlabelled data without predefined outputs.

## Examples
### Clustering Problems
- Customer segmentation
- Image segmentation
### Dimensionality Reduction Problems
- Principal Component Analysis (PCA)
- t-SNE
### Anomaly Detection
- Fraud detection
### Generative Models (partly unsupervised)
- Autoencoders
- GANs (Generative Adversarial Networks)
### Market Basket Analysis
- Identifying frequently co-occurring items
---
# 🧠 The Human Brain and Neural Complexity

## 🔍 Overview
The human brain is an incredibly complex organ and one of nature’s greatest engineering marvels.  
The **neocortex** plays a key role in higher cognitive functions such as:
- 🧠 Reasoning  
- 👁️ Perception  
- 🤔 Decision-making  
- 🗣️ Language  

Its massive connectivity gives the brain extraordinary computational power.

## 🔗 Neurons and Synapses
- The brain contains around **100 billion neurons** 🧩
- Each neuron connects to **1,000–10,000** other neurons through **synapses**
- Synapses transmit information via ⚡ electrical and 🧪 chemical signals
- The neocortex alone is estimated to have about **500 trillion synapses**, forming a massive biological network capable of learning and thought

## 🧵 Neural Wiring and Efficiency
- The neocortex contains roughly **300 million feet (≈91,440 km)** of neural wiring
- This wiring is compacted into a volume of about **1.5 quarts (≈1.4 liters)** 🤯
- Such efficiency is achieved through several biological optimizations:

### 🌀 Folding of the Cortex
- The brain surface is folded into **gyri** (ridges) and **sulci** (grooves)
- Folding increases surface area without increasing overall volume

### ⚡ Myelination
- Axons are coated with **myelin**, a fatty insulating layer
- Myelin speeds up signal transmission and reduces energy usage

### 🧩 Specialized Networks
- The brain is organized into specialized functional areas, such as:
  - 👀 Visual cortex
  - ✋ Motor cortex
  - 🧠 Prefrontal cortex
- Specialization minimizes unnecessary wiring and improves processing speed

---
# Neural Networks and Artificial Neural Networks

- Neural networks are inspired by biological brains.
- Artificial neurons approximate real neurons.
- ANNs are networks of artificial neurons.
- ANNs are simplified models of brain functionality.
- Practically, ANNs are parallel computational systems.

## Definitions
- **Neural Networks (NNs):** Networks of neurons similar to those found in biological brains.
- **Artificial Neurons:** Crude approximations of biological neurons, implemented as mathematical or software constructs.
- **Artificial Neural Networks (ANNs):** Networks of artificial neurons that approximate certain functions of real brains.
---
# 🧠 Biological vs Artificial Neurons

## 🔬 Biological Neurons
- Biological neurons have **synaptic gaps** of varying strengths 🔗
- These synapses connect to the **soma (cell body)** 🧠
- Signal strength depends on synaptic weight and connectivity
- Information flows via:
  - 🌿 Dendrites (input)
  - 🧠 Cell body (integration)
  - ⚡ Axon (signal transmission)
  - 🔚 Axon terminals (output)

## 🤖 Artificial Neurons
- Artificial neurons replace synapses with **numerical inputs**
- Inputs can come from:
  - Other neurons
  - Sensors
  - Data features
  - Variables
- Core operations include:
  - ➕ Weighted sum (Σ)
  - 📉 Activation / threshold function

## ⚙️ Computational Power
- The neocortex contains about **500 trillion synapses** operating **in parallel**
- Enables massive information processing and storage simultaneously
- The human brain operates on roughly **20 watts of power** 💡
- This is far more **energy-efficient** than modern supercomputers

## 🧩 Implications for Intelligence and Learning
- High neuron density and interconnectivity enable human intelligence
- **Plasticity** allows neural connections to reorganize with:
  - 📚 Learning
  - 🧠 Experience
  - 🩹 Recovery from injury
- This adaptability is central to skill acquisition and cognition

## 🧠 Artificial Neural Networks (ANNs)
- ANNs are inspired by biological neural systems
- They aim to approximate learning and decision-making
- While powerful, they are far less energy-efficient than the human brain
- The neocortex remains a benchmark for efficient computation and learning
---
# 🤖 Artificial Neural Networks. Why?

- 🧮 **Extremely powerful computational devices**  
  - Turing-equivalent universal computers

- ⚡ **Massive parallelism**  
  - Many simple units operate simultaneously, making computation efficient

- 📚 **Learning and generalization**  
  - Learn directly from training data  
  - No need for carefully handcrafted rules or designs

- 🛡️ **Fault-tolerant and noise-tolerant**  
  - Performance degrades gracefully even with imperfect data or failures

- 🧠 **Beyond symbolic systems**  
  - Can do everything a symbolic or logic-based system can, and more

- 📊 **Excellent with unstructured data**  
  - Particularly strong with:
    - 📝 Text  
    - 🖼️ Images  
    - 🔊 Audio  
    - Other semi-structured data
---
# 🧠 Deep Convolutional Neural Networks (CNNs)

## 📌 Definition
**Deep Convolutional Neural Networks (CNNs)** are a specialized type of neural network designed to process **structured, grid-like data**, especially **images** 🖼️.

## 🧩 Key characteristics
- Designed for **spatial data** arranged in grids
- Combine:
  - 🧠 **Convolutional layers** for feature extraction
  - 🔗 **Fully connected layers** for decision-making
- Inspired by the **human visual system**
- Highly effective when **spatial hierarchies** matter

## ⚙️ Core building blocks
- 🟨 **Input**  
- 🟩 **Convolutional Layer**. Extracts local features (edges, textures)
- 🟦 **Pooling Layer**. Reduces spatial size and computation
- 🧪 **Activation Function**. Adds non-linearity
- 🔵 **Fully Connected Layer**. Performs classification or prediction
- 🎯 **Output**

## 👁️ Why CNNs work well
- Learn **hierarchical features**. from simple edges to complex shapes
- Preserve **spatial relationships** in data
- Particularly strong for **vision-based tasks**

## 📊 Typical use cases
- Image classification
- Object detection
- Image segmentation
- Visual pattern recognition

## ✅ Advantages of CNNs
- ⚙️ **Automatic Feature Extraction**  
  - CNNs learn features directly from raw data  
  - No manual feature engineering required
- 🔁 **Parameter Sharing**  
  - Same filters are reused across the image  
  - Fewer parameters than fully connected networks
- 📍 **Translation Invariance**  
  - Recognize patterns (edges, shapes, objects) regardless of position in the image
- 📐 **Efficient for High-Dimensional Data**  
  - Scales well to large images and datasets
- 🏆 **State-of-the-Art Performance**  
  - Top results in:
    - Image classification
    - Object detection
    - Image segmentation
- 🌍 **Adaptability to Diverse Domains**  
  - Can be applied to:
    - 🖼️ Images
    - 🔊 Audio spectrograms
    - ⏱️ Time-series data  
  - Requires minimal architectural changes

## ⚠️ Limitations of CNNs
- 💻 **Computationally Intensive**  
  - Training requires powerful hardware (GPUs, TPUs)
- 📊 **Data Hungry**  
  - Needs large labelled datasets for good performance  
  - Data collection and annotation can be expensive
- 🔍 **Lack of Interpretability**  
  - Acts as a black-box model  
  - Difficult to understand or debug decisions
- 📉 **Overfitting Risk**  
  - Without proper regularisation, models may memorise training data
- 🎛️ **Sensitivity to Hyperparameters**  
  - Performance depends heavily on:
    - Architecture
    - Learning rate
    - Other tuning choices

---
# 🤖 GPT Models and How ChatGPT Works

## 🧮 Model Scale Comparison

| Model   | Number of Parameters |
|--------|----------------------|
| GPT-1  | 117M |
| GPT-2  | 1.5B |
| GPT-3  | 175B |
| GPT-3.5 | GPT-3 + ~6B |
| GPT-4  | ~1.7T |

## 🔁 How ChatGPT Works (End-to-End Flow)

➡️ **Pre-training**  
- Trained on massive internet text
- Learns to predict the **next token**
- Captures grammar, facts, and language patterns

➡️ **Fine-tuning**  
- Uses datasets reviewed by **human trainers**
- Learns to generate safer and more helpful responses
- Generalises from human feedback

➡️ **Input Processing**  
- User input is **tokenised** into words or subwords
- Tokens are fed into the transformer model

➡️ **Contextual Understanding**  
- Maintains conversation history 🧠
- Transformer architecture models long-range dependencies
- Enables context-aware responses

➡️ **Response Generation**  
- Predicts next tokens based on learned patterns
- Produces coherent, human-like text

➡️ **Sampling and Optimisation**  
- Uses probabilistic sampling 🎲
- Adds controlled randomness for natural responses
- Safety techniques reduce harmful outputs

➡️ **Post-processing**  
- Removes special tokens and formatting
- Final response is shown to the user 💬


# 🧠 Attention Mechanism in Transformers

## Attention
At the core of modern NLP lies **attention**.  
It allows models to **focus**, not memorise.  
Instead of treating all words equally, attention helps the model decide **what matters most** in context.


## 🔑 Query, Key, and Value (QKV)

To understand attention, everything revolves around three components:

### 🔍 Query (Q)
- Represents the **current word or token**
- Think of it as a word *asking* for relevant context

### 🗝️ Key (K)
- Represents **all words in the input sequence**
- Each word has a key used to measure relevance to the query

### 📦 Value (V)
- Contains the **actual information** (embeddings)
- This is the content passed forward once relevance is determined


## ⚙️ How Attention Works (Step-by-Step)

### 1️⃣ Compute Attention Scores
- Compute the **dot product** between:
  - Query (Q)
  - Each Key (K)
- Result. relevance scores for each word

### 2️⃣ Scale the Scores
- Scores are scaled by **√dk**
- Prevents large values from dominating
- Ensures numerical stability

### 3️⃣ Apply Softmax
- Converts scores into a **probability distribution**
- Determines how much attention each word receives

### 4️⃣ Weighted Sum of Values
- Probabilities are applied to Values (V)
- Produces a **context-aware representation**

## 🎯 Why Attention Matters
- Focuses on **relevant words**, regardless of position
- Handles **long-range dependencies**
- Essential for understanding meaning in complex sentences

# 🧩 Multi-Head Attention

## 🔀 What is Multi-Head Attention?
- Instead of one attention mechanism, the model uses **multiple heads**
- Each head attends to **different aspects** of the sequence

## ⚙️ How It Works
- Input is split into multiple Q, K, V sets
- Each head computes attention **independently**
- Outputs are:
  - Concatenated
  - Passed through a linear layer

➡️ Result. richer and more expressive representations


# 🧪 Feedforward Networks (FFN)

## 🔧 Role of Feedforward Networks
- Applied **after multi-head attention**
- Operates **independently on each position**

## 🏗️ Structure
- Linear layer
- ReLU activation
- Linear layer

## 🎯 Purpose
- Captures **more abstract patterns**
- Refines representations beyond word-to-word relationships
- Output is passed to the next transformer layer


# 🏗️ Transformer Architecture. Big Picture

- 🔡 Embeddings
- 🔁 Self-attention (QKV)
- 🔀 Multi-head attention
- 🧪 Feedforward networks
- 🔄 Stacked layers for depth

Together, these components allow transformers to:
- Understand context deeply
- Scale efficiently
- Generate high-quality, human-like text
---

# 🤖 Transformer Models, Generative AI, and Modern Architectures

## 🧠 BERT vs GPT vs T5

### 🟧 BERT
- **Bidirectional Encoder Representations from Transformers**
- Understands text in **both directions**
- Best suited for:
  - 😊 Sentiment analysis
  - 🏷️ Text classification
  - ❓ Question answering

### 🟩 GPT
- **Generative Pre-trained Transformer**
- **Autoregressive**. predicts the next word in a sequence
- Ideal for:
  - 💬 Conversational AI
  - ✍️ Content generation
  - 💻 Code generation

### 🟦 T5
- **Text-to-Text Transfer Transformer**
- Treats **all NLP tasks as text-to-text**
- Highly versatile for:
  - 🌍 Translation
  - 🧾 Summarisation
  - ❓ Question answering


# ✨ Generative AI and Transformers

- Generative AI creates **new, original content**
- Learns patterns from **large datasets**
- Transformers are the backbone of modern generative AI
- Powered by:
  - 🔁 Self-attention
  - 🧠 Deep learning
- Applications extend beyond text:
  - 🖼️ Image generation
  - 🎵 Music composition
  - 💻 Code generation


# 📚 Large Language Models (LLMs)

- Designed to **understand and generate human language**
- Built on **transformer architectures**
- Use self-attention for deep contextual understanding
- Trained on massive datasets with **billions of parameters**
- Capabilities include:
  - ❓ Question answering
  - ✍️ Text generation
  - 🌍 Translation
  - 💻 Code writing
- Used in:
  - 💬 Chatbots
  - 🗣️ Language systems
  - 📝 Content tools


# 👁️ Large Vision Models (LVMs)

- Designed for **visual understanding and generation**
- Process:
  - 🖼️ Images
  - 🎥 Videos
- Built using:
  - CNNs
  - Transformers
- Applications include:
  - 🎨 Image synthesis
  - 🔍 Object detection
  - 🚗 Autonomous systems
  - 📹 Video analysis


# ⚙️ Training a Transformer

- 🔡 Data preprocessing. tokenisation and embeddings
- 📍 Positional encodings added for sequence order
- 🎲 Random weight initialisation
- 🔁 Training via **backpropagation**
- 🎛️ Hyperparameters tuned:
  - Learning rate
  - Batch size
- Goal. minimise loss and optimise performance


# 💻 Computational Challenges

- Large models are **computationally intensive**
- Example:
  - GPT-3 has **175 billion parameters**
- CPUs are insufficient for training
- Requires:
  - ⚡ GPUs
  - 🚀 TPUs
- Computational needs have grown **exponentially**
- Hardware advances are critical for progress



# 🔄 Transformer Variants
- Transformers excel at sequential data
- Variants optimise performance for specific tasks


# 🧩 Hybrid Architectures

### CNN + Transformer
- Common in computer vision
- CNN captures **local spatial features**
- Transformer captures **long-range dependencies**
- Used in:
  - Vision Transformers (ViTs)
  - ResNet hybrids

### RNN + Transformer
- Useful for:
  - 🗣️ Speech recognition
  - ⏱️ Time-series forecasting
- RNN handles local sequences
- Transformer handles global context

### ✅ Benefits
- Reduced computational cost
- Better efficiency and scalability
- Strong performance on complex tasks


# ⚡ Efficient Transformers (Long Sequences)

- Standard transformers scale **quadratically**
- Problematic for long documents
- Sparse and efficient variants address this

### 📄 Longformer
- Uses **dilated attention**
- Focuses on selected tokens
- Ideal for:
  - Document-level QA
  - Summarisation

### 📉 Linformer
- Projects attention to lower dimensions
- Reduces memory and inference cost
- Scales well to long sequences

### 🔁 Reformer
- Optimised for memory efficiency
- Enables longer context handling



# ⚠️ Challenges in Text Generation

### 🤯 Hallucinations
- Generates plausible but incorrect information
- Caused by missing or weak training signals

### ⚖️ Bias
- Models inherit biases from training data
- Can reflect:
  - Gender bias
  - Cultural bias
  - Racial bias

### 🧭 Ethics
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

---






---
## Visualise Deep Learning Models:
  - https://projector.tensorflow.org/
  - https://adamharley.com/nn_vis/cnn/3d.html
