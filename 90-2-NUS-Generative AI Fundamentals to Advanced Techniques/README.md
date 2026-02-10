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

---
## Visualise Deep Learning Models:
  - https://projector.tensorflow.org/
  - https://adamharley.com/nn_vis/cnn/3d.html
