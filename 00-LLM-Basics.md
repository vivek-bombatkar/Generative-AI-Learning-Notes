# 00 LLM Basics | Reference Prompts and Definitions

This document provides reusable prompts for each major subtopic under "LLM Basic," along with definitions only for concepts unlikely to become outdated. 


---

## 1. LLM Basic Concept

**Definition**:  
- A Large Language Model (LLM) is an artificial intelligence system trained on vast text data to understand and generate human language.
- Large language models contain many billions of features, which captures a wide range of human knowledge.
- Transformers were introduced in a 2017 paper called "Attention Is All You Need." Some LLMs, such as ChatGPT, are built on the transformer architecture.
- An innovation of transformers is this self-attention mechanism.
- Self-attention works by computing a set of query, key and value vectors for each input token.
- Then it uses the dot products between these vectors to determine the attention weights.
- The output for each token is a weighted sum of the value vectors where the weights come from the attention scores.
- All language models are trained from large text databases, such as RefinedWeb, Common Crawl, StarCoder data, BookCorpus, Wikipedia, C4, and more.    
- every language-based generative AI model has a tokenizer that converts human text into a vector that contains token IDs or input IDs. Each input ID represents a token in the model's vocabulary.
- A vector is an ordered list of numbers that represent features or attributes of some entity or concept. In the context of generative AI, vectors might represent words, phrases, sentences, or other units.
- When you write a prompt for a language model, that prompt refers to its ***Latent Space*** against its database of statistics. It returns a pile of statistics that then get assembled as words.
- Latent space refers to a high-dimensional vector space where an LLM represents the semantic meaning of language in a compressed, abstract form.
   

**Prompt**:  
- *"Explain the core concept of Large Language Models (LLMs) and their primary function in AI."*

---

## 2. Key Factors for Selecting a Language Model

**Prompt**:  
- *"List and explain the main factors to consider when choosing a language model for a business or research use case."*

---

## 3. Inference Parameters

**Prompt**:  
- *"Describe the main inference parameters (such as temperature, top-k, top-p, response length, length penalty, stop sequences) and how each affects LLM output."*

---

## 4. Popular Chat Models

**Prompt**:  
- *"Compare the most popular LLM chat models (e.g., GPT-4, Claude 2, PaLM 2, LLaMA 2, Mistral 7B) in terms of architecture, parameter size, strengths, weaknesses, and use cases."*

---

## 5. Open-Weight vs. Closed-Weight Models

**Definition**:  
- **Open-weight models**: Models whose trained parameters (weights) are publicly available for download, allowing inspection, modification, and fine-tuning.
- **Closed-weight models**: Models where only API access is provided; internal weights are not shared.

**Prompt**:  
- *"Explain the difference between open-weight and closed-weight LLMs, and discuss their respective advantages and disadvantages."*

---

## 6. Downloading Model Weights vs. the Actual Model

**Prompt**:  
- *"Clarify the distinction between downloading just the model weights and downloading the full model (architecture + weights), with examples."*

---

## 7. Ideal Business Use Cases

**Prompt**:  
- *"Summarize the ideal business use cases for prompt engineering, retrieval-augmented generation (RAG), and LLM fine-tuning, including when to use each approach."*

---

## 8. LLM Performance Metrics

**Prompt**:  
- *"List and define the main metrics used to evaluate LLM performance, such as accuracy, coherence, groundedness, fluency, relevance, and similarity."*

---

## 9. Subtopics

### a. Model Alignment and Safety

**Prompt**:  
- *"What is model alignment in LLMs, and why is safety important in real-world applications?"*

### b. LLM Limitations and Risks

**Prompt**:  
- *"Discuss common limitations and risks associated with LLMs, such as hallucinations, bias, and privacy concerns."*

### c. LLM Ecosystem and Community

**Prompt**:  
- *"Describe the importance of ecosystem support and community contributions in the development and adoption of LLMs."*

---

## 10. General Prompt for Any LLM Subtopic

- *"Explain [subtopic] in the context of Large Language Models, including its definition, importance, and practical examples."*

---


## Resources:
- [microsoft / AutoGen / development of LLM applications using multiple agents  ](https://microsoft.github.io/autogen/docs/Getting-Started)
- https://research.trychroma.com/context-rot
- 
