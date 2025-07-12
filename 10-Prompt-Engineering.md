# 10 Prompt Engineering | Reference Prompts and Stable Definitions

This document provides **reusable prompts** for each major subtopic in Prompt Engineering, with definitions only for concepts unlikely to become outdated. New subtopics are suggested where relevant.

---

## 1. Prompt Engineering (Core Concept)

**Definition**:  
Prompt engineering is the practice of designing and refining inputs (prompts) to maximize the effectiveness and accuracy of large language model outputs.

**Prompt**:  
- "Explain what prompt engineering is and why it is important for working with large language models."

---

## 2. Common Prompt Engineering Techniques

**Prompt**:  
- "List and describe the main prompt engineering techniques used with large language models, such as zero-shot, few-shot, chain-of-thought, instruction-based, reframing, and role-playing."

---

## 3. Prompt Prefixes

**Prompt**:  
- "Show examples of how prompt prefixes (e.g., 'As a knowledgeable advisor:') can influence the style and quality of LLM responses."

---

## 4. Prompt Decomposition

**Prompt**:  
- "Explain how breaking down a complex prompt into sub-prompts (prompt decomposition) can improve the clarity and relevance of LLM outputs."

---

## 5. Prompt Reframing

**Prompt**:  
- "Demonstrate how to reframe a prompt to make it more specific or targeted for a desired outcome."

---

## 6. Prompt Constraints

**Prompt**:  
- "Provide examples of adding explicit constraints (e.g., word count, formality, cost) to prompts to guide LLM responses."

---

## 7. Prompt Mirroring

**Prompt**:  
- "Describe prompt mirroring and how asking the model to summarize or rephrase the prompt can help verify its understanding."

---

## 8. ReAct Prompting

**Prompt**:  
- "Explain the ReAct prompting framework and how it enables LLMs to interleave reasoning traces with actions for more reliable and interactive outputs."

---

## 9. Output Priming

**Prompt**:  
- "How does output priming, such as providing a partial answer or formatting hint, affect the LLM's output?"

---

## 10. Multi-Turn and Dynamic Prompting

**Prompt**:  
- "Describe multi-turn and dynamic prompting, and give examples of how prompts can adapt based on previous model responses."

---

## 11. Prompt Chaining

**Prompt**:  
- "Explain prompt chaining and how breaking a complex task into a sequence of prompts can improve LLM performance."

---

## 12. Temperature Control in Prompting

**Prompt**:  
- "How does adjusting the temperature parameter in prompting influence the creativity and determinism of LLM responses?"

---

## 13. Suggested New Subtopics

### a. Evaluation of Prompt Effectiveness

**Prompt**:  
- "What methods can be used to evaluate the effectiveness of different prompts with LLMs?"

### b. Prompt Engineering for Safety and Bias Mitigation

**Prompt**:  
- "How can prompt engineering be used to minimize bias and improve the safety of LLM outputs?"

---

## 14. General Prompt for Any Prompt Engineering Subtopic

- "Explain [subtopic] in the context of prompt engineering for large language models, including its definition, purpose, and practical examples."

---


## Prompt Engineering techniques commonly used with large language models like GPT-3, GPT-4, and other similar models:

| **Technique** | **Description** |
|---------------|-----------------|
| **Zero-Shot Prompting** | Asking the model to perform a task without providing any examples. The model relies on its pre-trained knowledge. |
| **Few-Shot Prompting** | Providing a few examples of the task (input-output pairs) in the prompt to guide the model’s output. This helps the model understand the task with minimal examples. |
| **One-Shot Prompting** | A special case of few-shot prompting where only one example is provided to guide the model's response. |
| **Chain-of-Thought Prompting** | A technique where the model is prompted to generate intermediate reasoning steps before reaching a final answer. This improves the model’s performance in tasks that require logical reasoning or problem-solving. |
| **Instruction-Based Prompting** | Giving explicit instructions to the model on how to respond or behave for a particular task. For example, “Explain this concept in simple terms” or “Generate a summary of this text.” |
| **CoT + Self-Consistency** | Similar to chain-of-thought prompting, but here multiple reasoning paths are generated. The final answer is chosen based on the most consistent reasoning path. |
| **Reframing** | Changing the prompt's wording or structure without changing its meaning to influence how the model responds. This can help steer the model toward more accurate or creative outputs. |
| **Multi-Turn Prompting** | Asking the model to respond to multiple prompts in sequence, which simulates a conversational or interactive problem-solving scenario. |
| **Output Priming** | Providing part of the expected answer or formatting the output in a specific way to guide the model toward a specific type of response (e.g., bulleted lists, steps, or tables). |
| **Role-Playing** | Assigning the model a specific role or persona to guide its responses, such as “You are a teacher,” “You are a data scientist,” or “You are a customer service agent.” |
| **Dynamic Prompting** | Dynamically changing the prompt based on the model's previous responses to make the interaction more adaptive and context-aware. |
| **Question-Answer Prompting** | Asking the model questions that lead to a final conclusion. Each question guides the model closer to the desired answer through a step-by-step process. |
| **Contextual Prompting** | Including relevant context (e.g., background information or constraints) in the prompt to ensure the model responds in a manner that is aligned with that context. |
| **Task Specification** | Clearly specifying the task or goal in the prompt. For instance, "Translate this text into French" or "Summarize this article in three bullet points." |
| **Iterative Prompting** | Iteratively refining the prompt based on the model's responses, often involving multiple interactions to get closer to the desired output. |
| **Prefix Prompting** | Adding a prefix or hint to the input to influence the model’s behavior. For example, adding “Write a formal letter” before a text to generate a more formal response. |
| **Temperature Control in Prompting** | Adjusting the **temperature** parameter to control randomness or creativity in responses (higher values for more creative, lower values for deterministic responses). |
| **Reverse Prompting** | Presenting the expected output first, and asking the model to reverse-engineer or explain how it arrived at that conclusion. This helps with complex problem-solving tasks. |
| **Prompt Chaining** | Breaking down a complex task into a series of smaller prompts, with each prompt building on the output of the previous one. This is useful for multi-step tasks. |
| **Demonstration-Based Prompting** | Providing detailed, step-by-step demonstrations of the task, sometimes including explanations, before asking the model to perform the task. |
| **Meta-Prompting** | Instructing the model on how to generate better prompts or adjust its output (i.e., giving instructions about how to give instructions). |
| **Paraphrasing Prompts** | Rewriting the same prompt in different ways to elicit a variety of responses, particularly useful for ambiguous tasks or creative outputs. |
| **Prompt Ensembling** | Using multiple prompts with slight variations, aggregating the results, or choosing the best response among them to improve reliability. |
| **Active Prompting** | Asking the model questions in the prompt itself that it needs to answer before generating the final response. This makes the model think actively as it generates its final output. |
| **Prompt Constraints** | Including explicit constraints in the prompt, such as character limits, formality level, or specific response formatting (e.g., “Answer in 100 words or less”). |
| **Few-Shot In-Context Learning** | Providing task-specific data within the prompt itself and leveraging it for in-context learning to adapt to a particular use case without fine-tuning. |
| **Template-Based Prompting** | Using predefined templates to structure the input in a way that aligns with the model’s strengths, ensuring more accurate outputs (e.g., pre-defined questions or sentence structures). |



## Reference
- https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/52338430/7aa0cda1-b1c4-4df8-825c-5c758b05779f/10-Prompt-Engineering.md
- https://www.promptingguide.ai/
- https://learn.microsoft.com/en-us/azure/ai-services/openai/concepts/advanced-prompt-engineering?pivots=programming-language-chat-completions
- https://github.com/microsoft/generative-ai-for-beginners/tree/main/04-prompt-engineering-fundamentals


