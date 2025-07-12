
# 20 GenAI Based Applications

##  Typical components of a generative AI application

![Foundation Model Architecture](https://render.skillbuilder.aws/cds/4fdd8f7e-99d9-4ac7-b5e4-f3967537eb64/assets/FoundationModelCircle_NOPROCESS_.png)

## Prompt history store
- Many FMs have a limited context window, which means you can only pass a fixed length of data as input.
- Storing state information in a multiple-turn conversation becomes a problem when the conversation exceeds the context window.
- To solve this problem, you can implement a prompt history store.
- It can persist the conversation state, making it possible to have a long-term history of the conversation.

## 2. Generative AI Application Architecture Patterns - 3 
### 1. Text summarization pattern For large documents or text
- map-reduce architecture and apply the concepts of chunking and chaining prompts.  The test summarization architecture for large documents includes the following steps:  
  1. Split a large document into multiple small number (n) chunks using tools such as LangChain.  
  2. Send each chunk to the FM to generate a corresponding summary.  
  3. Append the next chunk to the first summary generated and summarize again.  
  4. Iterate on each chunk to create a final summarized output.  
### 2. AI assistant pattern
- This architecture includes the following steps:
  1. The user queries the AI assistant.  
  2. The chat history (if there is any) is passed on to the Amazon Bedrock model along with the user’s current query.    
  3. The model then generates a response.     
  4. The model passes the response back to the user.  
### 3. AI assistant use cases
  1 Basic AI assistant: This is a zero-shot AI assistant with an FM model.  
  2 AI assistant using a prompt template: This is an AI assistant with some context provided in the prompt template.  
  3 AI assistant with a persona: This is an AI assistant with defined roles, such as a career coach with human interactions.  
  4 Contextual-aware AI assistant: This is an AI assistant that passes context through an external file by generating embeddings.  



## comparison of **LangSmith**, **LangGraph**, and **LangChain** in tabular format:

| **Feature/Aspect**         | **LangSmith**                                    | **LangGraph**                                 | **LangChain**                                 |
|----------------------------|--------------------------------------------------|-----------------------------------------------|-----------------------------------------------|
| **Primary Purpose**         | Model evaluation, testing, and monitoring for LLM workflows. | A visual interface for building and debugging LLM pipelines. | A framework for building, managing, and deploying LLM applications with advanced chains and integrations. |
| **Use Case**                | Evaluate and monitor the performance and efficiency of LLM-based workflows. | Visualize and debug the flow of data between various components in LLM systems. | Develop LLM-powered applications, such as chatbots, RAG systems, and multi-agent pipelines. |
| **Core Focus**              | Debugging, testing, and optimizing agent behaviors and prompt workflows. | Visual design of pipelines with a focus on data flow and interaction between steps. | Flexibility in building advanced chains for LLM-based apps with extensive integrations. |
| **Strengths**               | - Testing and monitoring LLM applications. <br> - Fine-tuning and performance analysis.<br> - Great for iterating and optimizing prompts. | - Intuitive visual design interface.<br> - Real-time debugging.<br> - Best for prototyping complex workflows visually. | - Extensive integration with external tools.<br> - Pre-built components for rapid development.<br> - Customizability for chaining LLM models with additional tools (like databases). |
| **Visual Interface**        | No dedicated visual interface; mainly focused on backend testing and debugging tools. | Yes, provides a graphical interface for building and visualizing workflows. | No native visual design interface but highly modular with code-based chain building. |
| **Chain Composition**       | Supports testing of chain components but not designed to build chains directly. | Focuses on visualizing how components interact in the chain. | Build and manage complex chains, integrating LLMs with APIs, databases, and more. |
