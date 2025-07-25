

# 30 RAG | Retrieval Augmented Generation


---

### RAG (Retrieval-Augmented Generation)
An architecture that combines information retrieval (e.g., vector search) with generative models to produce contextually accurate responses by injecting external documents into prompts.

### Embeddings
Vector representations of text that capture semantic meaning, enabling similarity search in vector databases.

### Vector Database
A specialized database for storing and querying embeddings. Commonly used in RAG to retrieve semantically similar chunks.

### Chunking
The process of splitting large documents into smaller, meaningful segments for embedding and retrieval.

### Re-ranking
A post-processing step that refines the list of retrieved documents using ML or heuristics to improve relevance.

### Hybrid Search
Combines traditional keyword-based search with vector search to improve recall and precision.

### Retrieval Pipeline
A sequence of operations: user input → embed → search → rerank → augment → generate.

### Context Window
The maximum number of tokens an LLM can consider in a single prompt. Retrieved content must fit this limit.

---

## 🧠 RAG Workflow Understanding
**Prompt:**  
> "Explain step-by-step how Retrieval-Augmented Generation works using embeddings and a vector store."

---

## 🔗 Chunking Strategy
**Prompt:**  
> "Compare fixed, token-based, semantic, and hierarchical chunking strategies. When should each be used in a RAG pipeline?"

---

## 🧬 Embeddings
**Prompt:**  
> "What are text embeddings and how are they used in LLM-based retrieval systems like RAG? Include example Python code using HuggingFace."

---

## 📦 Vector Database Selection
**Prompt:**  
> "Compare Pinecone, FAISS, Weaviate, and Milvus for use in LLM RAG applications. Evaluate based on speed, scalability, hybrid search, and integration."

---

## 🔁 RAG Retrieval + Generation Chain
**Prompt:**  
> "Design a LangChain pipeline that retrieves context from a Qdrant DB and uses Claude to answer the query."

---

## 🔍 OpenSearch Integration
**Prompt:**  
> "Explain how OpenSearch can be used as a vector store in a RAG system. Include the steps for uploading documents, indexing, and querying with DSL."

---

## 📊 RAG Evaluation
**Prompt:**  
> "List and describe the key metrics used to evaluate RAG systems, such as context precision, context recall, and faithfulness."

---

## 🧪 Prompt for RAG Debugging
**Prompt:**  
> "How can I validate if retrieved documents are contributing to the model's output? What debug techniques are used in LangSmith?"

---

## 🎛️ Hybrid Search
**Prompt:**  
> "Create a hybrid search query that combines keyword match and vector similarity in OpenSearch. Describe why it improves performance."

---

## 🧠 Chain of Thought in RAG
**Prompt:**  
> "How can chain-of-thought prompting enhance multi-step reasoning in RAG applications? Provide an example prompt."

---

## 🔐 Fine-Grained Access Control
**Prompt:**  
> "Explain how OpenSearch uses FGAC to secure access to vector search queries in a RAG application."

---

## 🛡️ Security Prompt
**Prompt:**  
> "What security layers should be added when deploying a production-grade RAG system? Discuss VPC, IAM, FGAC, and Cognito."



---

## AWS Workshop Notes

Before a text generation model is able to answer questions and we can take advantage of a RAG architecture, the documents must be processed and stored in a document store index, following these steps:

1. Load the documents  
2. Process and split them into smaller chunks  
3. Create a numerical vector representation of each chunk using the Amazon Bedrock Titan Embeddings model  
4. Create an index using the chunks and the corresponding embeddings  

When the documents index is prepared, you are ready to ask the questions. The following steps will be executed:

1. Create an embedding of the input question  
2. Compare the question embedding with the embeddings in the index  
3. Fetch the (top N) relevant document chunks  
4. Add those chunks as part of the context in the prompt  
5. Send the prompt to the model under Amazon Bedrock  
6. Get the contextual answer based on the documents retrieved  

## Indexing Strategies: Optimizing How Documents Are Stored and Retrieved

1. **Indexing Strategies:** Optimizing how documents are stored and retrieved  
   - Select the right index type (e.g., dense vector index, hybrid index)  
   - Use appropriate sharding and replication strategies  
   - Implement versioning and re-indexing pipelines for updates  

2. **Data Preparation:**  
   - **Effective Chunking Strategies**  
     - Break documents into meaningful, context-preserving chunks  
     - Use overlap to maintain context between chunks  
     - Optimize chunk size for your embedding model’s token limit  
   
   - **Creating Embedded Vectors for Enhanced Search Capabilities**  
     - Generate vector embeddings using Amazon Bedrock Titan or other LLM models  
     - Store embeddings alongside metadata for hybrid search  
     - Normalize and deduplicate content to reduce noise  

3. **Document Schema Optimization:** Structuring your data to fully leverage OpenSearch's powerful search and analytics capabilities  
   - Define a clear schema with well-typed fields (text, keyword, date, vector)  
   - Use analyzers, filters, and custom mappings to improve text search  
   - Include metadata fields (e.g., source, timestamp, document type) for filtering and scoring  
   - Plan for field-level access control and fine-grained relevance tuning  


## OpenSearch through queries:


| **Code Section**              | **Description** |
|------------------------------|------------------|
| **Index Settings**           | - Enables **KNN search** (`index.knn: true`) using the HNSW algorithm, allowing approximate nearest neighbor search for vector embeddings. <br> - Sets `ef_search: 512` to balance search accuracy and performance. <br> - Uses **5 shards** for distributed storage and scalability. <br> - Defines two analyzers: the default uses standard English tokenization with stopword removal, and a custom `content_analyzer` adds lowercase, stopword, and snowball filters. |
| **Mappings**                 | - `file_name`: A text field with a subfield of keyword type for sorting/filtering. <br> - `chunk_id`: A numeric field identifying document segments. <br> - `content_text`: Uses the `content_analyzer` for advanced text processing; supports position indexing and scoring. <br> - `content_vector_titan_v1`: A 1536-dimensional vector for similarity search using Titan v1 embeddings. <br> - `content_vector_titan_v2:0`: A 1024-dimensional vector for Titan v2. Both use **HNSW** (Hierarchical Navigable Small World) algorithm with `l2` (Euclidean) distance and are optimized using `nmslib`. |
| **Dynamic Templates**        | - Automatically maps fields in `metadata_fields` based on their suffix: <br> - `*_id`: Mapped as keyword (for exact search). <br> - `*_s`: Mapped as text (for full-text search). <br> - `*_t`: Mapped as date with timestamp format. <br> - `*_d`: Mapped as date with simpler date format. This allows flexibility in storing and querying structured metadata. |


# Chunking

Chunking strategies are important techniques for breaking down large amounts of text or data into smaller, more manageable pieces. The choice of strategy often depends on the specific requirements of the application, the nature of the text data, and the downstream tasks to be performed. Here are some examples of chunking strategies:

| **Attribute**              | **Fixed-Size Chunking**                                      | **Sentence-Based Chunking**                                  | **Token-Based Chunking**                                      |
|---------------------------|---------------------------------------------------------------|---------------------------------------------------------------|----------------------------------------------------------------|
| **Splitting Method**      | By character count (e.g., 1000 chars)                         | At sentence boundaries                                        | After fixed number of tokens (words or subwords)              |
| **Ease of Implementation**| Very simple                                                   | Requires sentence segmentation logic                          | Requires tokenization using NLP libraries                     |
| **Chunk Size Consistency**| High (uniform chunk sizes)                                    | Variable (depends on sentence length)                         | Moderate to high (more semantically stable than char-based)   |
| **Semantic Preservation** | Low to Moderate                                               | High                                                          | High                                                           |
| **Best For**              | Quick prototyping, fixed-size models                         | Language tasks, summarization, QA                             | Embedding generation, semantic search                         |

# Embedding Vectors
For the embedding process we decided to use third-party models from HuggingFace, to demonstrate how to connect external endpoints into OpenSearch, and we will dive deep into that in the next sections.



# Upload documents into OpenSearch
After chunking and embedding the documents, the next step is to upload them to OpenSearch. This process requires formatting the data to match the structure of the OpenSearch index.

By carefully aligning the uploaded data with the OpenSearch index structure, we ensure optimal performance in data retrieval and querying operations. This structured approach forms the foundation for building powerful search and analytics capabilities on top of the uploaded data.



# Performing Basic Search Queries in Open Search


| **Query Type**             | **Objective**                                                                                           | **Query Code (Elasticsearch DSL)**                                                                                                                                                                                                                                                                                                                                                                                                                         | **Guidance**                                                                                                                                                             |
|----------------------------|---------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **1. Simple Text Search**  | Retrieve documents containing a specific phrase.                                                        | ```json<br>GET /10-k/_search<br>{<br>  "_source": {<br>    "excludes": ["content_vector_titan_v1", "content_vector_titan_v2:0", "metadata_fields"]<br>  },<br>  "query": {<br>    "match": {<br>      "content_text": {<br>        "query": "3M revenue contribution from Asia-Pacific 2023"<br>      }<br>    }<br>  }<br>}```                                                                                                     | Searches for a specific phrase in `content_text`. Vector and metadata fields are excluded.                                         |
| **2. Fuzzy Search**        | Search for approximate matches, useful for handling typos.                                              | ```json<br>GET /10-k/_search<br>{<br>  "_source": {<br>    "excludes": ["content_vector_titan_v1", "content_vector_titan_v2:0", "metadata_fields"]<br>  },<br>  "query": {<br>    "fuzzy": {<br>      "content_text": {<br>        "value": "aparel",<br>        "fuzziness": "AUTO"<br>      }<br>    }<br>  }<br>}```                                                                                                  | Finds near-matches for `"aparel"` using fuzzy logic. Useful for typo handling.                                                    |
| **3. Match Phrase Query**  | Find documents with an exact phrase for high relevance.                                                 | ```json<br>GET /10-k/_search<br>{<br>  "_source": {<br>    "excludes": ["content_vector_titan_v1", "content_vector_titan_v2:0", "metadata_fields"]<br>  },<br>  "query": {<br>    "bool": {<br>      "must": [<br>        {<br>          "match": {<br>            "content_text": "This brand includes a wide assortment of baby and toddler apparel"<br>          }<br>        }<br>      ]<br>    }<br>  }<br>}``` | Matches exact long-form phrases in the content field. Uses `bool` with `must` clause.                                              |
| **4. Boost Query Search**  | Prioritize certain fields over others for fine-tuned relevance.                                         | ```json<br>GET /10-k/_search<br>{<br>  "_source": {<br>    "excludes": ["content_vector_titan_v1", "content_vector_titan_v2:0", "metadata_fields"]<br>  },<br>  "query": {<br>    "bool": {<br>      "must": [<br>        {<br>          "match": {<br>            "metadata_fields.exchange_id": {<br>              "query": "NYSE",<br>              "boost": 3<br>            }<br>          }<br>        },<br>        {<br>          "range": {<br>            "metadata_fields.document_period_end_date_d": {<br>              "gte": "Dec 01 2021",<br>              "lte": "Dec 31 2023"<br>            }<br>          }<br>        },<br>        {<br>          "match": {<br>            "content_text": {<br>              "query": "financial statements",<br>              "boost": 1<br>            }<br>          }<br>        }<br>      ]<br>    }<br>  }<br>}``` | Boosts `"exchange_id"` match over others. Filters by date range and text relevance.                                                |
| **5. Metadata Search**     | Target specific metadata fields, essential for filtering by document type and date range.              | ```json<br>GET /10-k/_search<br>{<br>  "_source": {<br>    "excludes": ["content_vector_titan_v1", "content_vector_titan_v2:0", "metadata_fields"]<br>  },<br>  "query": {<br>    "bool": {<br>      "must": [<br>        {<br>          "match": {<br>            "metadata_fields.form_type_s": "10-K"<br>          }<br>        },<br>        {<br>          "range": {<br>            "metadata_fields.filed_as_of_date_d": {<br>              "gte": "Dec 01 2021",<br>              "lte": "Aug 31 2024"<br>            }<br>          }<br>        }<br>      ]<br>    }<br>  }<br>}``` | Filters by `"form_type_s": "10-K"` and date between Dec 2021 and Aug 2024. Enables precise metadata-level filtering.                |


# Text to Vector Conversion
In vector search, text data needs to be transformed into a format that machines can understand and compare—this is where embeddings come in. Embeddings are dense vector representations of text, where similar pieces of text are positioned closer together in vector space. This transformation allows for semantic searching, where the system can retrieve documents not just based on keyword matching, but on the meaning of the text itself. By generating embeddings for text, we create the foundation for advanced search capabilities that can handle nuanced queries, improving the accuracy and relevance of search results.

- This query sends the sentence "today is sunny" to a model to generate its vector embedding.
- The return_number option indicates that numerical values of the embeddings should be returned.
- The target_response specifies the embedding type to return (in this case, sentence_embedding).

# enrich queries with neural models, enhancing search accuracy.
In vector search, consistency and efficiency are key. A Neural Query Enrichment Pipeline automates the process of enriching search queries with pre-configured neural models, ensuring that every search benefits from sophisticated machine learning enhancements without needing manual configuration each time. This setup allows for faster, more reliable, and more accurate searches by automatically applying the best-suited models to different content fields. It also simplifies the process for developers and users by standardizing the search experience across different datasets or indices.

| **Query Type**                               | **Objective**                                                                                      | **Guidance**                                                                                                                                                                                |
|----------------------------------------------|----------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **1. Vector Search (Default Pipeline)**       | Execute a vector search to find semantically similar documents using a single embedding model.     | Searches the `content_vector_titan_v1` field with a query text. Returns top 5 semantically relevant documents. Excludes unnecessary fields for clarity.                                   |
| **2. Multi-Embeddings Vector Search**         | Use multiple vector fields to conduct a more comprehensive and robust semantic search.             | Combines `content_vector_titan_v1` and `content_vector_titan_v2:0` to ensure the returned results are relevant across different embedding representations.                                |
| **3. Hybrid Search with Normalization**       | Combine traditional text search with vector search, using normalization techniques to refine output. | Merges a `match` query and a `neural` vector query. Uses L2 normalization and weighted arithmetic mean to balance relevance. Ideal for combining lexical and semantic relevance signals. |


# Reranking with Neural Plugin


| **Query Type**                       | **Objective**                                                                                              | **Description**                                                                                                                                                                                                                                   |
|-------------------------------------|------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **1. Lexical Search with Reranking** | Retrieve documents using keyword match, then refine results using a neural reranking model.               | Searches for documents containing the phrase "3M Food Safety Division divestiture impact 2023" using `match`. Results are reranked using an ML model, focusing on `content_text` while excluding vector and metadata fields.                      |
| **2. Single Vector Search with Reranking** | Use a neural vector search (Titan V1) followed by reranking for more relevant results.                     | Executes a semantic search for a growth-related query using Titan V1 vectors. The top 5 results are reranked by a reranking model to improve the ordering of the most relevant documents from financial reports.                                 |
| **3. Multi-Vector Search with Reranking** | Combine two neural vector fields (Titan V1 and Titan V2) for a robust search, then rerank results.         | Searches using both vector fields for the query "Carter's U.S. Retail vs Wholesale revenue 2023". Applies reranking on the combined results using an ML model, targeting the most relevant passages from multiple embeddings.                     |
| **4. Hybrid Search with Reranking** | Mix traditional lexical (`match`) and neural vector (`neural`) queries, then normalize and rerank.         | Combines exact text match with semantic vector search (Titan V1 and V2). Normalizes using L2 and reranks results using weighted arithmetic mean. Ideal for balanced relevance from both lexical and semantic perspectives.                         |


| **Feature**                  | **Objective**                                                                                       | **Why It's Important**                                                                                                              | **Description**                                                                                                                                                 |
|-----------------------------|-----------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **1. Highlight Query Matching** | Emphasize the search term(s) within the content of matched documents.                              | Allows users to quickly see why a document was returned by highlighting the matching text, improving user trust and usability.     | Adds a `"highlight"` section in the search query to wrap matching content fields (e.g., `content_text`) in HTML tags, making matches easy to spot visually.    |
| **2. OpenSearch Explain**       | Understand how and why a document received its relevance score in a search query.                 | Helps developers, analysts, and users understand ranking logic, enabling better debugging and refinement of queries.                | Adds `?explain=true` to the search request. The response includes detailed scoring breakdowns for each document, including term frequency, field boosts, etc.    |



| **Security Layer / Component**           | **Purpose**                                                                                             | **Key Details**                                                                                                                                                                                                 |
|------------------------------------------|---------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **1. Networking**                        | Restrict access at the network level                                                                    | Choose between **VPC access** (more secure) and **Public access**. VPC setup can use NGINX Proxy and Security Groups to control inbound access to OpenSearch Dashboards.                                         |
| **2. Domain Access Policies**            | Allow or deny access to OpenSearch endpoints before reaching the service                                 | Resource-based access policies are evaluated at the edge of the domain (URI-level), controlling whether requests are forwarded to OpenSearch.                                                                 |
| **3. Fine-Grained Access Control (FGAC)**| Authenticate and authorize users after domain-level access                                               | Includes user authentication, role mapping, and permission evaluation (index, document, and field level). Allows Open Domain Access Policies when FGAC is enabled.                                              |
| **FGAC: Roles**                          | Assign granular permissions (cluster/index/document/field level)                                         | Roles are different from IAM roles. One user can have multiple roles (e.g., dashboard access, read-only on index1, write on index2).                                                                            |
| **FGAC: Mapping**                        | Map users to one or more roles                                                                           | Roles are assigned to users to define what parts of the cluster they can access and with what permissions.                                                                                                      |
| **FGAC: Users**                          | People or apps making requests with credentials                                                          | Users can be IAM-authenticated or use username/password. Master users can create/manage roles, users, and mappings.                                                                                            |
| **Master User (FGAC Enabled)**           | Full access to OpenSearch cluster and security plugin                                                    | Defined by an IAM ARN or username/password. Mapped to built-in roles: `all_access` (full access) and `security_manager` (manage users/permissions). IAM policies **don’t affect authorization**.                |
| **Cognito Authentication**               | Enables SSO-like authentication via Amazon Cognito                                                      | Requires setup of **User Pool**, **Identity Pool**, and an IAM role with `AmazonOpenSearchServiceCognitoAccess` policy.                                                                                         |
| **User Pool**                            | Handles user directory and login                                                                         | Requires domain name and predefined attributes (e.g., name, email). Configurable for password policies and multi-factor auth. Can optionally integrate external identity providers.                              |
| **Identity Pool**                        | Provides temporary IAM roles post-login                                                                 | Assigns **authenticated** and **unauthenticated** roles to users. Initially enables guest access, but OpenSearch disables it after configuration.                                                              |
| **IAM Role: CognitoAccessForOpenSearch** | Gives OpenSearch permission to manage Cognito integration                                                | AWS-managed policy (`AmazonOpenSearchServiceCognitoAccess`) provides minimum required permissions for OpenSearch to configure Cognito authentication and link identity/user pools.                              |


| **Component**             | **Purpose / Functionality**                                                                                           | **Details**                                                                                                                                                                                                 |
|---------------------------|------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Security Plugin**       | Provides advanced security features like encryption, access control, audit logging, authentication, and multi-tenancy | Core tool used to enforce fine-grained access controls and manage tenants, users, roles, and role mappings                                                                                                  |
| **Tenants**               | Logical spaces to store Dashboards assets (index patterns, visualizations, dashboards)                                 | - **Global Tenant**: Shared across all users<br>- **Private Tenant**: Exclusive to a user<br>- **Custom Tenant**: Admin-created and role-restricted                                                         |
| **Creating Tenants**      | Enables isolation for teams or business units                                                                          | Example: `investment_banking` tenant created using Security API to isolate dashboards and visualizations for investment banking users                                                                       |
| **Creating Roles**        | Define permissions at cluster, index, document, and field levels                                                       | - Access only to `10-k` index<br>- Field-level restriction (e.g., deny `chunk_id`, mask `file_name`)<br>- Tenant-level access to `investment_banking` and `global_tenant` with write permissions              |
| **Field-Level Security**  | Control visibility of fields within an index                                                                           | - `~chunk_id`: field hidden from user<br>- `file_name`: masked (encrypted/hashed)<br>- Only explicitly listed fields are shown                                                                              |
| **Role Mapping**          | Connect IAM roles or Cognito groups to OpenSearch roles                                                                | - IAM role `os-dev-limited-opensearch-role` is mapped to the `investment_banking_role`<br>- This mapping ensures Cognito-authenticated users get appropriate permissions                                    |
| **Master User (via FGAC)**| Super admin that can manage users, roles, and mappings                                                                 | Typically defined via IAM ARN or static credentials; has full access to all features                                                                                                                         |
| **Access Control Model**  | Multi-layered access enforcement                                                                                        | 1. **Network-level** (VPC, NGINX)<br>2. **Domain-level** (access policy)<br>3. **User-level** (FGAC via security plugin, mapped roles)                                                                       |
| **Use Case**              | Secure Dashboards for specific teams (e.g., Investment Banking)                                                        | Teams can view and manage only their own data and dashboards, while being restricted from accessing or modifying data belonging to other business units                                                     |



## Evaluating RAG Performance
- context precision and context recall metrics
- Faithfulness metric : Faithfulness measures the factual consistency of the generated response against the given context.
- Answer relevancy metric: The evaluation metric, answer relevancy, focuses on assessing how pertinent the generated answer is to the given prompt.
- Context recall metric: Context recall measures the extent to which the retrieved context aligns with the annotated answer, treated as the ground truth. It is computed based on the ground truth and the retrieved context.
- Context precision metric: Context Precision is a metric that evaluates whether all the ground truth relevant items present in the contexts are ranked higher or not.


## Chunking strategy
- Fixed Chunking: The granularity of chunks is important. Anything too large may contain irrelevant information and too small may lack sufficient context.
- Semantic Chunking: Semantic chunking should be used in scenarios where maintaining the semantic integrity of the text is crucial.
- Hierarchical chunking: is best suited for complex documents that have a nested or hierarchical structure, such as technical manuals, legal documents, or academic papers with complex formatting and nested tables. 

## Re-ranking
- Custom re-ranking is a technique for refining documents retrieved in response to a user query.





## RAG Applications: Built and deployed Retrieval-Augmented Generation (RAG) applications utilizing vector databases and Mistral AI-Language models with LangChain.
| Question | Answer |
|----------|--------|
| What is a RAG (Retrieval-Augmented Generation) application? | RAG is a technique where external knowledge is retrieved from a database (such as a vector store) and used to augment the generation of responses by an AI model. This enables the model to provide more accurate and up-to-date information by retrieving specific context during inference. |
| How does the integration of vector databases improve RAG applications? | Vector databases store embeddings of documents or data as vectors, enabling semantic search. Instead of exact keyword matching, the vector store finds contextually relevant data, enhancing the model's ability to retrieve relevant information for generation. |
| What is the role of LangChain in building RAG applications? | LangChain is a framework that simplifies the creation of chains of prompts, LLMs, and retrievers. It allows developers to easily connect language models with retrieval systems, enabling complex workflows like RAG applications where context needs to be fetched from external sources. |
| What is Mistral AI, and how is it used in RAG applications? | Mistral AI provides high-performing, open-source language models that can be used for generating text. In RAG applications, these models generate responses based on the context retrieved from vector databases, making them more accurate and relevant. |
| How do vector databases store and retrieve information? | Vector databases store data as high-dimensional embeddings, which are mathematical representations of text. When a query is made, it is transformed into an embedding, and the database retrieves data based on proximity (similarity) to this embedding using techniques like nearest neighbor search. |
| Can you describe how you integrated Mistral AI-Language models into your RAG system? | Mistral AI models were integrated into the RAG workflow as the generation engine. After retrieving relevant context from the vector database, the retrieved data is passed to the Mistral model for generating the final response that incorporates both the model's knowledge and the retrieved information. |
| What challenges did you face while deploying RAG applications? | Some of the challenges include managing the latency involved in retrieving data from vector stores, ensuring that the retrieved information is relevant to the prompt, and effectively integrating the retrieval process with language generation to produce coherent and contextually accurate responses. |
| How does the retrieval process in a RAG application work? | The retrieval process begins with converting the user's query into a vector (embedding). This vector is then used to search the vector database for semantically similar embeddings (documents). The retrieved documents are passed as context to the language model, which then generates the response. |
| What type of use cases are suitable for RAG applications? | RAG applications are ideal for use cases requiring dynamic, up-to-date information retrieval, such as customer support, knowledge base querying, document generation, technical support, and real-time data analysis. |
| How do you evaluate the effectiveness of a RAG application? | Effectiveness is evaluated by measuring the relevance of retrieved information, the coherence of the generated response, and the overall performance (latency and accuracy). Metrics such as precision, recall, and F1 score can be used for the retrieval component, while BLEU or ROUGE scores can be used for generation. |
| What are the advantages of using vector-based search over traditional keyword-based search in RAG applications? | Vector-based search allows for semantic search, meaning it understands the meaning of the text and can retrieve contextually relevant information even if the exact words are not present in the query. This leads to more accurate and meaningful retrieval in response to user queries. |
| How do you ensure the retrieval of relevant documents for a query in a RAG system? | Relevance is ensured by fine-tuning the embedding models that generate vector representations, using similarity metrics (e.g., cosine similarity) to match the query vector to stored vectors, and carefully designing the retrieval logic to focus on the most relevant sections of the knowledge base. |
| What role do embeddings play in the RAG architecture? | Embeddings are vectorized representations of text that capture semantic meaning. They are essential for the retrieval process in RAG, enabling efficient and relevant matching between the query and stored documents in a vector database. |
| Can you explain how chain-of-thought prompting could be combined with RAG? | Chain-of-thought prompting can be used with RAG by first retrieving relevant documents and then guiding the model to reason through the task step-by-step. This could involve a sequence of prompts that allow the model to iteratively refine its response using the retrieved context. |
| How did you manage the interaction between retrieval and generation in the LangChain framework? | The LangChain framework was used to manage the chain between the retrieval and generation components. After receiving the query, the framework invokes the retriever (vector database), fetches relevant documents, and then passes these documents to the LLM to generate a final response. This workflow ensures a seamless transition between retrieval and generation. |



## Factors to Consider When Selecting a Vector Database for LLM RAG

| **Factor**                        | **Description**                                                                 | **Examples of Vector Databases**                                |
|-----------------------------------|---------------------------------------------------------------------------------|----------------------------------------------------------------|
| **Search Efficiency**             | How fast and accurately the database can retrieve relevant vectors.              | Pinecone, Weaviate, FAISS                                      |
| **Scalability**                   | Ability to handle large datasets with millions or billions of vectors.           | Pinecone, Milvus, Vespa                                        |
| **Indexing & Querying Speed**     | Speed of creating indexes and querying them (e.g., ANN techniques like HNSW, IVF)| FAISS, Milvus, Qdrant                                          |
| **Distributed & Fault-Tolerance** | Support for distributed architectures and high availability.                     | Pinecone, Milvus, Vespa                                        |
| **Integration with LLM Pipelines**| How well it integrates with LLM workflows, APIs, and libraries like LangChain.   | Pinecone, Weaviate, Zilliz                                     |
| **Support for Hybrid Search**     | Combination of vector and traditional keyword search for more precise results.   | Vespa, Weaviate, Qdrant                                        |
| **Extensibility & Flexibility**   | Customizable features, ability to adapt to various use cases (RAG, recommendation)| Weaviate, Milvus, Vespa                                        |
| **Real-Time Search**              | Support for real-time updates and searches, important for dynamic datasets.       | Qdrant, Weaviate                                               |
| **Memory & Resource Efficiency**  | Resource usage, especially in terms of memory and disk space, when scaling up.   | FAISS, Pinecone                                                |
| **Data Privacy & Security**       | Secure handling of data, including encryption and compliance with regulations.    | Pinecone, Milvus                                               |
| **Cloud vs On-Premise Support**   | Whether the solution supports cloud-native deployment or on-premise installations.| Pinecone (cloud-native), Weaviate (both cloud and on-premise)   |
| **Community & Support**           | Availability of community, enterprise support, and documentation.                 | Milvus (open-source, large community), Pinecone (enterprise)    |

### Example Databases for LLM RAG Use Cases

| **Database**     | **Key Features**                                                     | **Best for RAG Use Cases**                                      |
|------------------|---------------------------------------------------------------------|-----------------------------------------------------------------|
| **Pinecone**     | Fully managed, scalable, fast approximate nearest neighbor (ANN)     | Fast, scalable retrieval; integrates well with LangChain        |
| **Weaviate**     | Supports hybrid search, modular, open-source, flexible deployment    | Hybrid search, real-time indexing, flexible LLM integration     |
| **Milvus**       | Open-source, distributed, scalable, HNSW/IVF-based indexing          | Large-scale deployments, integration with vector search models  |
| **FAISS**        | Facebook’s open-source library for efficient similarity search       | High-performance local deployments, resource-efficient          |
| **Vespa**        | Hybrid search (vector + text), supports advanced use cases like recommendations | Hybrid search, recommendation engines                           |
| **Qdrant**       | Open-source, optimized for real-time vector search                   | Real-time search, efficient for dynamic datasets                |



# Resources
- https://github.com/aws-samples/generative-ai-applications-foundational-architecture?tab=readme-ov-file
