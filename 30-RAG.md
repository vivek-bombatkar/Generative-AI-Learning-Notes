

# Retrieval Augmented Generation (RAG)
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


# Resources
- https://github.com/aws-samples/generative-ai-applications-foundational-architecture?tab=readme-ov-file
