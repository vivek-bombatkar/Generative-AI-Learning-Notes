

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




| **Code Section**              | **Description** |
|------------------------------|------------------|
| **Index Settings**           | - Enables **KNN search** (`index.knn: true`) using the HNSW algorithm, allowing approximate nearest neighbor search for vector embeddings. <br> - Sets `ef_search: 512` to balance search accuracy and performance. <br> - Uses **5 shards** for distributed storage and scalability. <br> - Defines two analyzers: the default uses standard English tokenization with stopword removal, and a custom `content_analyzer` adds lowercase, stopword, and snowball filters. |
| **Mappings**                 | - `file_name`: A text field with a subfield of keyword type for sorting/filtering. <br> - `chunk_id`: A numeric field identifying document segments. <br> - `content_text`: Uses the `content_analyzer` for advanced text processing; supports position indexing and scoring. <br> - `content_vector_titan_v1`: A 1536-dimensional vector for similarity search using Titan v1 embeddings. <br> - `content_vector_titan_v2:0`: A 1024-dimensional vector for Titan v2. Both use **HNSW** (Hierarchical Navigable Small World) algorithm with `l2` (Euclidean) distance and are optimized using `nmslib`. |
| **Dynamic Templates**        | - Automatically maps fields in `metadata_fields` based on their suffix: <br> - `*_id`: Mapped as keyword (for exact search). <br> - `*_s`: Mapped as text (for full-text search). <br> - `*_t`: Mapped as date with timestamp format. <br> - `*_d`: Mapped as date with simpler date format. This allows flexibility in storing and querying structured metadata. |







# Resources
- https://github.com/aws-samples/generative-ai-applications-foundational-architecture?tab=readme-ov-file
