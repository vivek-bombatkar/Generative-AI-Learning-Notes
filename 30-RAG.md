

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
```
PUT /10-k
{
  "settings": {
    "index.knn": true,
    "index.knn.algo_param.ef_search": 512,
    "number_of_shards": 5,
    "analysis": {
      "analyzer": {
        "default": {
          "type": "standard",
          "stopwords": "_english_"
        },
        "content_analyzer": {
          "type": "custom",
          "tokenizer": "standard",
          "filter": [
            "lowercase",
            "stop",
            "snowball"
          ]
        }
      }
    }
  },
  "mappings": {
    "properties": {
      "file_name": {
        "type": "text",
        "fields": {
          "keyword": {
            "type": "keyword",
            "ignore_above": 256
          }
        }
      },
      "chunk_id": {
        "type": "long"
      },
      "content_text": {
        "type": "text",
        "analyzer": "content_analyzer",
        "norms": true,
        "index_options": "positions"
      },
      "content_vector_titan_v1": {
        "type": "knn_vector",
        "dimension": 1536,
        "method": {
          "name": "hnsw",
          "space_type": "l2",
          "engine": "nmslib",
          "parameters": {
            "ef_construction": 256,
            "m": 16
          }
        }
      },
      "content_vector_titan_v2:0": {
        "type": "knn_vector",
        "dimension": 1024,
        "method": {
          "name": "hnsw",
          "space_type": "l2",
          "engine": "nmslib",
          "parameters": {
            "ef_construction": 256,
            "m": 16
          }
        }
      }
    },
    "dynamic_templates": [
      {
        "strings_id_suffix": {
          "match_pattern": "regex",
          "path_match": "metadata_fields.*",
          "match": ".*_id$",
          "mapping": {
            "type": "keyword",
            "ignore_above": 128
          }
        }
      },
      {
        "strings_s_suffix": {
          "match_pattern": "regex",
          "path_match": "metadata_fields.*",
          "match": ".*_s$",
          "mapping": {
            "type": "text"
          }
        }
      },
      {
        "strings_t_suffix": {
          "match_pattern": "regex",
          "path_match": "metadata_fields.*",
          "match": ".*_t$",
          "mapping": {
            "type": "date",
            "format": "MMM dd yyyy hh:mm:ss a || MMM dd yyyy HH:mm:ss"
          }
        }
      },
      {
        "strings_d_suffix": {
          "match_pattern": "regex",
          "path_match": "metadata_fields.*",
          "match": ".*_d$",
          "mapping": {
            "type": "date",
            "format": "MMM dd yyyy || yyyy/MM/dd"
          }
        }
      }
    ]
  }
}
```

| **Code Section**              | **Description** |
|------------------------------|------------------|
| **Index Settings**           | - Enables **KNN search** (`index.knn: true`) using the HNSW algorithm, allowing approximate nearest neighbor search for vector embeddings. <br> - Sets `ef_search: 512` to balance search accuracy and performance. <br> - Uses **5 shards** for distributed storage and scalability. <br> - Defines two analyzers: the default uses standard English tokenization with stopword removal, and a custom `content_analyzer` adds lowercase, stopword, and snowball filters. |
| **Mappings**                 | - `file_name`: A text field with a subfield of keyword type for sorting/filtering. <br> - `chunk_id`: A numeric field identifying document segments. <br> - `content_text`: Uses the `content_analyzer` for advanced text processing; supports position indexing and scoring. <br> - `content_vector_titan_v1`: A 1536-dimensional vector for similarity search using Titan v1 embeddings. <br> - `content_vector_titan_v2:0`: A 1024-dimensional vector for Titan v2. Both use **HNSW** (Hierarchical Navigable Small World) algorithm with `l2` (Euclidean) distance and are optimized using `nmslib`. |
| **Dynamic Templates**        | - Automatically maps fields in `metadata_fields` based on their suffix: <br> - `*_id`: Mapped as keyword (for exact search). <br> - `*_s`: Mapped as text (for full-text search). <br> - `*_t`: Mapped as date with timestamp format. <br> - `*_d`: Mapped as date with simpler date format. This allows flexibility in storing and querying structured metadata. |


# Chunking
```
from langchain.text_splitter import TokenTextSplitter

chunk_size = 200
chunk_overlap = 100

def token_chunking(content):
    logger.info("Splitting document into chunks")
    text_splitter = TokenTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_text(content)
    return chunks
```
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

```
embedding_endpoint_dictionary = {
    'amazon.titan-embed-text-v1':"amazon.titan-embed-text-v1",
    'amazon.titan-embed-text-v2:0':  "amazon.titan-embed-text-v2:0"
}

def get_embedding(chunk_list):
    result = []
    for chunk in chunk_list:
        result.append({
            'content': chunk
        })
    for model_id, endpoint_name in embedding_endpoint_dictionary.items():
        for i in range(0, len(chunk_list), max_batch):
            try:
                batch = chunk_list[i:i + max_batch]
            except:
                batch = chunk_list[i:]

            for text in batch:
                data = [text]
                json_data = json.dumps(data)

                try:
                    response = sagemaker_runtime_client.invoke_endpoint(
                        EndpointName=endpoint_name,
                        ContentType=content_type,
                        Body=json_data,
                    )

                    if response["ResponseMetadata"]["HTTPStatusCode"] != 200:
                        logger.error(f"Error invoking endpoint {endpoint_name}: {response}")
                        raise Exception(f"Error invoking endpoint {endpoint_name}")

                    embeddings = json.loads(response["Body"].read().decode("utf-8"))
                    result[i + batch.index(text)][model_id] = embeddings
                except Exception as e:
                    logger.info(f"Error invoking endpoint {endpoint_name}: {e}")
                    raise

    return result
```

# Upload documents into OpenSearch
After chunking and embedding the documents, the next step is to upload them to OpenSearch. This process requires formatting the data to match the structure of the OpenSearch index.

By carefully aligning the uploaded data with the OpenSearch index structure, we ensure optimal performance in data retrieval and querying operations. This structured approach forms the foundation for building powerful search and analytics capabilities on top of the uploaded data.

```
def index_document(index_name, embeddings, object_key, metadata=None):
    file_name = object_key.split('/')[-1]
    file_base_name = file_name.split('.')[0]
    clean_file_base_name = file_base_name.replace('content', '')
    logger.info(f"Indexing content and vector-embedding into OpenSearch index {index_name} for file {file_name}")
    print("LEN CHUNKS in index_document ", len(embeddings))
    documents = []
    
    # Create a document for each chunk of content and vector-embedding
    for i, embeddings_dictionary in enumerate(embeddings):
        document = {
            '_op_type': 'index',
            'file_name': clean_file_base_name,
            'chunk_id': i,
            'content_text': embeddings_dictionary['content'],
        }
        for model_id, endpoint_name in embedding_endpoint_dictionary.items():
            vector_embedding = embeddings_dictionary[model_id][0]
            document[f'content_vector_{"_".join(model_id.split("/")[-1].split("-")[:2])}'] = vector_embedding


        # Process and add metadata fields to the document
        if metadata and "fields" in metadata:
            metadata_fields = {}
            for field in metadata["fields"]:
                if 'name' in field:
                    if 'value' in field:
                        metadata_fields[field['name']] = field['value']
                    elif 'values' in field:  
                        metadata_fields[field['name']] = field['values']
                    else:
                        logger.warning(f"Field missing 'value' or 'values': {field}")
                else:
                    logger.warning(f"Field missing 'name': {field}")

            document['metadata_fields'] = metadata_fields
        
        # Add the document to the list of documents to be indexed
        documents.append(document)

    logger.info(f"Complete document to be indexed: {len(documents)}")

    try:
        logger.info(f"Indexing document into OpenSearch index {index_name}")
        
        # Bulk index the documents into the OpenSearch index
        success, failed = bulk(opensearch, documents, index=index_name)
        logger.info(f'Indexed {success} documents successfully.')
        logger.info(f'Indexed {failed} documents failed.')
    except Exception as e:
        logger.error(f"Error indexing document: {e}")
```

# Performing Basic Search Queries in Open Search


| **Query Type**             | **Objective**                                                                                           | **Query Code (Elasticsearch DSL)**                                                                                                                                                                                                                                                                                                                                                                                                                         | **Guidance**                                                                                                                                                             |
|----------------------------|---------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **1. Simple Text Search**  | Retrieve documents containing a specific phrase.                                                        | ```json<br>GET /10-k/_search<br>{<br>  "_source": {<br>    "excludes": ["content_vector_titan_v1", "content_vector_titan_v2:0", "metadata_fields"]<br>  },<br>  "query": {<br>    "match": {<br>      "content_text": {<br>        "query": "3M revenue contribution from Asia-Pacific 2023"<br>      }<br>    }<br>  }<br>}```                                                                                                     | Searches for a specific phrase in `content_text`. Vector and metadata fields are excluded.                                         |
| **2. Fuzzy Search**        | Search for approximate matches, useful for handling typos.                                              | ```json<br>GET /10-k/_search<br>{<br>  "_source": {<br>    "excludes": ["content_vector_titan_v1", "content_vector_titan_v2:0", "metadata_fields"]<br>  },<br>  "query": {<br>    "fuzzy": {<br>      "content_text": {<br>        "value": "aparel",<br>        "fuzziness": "AUTO"<br>      }<br>    }<br>  }<br>}```                                                                                                  | Finds near-matches for `"aparel"` using fuzzy logic. Useful for typo handling.                                                    |
| **3. Match Phrase Query**  | Find documents with an exact phrase for high relevance.                                                 | ```json<br>GET /10-k/_search<br>{<br>  "_source": {<br>    "excludes": ["content_vector_titan_v1", "content_vector_titan_v2:0", "metadata_fields"]<br>  },<br>  "query": {<br>    "bool": {<br>      "must": [<br>        {<br>          "match": {<br>            "content_text": "This brand includes a wide assortment of baby and toddler apparel"<br>          }<br>        }<br>      ]<br>    }<br>  }<br>}``` | Matches exact long-form phrases in the content field. Uses `bool` with `must` clause.                                              |
| **4. Boost Query Search**  | Prioritize certain fields over others for fine-tuned relevance.                                         | ```json<br>GET /10-k/_search<br>{<br>  "_source": {<br>    "excludes": ["content_vector_titan_v1", "content_vector_titan_v2:0", "metadata_fields"]<br>  },<br>  "query": {<br>    "bool": {<br>      "must": [<br>        {<br>          "match": {<br>            "metadata_fields.exchange_id": {<br>              "query": "NYSE",<br>              "boost": 3<br>            }<br>          }<br>        },<br>        {<br>          "range": {<br>            "metadata_fields.document_period_end_date_d": {<br>              "gte": "Dec 01 2021",<br>              "lte": "Dec 31 2023"<br>            }<br>          }<br>        },<br>        {<br>          "match": {<br>            "content_text": {<br>              "query": "financial statements",<br>              "boost": 1<br>            }<br>          }<br>        }<br>      ]<br>    }<br>  }<br>}``` | Boosts `"exchange_id"` match over others. Filters by date range and text relevance.                                                |
| **5. Metadata Search**     | Target specific metadata fields, essential for filtering by document type and date range.              | ```json<br>GET /10-k/_search<br>{<br>  "_source": {<br>    "excludes": ["content_vector_titan_v1", "content_vector_titan_v2:0", "metadata_fields"]<br>  },<br>  "query": {<br>    "bool": {<br>      "must": [<br>        {<br>          "match": {<br>            "metadata_fields.form_type_s": "10-K"<br>          }<br>        },<br>        {<br>          "range": {<br>            "metadata_fields.filed_as_of_date_d": {<br>              "gte": "Dec 01 2021",<br>              "lte": "Aug 31 2024"<br>            }<br>          }<br>        }<br>      ]<br>    }<br>  }<br>}``` | Filters by `"form_type_s": "10-K"` and date between Dec 2021 and Aug 2024. Enables precise metadata-level filtering.                |


<table border="1" cellspacing="0" cellpadding="6">
  <thead>
    <tr>
      <th>Query Type</th>
      <th>Objective</th>
      <th>Query Code (Elasticsearch DSL)</th>
      <th>Guidance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><b>1. Simple Text Search</b></td>
      <td>Retrieve documents containing a specific phrase.</td>
      <td>
        <pre>
GET /10-k/_search
{
  "_source": {
    "excludes": ["content_vector_titan_v1", "content_vector_titan_v2:0", "metadata_fields"]
  },
  "query": {
    "match": {
      "content_text": {
        "query": "3M revenue contribution from Asia-Pacific 2023"
      }
    }
  }
}
        </pre>
      </td>
      <td>Searches for a specific phrase in <code>content_text</code>. Vector and metadata fields are excluded.</td>
    </tr>

    <tr>
      <td><b>2. Fuzzy Search</b></td>
      <td>Search for approximate matches, useful for handling typos.</td>
      <td>
        <pre>
GET /10-k/_search
{
  "_source": {
    "excludes": ["content_vector_titan_v1", "content_vector_titan_v2:0", "metadata_fields"]
  },
  "query": {
    "fuzzy": {
      "content_text": {
        "value": "aparel",
        "fuzziness": "AUTO"
      }
    }
  }
}
        </pre>
      </td>
      <td>Finds near-matches for <code>"aparel"</code> using fuzzy logic. Useful for typo handling.</td>
    </tr>

    <tr>
      <td><b>3. Match Phrase Query</b></td>
      <td>Find documents with an exact phrase for high relevance.</td>
      <td>
        <pre>
GET /10-k/_search
{
  "_source": {
    "excludes": ["content_vector_titan_v1", "content_vector_titan_v2:0", "metadata_fields"]
  },
  "query": {
    "bool": {
      "must": [
        {
          "match": {
            "content_text": "This brand includes a wide assortment of baby and toddler apparel"
          }
        }
      ]
    }
  }
}
        </pre>
      </td>
      <td>Matches exact long-form phrases in the content field using <code>bool</code> and <code>must</code>.</td>
    </tr>

    <tr>
      <td><b>4. Boost Query Search</b></td>
      <td>Prioritize certain fields over others for fine-tuned relevance.</td>
      <td>
        <pre>
GET /10-k/_search
{
  "_source": {
    "excludes": ["content_vector_titan_v1", "content_vector_titan_v2:0", "metadata_fields"]
  },
  "query": {
    "bool": {
      "must": [
        {
          "match": {
            "metadata_fields.exchange_id": {
              "query": "NYSE",
              "boost": 3
            }
          }
        },
        {
          "range": {
            "metadata_fields.document_period_end_date_d": {
              "gte": "Dec 01 2021",
              "lte": "Dec 31 2023"
            }
          }
        },
        {
          "match": {
            "content_text": {
              "query": "financial statements",
              "boost": 1
            }
          }
        }
      ]
    }
  }
}
        </pre>
      </td>
      <td>Boosts match on <code>exchange_id</code>. Adds date range and content-based scoring.</td>
    </tr>

    <tr>
      <td><b>5. Metadata Search</b></td>
      <td>Target specific metadata fields, essential for filtering by document type and date range.</td>
      <td>
        <pre>
GET /10-k/_search
{
  "_source": {
    "excludes": ["content_vector_titan_v1", "content_vector_titan_v2:0", "metadata_fields"]
  },
  "query": {
    "bool": {
      "must": [
        {
          "match": {
            "metadata_fields.form_type_s": "10-K"
          }
        },
        {
          "range": {
            "metadata_fields.filed_as_of_date_d": {
              "gte": "Dec 01 2021",
              "lte": "Aug 31 2024"
            }
          }
        }
      ]
    }
  }
}
        </pre>
      </td>
      <td>Filters for <code>form_type_s = 10-K</code> and a specific filing date range. Ideal for compliance or historical review.</td>
    </tr>
  </tbody>
</table>


# Resources
- https://github.com/aws-samples/generative-ai-applications-foundational-architecture?tab=readme-ov-file
