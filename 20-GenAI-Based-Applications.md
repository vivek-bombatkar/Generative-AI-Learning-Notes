
# 20 GenAI Based Applications

#  Typical components of a generative AI application

![Foundation Model Architecture](https://render.skillbuilder.aws/cds/4fdd8f7e-99d9-4ac7-b5e4-f3967537eb64/assets/FoundationModelCircle_NOPROCESS_.png)

# Prompt history store
- Many FMs have a limited context window, which means you can only pass a fixed length of data as input.
- Storing state information in a multiple-turn conversation becomes a problem when the conversation exceeds the context window.
- To solve this problem, you can implement a prompt history store.
- It can persist the conversation state, making it possible to have a long-term history of the conversation.
