# RAG Project

The two primary goals of the AI architecture known as retrieval-augmented generation, or **RAG**, are to increase the caliber of responses produced by linking the model to an outside knowledge base and guarantee that users may access the model's sources to verify the veracity of the answers. By using RAG to connect the huge language model to customized data sources from which it may retrieve information, we can also guarantee that the model has access to proprietary data. This project involves using various tools to create a retrieval-augmented generation model that can answer questions about cryptocurrencies. The tools used are:

- **[Chromadb](https://www.trychroma.com/):** An open-source embedding database that allows us to plug LLMs to knowledge bases. It allows us to store and query embeddings and their metadata.
- **[LangChain](https://www.langchain.com/):** A framework that allows us to develop several applications powered by LLMs.
- **[Sentence Transformers](https://pypi.org/project/sentence-transformers/):** A framework that provides an easy method to compute dense vector representations for sentences, paragraphs, and images by leveraging pre-trained transformer models.
- **[bitsandbytes](https://github.com/TimDettmers/bitsandbytes):** A library designed to optimize the training and deployment of large models through 4-bit quantization of the model's weights, reducing memory footprint and enhancing memory efficiency.
