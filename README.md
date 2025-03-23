# Framework for Text Understanding and Generation

This project provides a flexible framework for various text understanding and generation tasks, leveraging machine learning techniques, vector databases, and large language models (LLMs). While primarily designed for dialogue summarization, it can be adapted for other applications such as question answering, text classification, and content generation.

## Overview

The framework consists of several modules:

1.  **Data Loading**: Reads seed and unlabeled datasets from local files.
2.  **Text Preprocessing**: Cleans and preprocesses text data using techniques like HTML tag removal, URL removal, chat word conversion, stop word removal, spelling correction, and emoji rewriting.
3.  **Embedding Generation**: Generates embeddings for the preprocessed text using transformer models.
4.  **Vector Database Interaction**: Stores and retrieves embeddings from a vector database for similarity search.
5.  **LLM Interaction**: Uses LLMs to generate summaries for dialogues based on nearest neighbor examples retrieved from the vector database.
6.  **Clustering**: Clusters embeddings to identify groups of similar dialogues.

```

You can install these dependencies using pip:

```bash
pip install -r requirements.txt
```

## Setup

1.  **Install Dependencies**: Install the required libraries using the command above.
2.  **Environment Variables**: Set the following environment variables in a `.env` file:

    *   `MILVUS_URI`: URI for connecting to the Milvus database.
    *   `MILVUS_TOKEN`: Authentication token for Milvus.
    *   `OPENROUTER_API_KEY`: API key for accessing the OpenRouter LLM service.
3.  **Data**: Place your seed and unlabeled data in the `data/` directory. 

## Usage

The main workflow is orchestrated in the `algorithm.ipynb` Jupyter Notebook. Follow the steps in the notebook to:

1.  Load and preprocess the data.
2.  Generate embeddings for the text data.
3.  Store the embeddings in the Milvus vector database.
4.  Iterate through the unlabeled data, retrieve nearest neighbors from the vector database, and generate summaries using an LLM.
5.  Update the vector database with the generated summaries.


## Configuration

*   `DEFAULT_MODEL`: Default model name for embedding generation.
*   `TASK_DESCRIPTION`, `EMPTY_TASK_DESCRIPTION`, `INPUT_PREFIX`, `OUTPUT_PREFIX`: Prompt settings for LLM interaction.

## Contributing

Feel free to contribute to this project by submitting issues or pull requests.

