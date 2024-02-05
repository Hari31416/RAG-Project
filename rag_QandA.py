import argparse


def get_default(value, default):
    """Gets the default value if the value is None."""
    if value is None:
        return default
    return value


def create_quantization_config(quantization=False):
    """Creates the quantization config for the model."""
    import transformers
    from torch import bfloat16

    if not quantization:
        return None

    quantization_config = transformers.BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=bfloat16,
    )
    return quantization_config


def create_embedding_config(args):
    """Creates the embedding config for the model."""
    embedding_config = {
        "dataset_dir": get_default(
            args.dataset_dir, "data/wikipedia_crypto_articles.csv"
        ),
        "model_id": get_default(
            args.embedding_model_id, "sentence-transformers/all-MiniLM-L6-v2"
        ),
        "chunk_size": get_default(args.chunk_size, 1000),
        "chunk_overlap": get_default(args.chunk_overlap, 20),
        "chroma_db_path": get_default(args.chroma_db_path, "chroma_db"),
    }
    return embedding_config


def create_pipeline_config(args):
    """Creates the pipeline config for the model."""
    pipeline_config = {
        "max_length": get_default(args.max_length, 1024),
        "temperature": get_default(args.temperature, 1.0),
        "no_repeat_ngram_size": get_default(args.no_repeat_ngram_size, 3),
        "do_sample": get_default(args.do_sample, True),
    }
    return pipeline_config


def create_config(args):
    """Creates the config for the model."""
    config = {
        "quantization_config": create_quantization_config(args.quantization),
        "embedding_config": create_embedding_config(args),
        "pipeline_config": create_pipeline_config(args),
    }
    return config


def main(args):
    """Main function for the script. Takes in the args and runs the model."""
    from chains import RAG

    config = create_config(args)

    llm_model_id = get_default(args.llm_model_id, "mistralai/Mistral-7B-Instruct-v0.1")
    debug = get_default(args.debug, False)
    rag = RAG(
        llm_model_id=llm_model_id,
        quantization_config=config["quantization_config"],
        embedding_config=config["embedding_config"],
        pipeline_config=config["pipeline_config"],
        debug=debug,
    )
    print("Preparing RAG")
    rag.create_retrieval_qa()
    print("RAG ready. Ask a question! Enter q to quit.")

    while True:
        question = input("Question: ")
        if question == "q":
            break
        _ = rag.answer(question)
        print("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--llm_model_id",
        "-llm",
        type=str,
        help="Language model id to use for generation",
        default=None,
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        help="Directory containing the dataset",
        default=None,
    )
    parser.add_argument(
        "--embedding_model_id",
        "-emb",
        type=str,
        help="Embedding model id to use for retrieval",
        default=None,
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        help="Chunk size for the embedding model",
        default=None,
    )
    parser.add_argument(
        "--quantization",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--chunk_overlap",
        type=int,
        help="Chunk overlap for the embedding model",
        default=None,
    )
    parser.add_argument(
        "--chroma_db_path",
        type=str,
        help="Path to the chroma db",
        default=None,
    )
    parser.add_argument(
        "--max_length",
        type=int,
        help="Maximum length for the generation",
        default=None,
    )
    parser.add_argument(
        "--temperature",
        type=float,
        help="Temperature for the generation",
        default=None,
    )
    parser.add_argument(
        "--no_repeat_ngram_size",
        action=argparse.BooleanOptionalAction,
        help="No repeat ngram size for the generation",
        default=True,
    )
    parser.add_argument(
        "--do_sample",
        help="Do sample for the generation",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--debug",
        type=bool,
        help="Debug mode",
        default=None,
    )
    args = parser.parse_args()
    main(args)
