# Data Handling
import transformers
import pandas as pd
import logging
import os
from torch import bfloat16

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# LangChain
from langchain.llms import HuggingFacePipeline
from langchain.document_loaders import DataFrameLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma

data_dir = "data/wikipedia_crypto_articles.csv"
model_id = "mistralai/Mistral-7B-v0.1"
quantization_config = transformers.BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=bfloat16,
)
embedding_config = {
    "dataset_dir": data_dir,
    "model_id": "sentence-transformers/all-MiniLM-L6-v2",
    "chunk_size": 1000,
    "chunk_overlap": 20,
    "chroma_db_path": "chroma_db",
}
pipeline_config = {
    "max_length": 1024,
    "temperature": 0.5,
    "no_repeat_ngram_size": 3,
    "do_sample": False,
}


def create_simple_logger(logger_name, level="info"):
    level_str_to_int = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
    }
    logger = logging.getLogger(logger_name)
    logger.handlers.clear()
    logger.setLevel(level_str_to_int[level])
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


class RAG:
    """Implementation of the RAG model for question answering."""

    def __init__(
        self,
        llm_model_id=model_id,
        quantization_config=quantization_config,
        embedding_config=embedding_config,
        pipeline_config=pipeline_config,
        debug=False,
        logger=None,
    ) -> None:
        """Initializes the RAG model.

        Parameters
        ----------
        llm_model_id : str
            The model id of the language model to use.
        quantization_config : transformers.BitsAndBytesConfig
            The quantization configuration for the language model.
        embedding_config : dict
            The configuration for the embedding model. Must have the following keys:
            - dataset_dir: str
                The path to the dataset to use for creating the embedding database.
            - model_id: str
                The model id of the embedding model to use.
            - chunk_size: int
                The size of the chunks to split the dataset into.
            - chunk_overlap: int
                The number of characters to overlap between chunks.
            - chroma_db_path: str
                The path to the directory to store the embedding database.
        pipeline_config : dict
            The configuration for the HuggingFace pipeline. Must have the following keys:
            - max_length: int
                The maximum length of the generated text.
            - temperature: float
                The temperature for the sampling.
            - no_repeat_ngram_size: int
                The number of ngrams to avoid repeating.
            - do_sample: bool
                Whether to sample or not.
        debug : bool
            Whether to print debug messages or not. This will also set the log level to debug.
        logger : logging.Logger
            The logger to use. If None, a simple logger will be created.
        """
        self.llm_model_id = llm_model_id
        self.quantization_config = quantization_config
        self.embedding_config = embedding_config
        self.pipeline_config = pipeline_config
        self.debug = debug
        self.chroma_database = None
        self.hf_pipeline = None
        self.retrieval_qa = None
        log_level = "debug" if debug else "info"
        self.logger = logger or create_simple_logger("RAG", level=log_level)

    def create_embedding(self, overwrite=False):
        """Creates the embedding database and the chroma database.

        Parameters
        ----------
        overwrite : bool
            Whether to overwrite the existing database or not.

        Returns
        -------
        Chroma
            The chroma database.
        """
        if self.chroma_database is not None and not overwrite:
            self.logger.info("Using existing Chroma database.")
            return self.chroma_database

        self.logger.debug("Loading embedding model.")
        embedding_model = HuggingFaceEmbeddings(
            model_name=self.embedding_config["model_id"]
        )

        self.logger.debug("Loading dataset and splitting into chunks.")
        df = pd.read_csv(self.embedding_config["dataset_dir"])
        articles = DataFrameLoader(df, page_content_column="title").load()
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.embedding_config["chunk_size"],
            chunk_overlap=self.embedding_config["chunk_overlap"],
        )
        documents = splitter.split_documents(articles)

        self.logger.debug("Creating Chroma database.")
        if os.path.exists(self.embedding_config["chroma_db_path"]):
            self.logger.info("Removing the existing chroma db folder.")
            os.system(f"rm -r {self.embedding_config['chroma_db_path']}")
        chroma_database = Chroma.from_documents(
            documents,
            embedding_model,
            persist_directory=self.embedding_config["chroma_db_path"],
        )
        self.chroma_database = chroma_database

        self.logger.info("Chroma database created.")
        return chroma_database

    def create_llm_pipeline(self, overwrite=False):
        """Creates the HuggingFace pipeline for the language model.

        Parameters
        ----------
        overwrite : bool
            Whether to overwrite the existing pipeline or not.

        Returns
        -------
        HuggingFacePipeline
            The HuggingFace pipeline.
        """
        if self.hf_pipeline is not None and not overwrite:
            self.logger.info("Using existing HuggingFace pipeline.")
            return self.hf_pipeline

        self.logger.debug("Loading model and tokenizer.")
        tokenizer = AutoTokenizer.from_pretrained(self.llm_model_id)
        llm = AutoModelForCausalLM.from_pretrained(
            self.llm_model_id, quantization_config=quantization_config
        )

        self.logger.debug("Creating HuggingFace pipeline.")
        pipe = pipeline(
            "text-generation",
            model=llm,
            tokenizer=tokenizer,
            **self.pipeline_config,
        )
        hf_pipeline = HuggingFacePipeline(pipeline=pipe)
        self.hf_pipeline = hf_pipeline
        self.logger.info("HuggingFace pipeline created.")
        return hf_pipeline

    def create_retrieval_qa(self, overwrite=False):
        """Creates the RetrievalQA pipeline.

        Parameters
        ----------
        overwrite : bool
            Whether to overwrite the existing pipeline or not.

        Returns
        -------
        RetrievalQA
            The RetrievalQA pipeline.
        """
        if self.retrieval_qa is not None and not overwrite:
            self.logger.debug("Using existing RetrievalQA pipeline.")
            return self.retrieval_qa

        chroma_database = self.create_embedding()
        retriever = chroma_database.as_retriever()
        llm_pipeline = self.create_llm_pipeline()

        self.logger.debug("Creating RetrievalQA pipeline.")
        retrieval_qa = RetrievalQA.from_chain_type(
            llm=llm_pipeline,
            retriever=retriever,
            chain_type="stuff",
            verbose=self.debug,
        )
        self.retrieval_qa = retrieval_qa
        self.logger.info("RetrievalQA pipeline created.")
        return retrieval_qa

    def answer(self, question):
        """Answers the given question.

        Parameters
        ----------
        question : str
            The question to answer.

        Returns
        -------
        str
            The answer to the question.
        """
        retrieval_qa = self.create_retrieval_qa()
        self.logger.debug("Asking question.")
        answer = retrieval_qa.invoke(question)["result"]
        print(f"\033[1mQuestion:\033[0m {question}\n")
        print(f"\033[1mAnswer:\033[0m ", answer)
        return answer
