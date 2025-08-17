import glob
import os
import yaml
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from transformers import StoppingCriteria


def get_documents(extension: list, documents_path: str) -> list:
    docs: list = []
    
    for ext in extension:

        files = glob.glob(os.path.join(documents_path, f"*.{ext}"))
        print(f"Found {len(files)} files with extension {ext}.")
        if not files:
            print(f"No files found with extension {ext} in {documents_path}.")
            continue

        for file in files:
            try:
                if file.endswith('.pdf'):
                    loader = PyPDFLoader(file)
                elif file.endswith('.txt'):
                    loader = TextLoader(file, encoding="utf-8")
            
                doc = loader.load()
                docs.extend(doc)
            except Exception as e:
                print(f"Error loading {file}: {e}")
        return docs


def create_vector_store(
        text_splitter: RecursiveCharacterTextSplitter,
        documents: list,
        embed_model: HuggingFaceEmbeddings,
        vectordb_path: str
) -> Chroma:
    splits = text_splitter.split_documents(documents)

    vectorstore = Chroma.from_documents(documents=splits,
                                        embedding=embed_model,
                                        persist_directory=vectordb_path)
    return vectorstore


def create_rag_chain(retriever, question: str):
    # Document retrieval
    relevant_docs = retriever.invoke(question)
    context = "\n\n".join(f"Document: {i+1}\n{doc.page_content}" 
                          for i, doc in enumerate(relevant_docs))

    # RAG prompt generation
    with open("./config/config.yaml", "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
        
    rag_prompt = f"""
===== Instructions =====
{config["rag_prompt"]}

===== Reference =====
{context}

===== Question =====
{question}

===== Answer =====
"""
    return rag_prompt, relevant_docs


class StopOnKeyword(StoppingCriteria):
    def __init__(self, tokenizer, stop_keyword: str):
        self.tokenizer = tokenizer
        self.stop_ids = tokenizer.encode(stop_keyword, add_special_tokens=False)
        self.start_len = None

    def __call__(self, input_ids, scores, **kwargs):
        if self.start_len is None:
            self.start_len = input_ids.shape[1]
            return False
        
        gen = input_ids[0, self.start_len:].tolist()
        if len(gen) < len(self.stop_ids):
            return False
        return gen[-len(self.stop_ids):] == self.stop_ids

