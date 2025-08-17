# %%
# Import libraries
import yaml
import torch
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from utils import get_documents, create_vector_store, create_rag_chain
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TextIteratorStreamer,
    StoppingCriteriaList,
    GenerationConfig,
)
from utils import StopOnKeyword
from threading import Thread


# %%
# Load configuration from 'yaml' file
with open("./config/config.yaml", "r", encoding="utf-8") as file:
    config = yaml.safe_load(file)

# Device
DEVICE = torch.device(config["device"])

# Embedding model and LLM
EMBED_MODEL = config["embed_model"]
LLM = config["llm"]

# Paths
DOCUMENTS_PATH = config["documents_path"]
VECTORDB_PATH = config["vectordb_path"]

# EOS token character
STOP_KEYWORD = config["stop_keyword"]

# TextSplitter
CHUNK_SIZE = config["chunk_size"]
CHUNK_OVERLAP = config["chunk_overlap"]

# GenerationConfig
TEMPERATURE = config["temperature"]
MAX_NEW_TOKENS = config["max_new_tokens"]


# %%
# Embedding model
embed_model = HuggingFaceEmbeddings(model_name=EMBED_MODEL,
                                    model_kwargs={'device': DEVICE},
                                    encode_kwargs={'normalize_embeddings': True})

text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE,
                                               chunk_overlap=CHUNK_OVERLAP,
                                               length_function=len)

# Vector store creation
documents = get_documents(extension=["pdf", "txt"], documents_path=DOCUMENTS_PATH)
vectorstore = create_vector_store(text_splitter=text_splitter,
                                  documents=documents,
                                  embed_model=embed_model,
                                  vectordb_path=VECTORDB_PATH)
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})

# LLM setup
llm = AutoModelForCausalLM.from_pretrained(LLM).to(DEVICE)
tokenizer = AutoTokenizer.from_pretrained(LLM)
streamer = TextIteratorStreamer(tokenizer, skip_prompt=True)
stop_criteria = StoppingCriteriaList([StopOnKeyword(tokenizer, stop_keyword=STOP_KEYWORD)])
generation_config = GenerationConfig(temperature=TEMPERATURE,
                                     max_new_tokens=MAX_NEW_TOKENS,
                                     do_sample=True,
                                     eos_token_id=tokenizer.eos_token_id)


# %%
conversation_history = ""


# Main loop for user interaction
while True:
    user_input = input("Input Question: ")
    if user_input.lower() in ["exit", "quit"]:
        break

    # Prepare the RAG prompt
    formatted_question = f"{user_input}\nWhen you are finished answering, please say '{STOP_KEYWORD}'."
    rag_prompt, _ = create_rag_chain(retriever=retriever, question=formatted_question)
    full_prompt = f"{conversation_history}\n\n{rag_prompt}"
    inputs = tokenizer(full_prompt, return_tensors="pt").to(DEVICE)

    # Create text generation thread
    thread = Thread(target=lambda: llm.generate(
        **inputs,
        streamer=streamer,
        stopping_criteria=stop_criteria,
        generation_config=generation_config
    ))
    thread.start()

    # Print
    print("\n===== Generated Answer: =====\n")
    current_response = ""
    for token in streamer:
        print(token, end="", flush=True)
        current_response += token
    thread.join()

    # Update conversation history
    conversation_history += f"\nUser: {user_input}\nModel: {current_response.strip()}"
    print("\n" + "="*40)


# %%
# Clean up GPU
if torch.cuda.is_available():
    llm.cpu()
    del llm
    torch.cuda.empty_cache()


# %%
