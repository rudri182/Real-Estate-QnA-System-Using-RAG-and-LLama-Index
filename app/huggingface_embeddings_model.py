import torch

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, ServiceContext 
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index.llms.huggingface import HuggingFaceLLM
from huggingface_hub.hf_api import HfFolder

# load data
reader = SimpleDirectoryReader(input_dir="/data", input_files=["List of files here"], required_exts=[".pdf", ".docx", ".xlsx"])
documents = reader.load_data()

# LLM instruction 
system_prompt = """
As a Q&A assistant, your objective is to provide accurate answers to questions based on the given instructions and context.
"""

HfFolder.save_token('HF_TOKEN_HERE')

# initialize LLM
llm = HuggingFaceLLM(
    context_window = 4096,
    max_new_tokens = 256,
    generate_kwargs = {"temperature": 0.0, "do_sample": False},
    system_prompt = system_prompt,
    tokenizer_name = "openbmb/MiniCPM-Llama3-V-2_5",
    model_name = "openbmb/MiniCPM-Llama3-V-2_5",
    device_map = "auto",
    # loading model in 8bit for reducing memory
    model_kwargs = {"torch_dtype": torch.float16 , "load_in_8bit":True}
)

# embeddings model
embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/multi-qa-mpnet-base-dot-v1")

# initialize utility function of LLAMA index
service_context = ServiceContext.from_defaults(
    chunk_size = 1024,
    llm = llm,
    embed_model = embed_model
)

# index the documents uisng vector index 
index = VectorStoreIndex.from_documents(documents, service_context=service_context)

# to ask question to the data
query_engine=index.as_query_engine()

# ask question
def ask(question):
    response = query_engine.query(question)
    return response