from llama_index.core import (
    SimpleDirectoryReader,
    StorageContext,
    ServiceContext,
    load_index_from_storage
)


from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.embeddings.google import GeminiEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.groq import Groq
from llama_index.core import Settings
import google.generativeai as genai



reader = SimpleDirectoryReader(input_dir = "/data", input_files = ["List of files here"], required_exts = [".pdf", ".docx", ".xlsx"])
documents = reader.load_data()

embed_model = GeminiEmbedding(model_name = "models/embedding-001")

genai.configure(api_key = 'GEMINI_API_KEY')


splitter = SemanticSplitterNodeParser(
              buffer_size = 1,
              breakpoint_percentile_threshold = 95,
              embed_model = embed_model
           )

nodes = splitter.get_nodes_from_documents(documents, show_progress = True)


Settings.text_splitter = SentenceSplitter(chunk_size = 1024)
Settings.chunk_overlap = 20
Settings.transformations = [SentenceSplitter(chunk_size = 1024)]
# maximum input size to the LLM
Settings.context_window = 4096
# number of tokens reserved for text generation.
Settings.num_output = 256

storage_context = StorageContext.from_defaults(persist_dir = "./storage")

llm = Groq(model = "llama3-70b-8192", api_key = "GROQ_API_KEY")

service_context = ServiceContext.from_defaults(
    chunk_size = 512,
    llm = llm,
    embed_model = embed_model
)


index = load_index_from_storage(
            storage_context,
            service_context = service_context
        )

query_engine = index.as_query_engine(
                  service_context = service_context,
                  similarity_top_k = 10,
                )

def ask(query):
    response = query_engine.query(query)
    return response