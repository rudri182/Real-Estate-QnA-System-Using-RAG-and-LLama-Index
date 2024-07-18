from transformers import GPTJForCausalLM, GPT2Tokenizer
from llama_index import LLMPredictor, QuestionAnswerPrompt, ServiceContext
from llama_index.core import SimpleDirectoryReader, GPTVectorStoreIndex, PromptHelper


# Load the GPT-J model and tokenizer
model = GPTJForCausalLM.from_pretrained("EleutherAI/gpt-j-6B")
tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-j-6B")

# Create the LLMPredictor
llm_predictor = LLMPredictor(llm=model, tokenizer=tokenizer)

# Load the real estate project data
reader = SimpleDirectoryReader(input_dir="/data", input_files=["List of files here"], required_exts=[".pdf", ".docx", ".xlsx"])
documents = reader.load_data()


# Set the prompt helper
prompt_helper = PromptHelper(max_input_size=4096, num_output=512, max_chunk_overlap=20)

# Create the index
index = GPTVectorStoreIndex.from_documents(documents, llm_predictor=llm_predictor, prompt_helper=prompt_helper)


# Define the prompt
system_prompt = "Answer the question using the provided context. If you cannot find a relevant answer, say 'I don't have enough information to answer that."
qa_prompt = QuestionAnswerPrompt(system_prompt)

# Create the service context
service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, prompt_helper=prompt_helper)

def ask(query):
    response = index.query(query, service_context=service_context, text_qa_template=qa_prompt)
    return response