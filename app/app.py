from fastapi import FastAPI
from pydantic import BaseModel

from GPTJ_model import ask as ask_gpt
from gemini_embeddings_groq_model import ask as ask_gemini
from huggingface_embeddings_model import ask as ask_hf

class Validate(BaseModel):
    question: str

app = FastAPI()

@app.post("/")
def index():    
    return "Welcome to QnA system"

@app.post("/GPT")
def ask_question_GPT(question:Validate):
    """Function that takes user question as an input and returns the answer from the GPT-J embeddings model

    Args:
        question (Validate): user query
    """

    ans_gpt = ask_gpt(question)    
    return ans_gpt

@app.post("/Gemini")
def ask_question_gemini(question:Validate):
    """Function that takes user question as an input and returns the answer from the Gemini embeddings model

    Args:
        question (Validate): user query
    """

    ans_gemini = ask_gemini(question)
    return ans_gemini

@app.post("/Huggingface")
def ask_question_hf(question:Validate):
    """Function that takes user question as an input and returns the answer from the Huggingface embeddings model

    Args:
        question (Validate): user query
    """
    ans_hf = ask_hf(question)
    return ans_hf
