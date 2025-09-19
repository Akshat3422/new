from fastapi import FastAPI, APIRouter, UploadFile, Form, File, HTTPException
from typing import Optional
from dotenv import load_dotenv
from fastapi.concurrency import run_in_threadpool
from langchain_groq import ChatGroq
from pydantic import BaseModel
import random

app = FastAPI()
router = APIRouter(tags=["text_generator"])

class Output_Response(BaseModel):
    questions: list

# Prompts
prompt = """You are a smart viva assistant. 
From the content provided, generate 15–20 questions that test understanding, application, and reasoning. 
Classify them by difficulty: easy, medium, hard. 
Return only a structured list of questions as strings. 
Do not include any additional text or instructions.
"""

load_dotenv()
llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0.7)

# Synchronous LLM call
def generate_question(text: Optional[str]):
    try:
        if text:
            full_prompt = f"{prompt}\n\nContent:\n{text}"
        else:
            full_prompt = "Give the user a hint about the topic if no content is provided. Provide output as a list of questions."

        response = llm.with_structured_output(Output_Response).invoke(full_prompt)
        questions = response.questions if response.questions else []
        return questions
    except Exception as e:
        # Prevent the endpoint from crashing
        print("LLM Exception:", e)
        return []

@router.post("/get_answer")
async def get_answer(
    text: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None)
):
    try:
        if file:
            content_bytes = await file.read()
            content = content_bytes.decode("utf-8")
        else:
            content = text

        # Run the blocking LLM in a threadpool
        questions = await run_in_threadpool(generate_question, content)

        if not questions:
            return {"error": "No questions could be generated."}

        top_question = random.choice(questions)
        return {"selected_question": top_question}

    except Exception as e:
        # Return a proper HTTP error instead of crashing
        raise HTTPException(status_code=500, detail=str(e))

app.include_router(router)
