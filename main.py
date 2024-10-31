from fastapi import FastAPI, UploadFile, File, Request, Form, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy import create_engine, Column, Integer, Text
from sqlalchemy.ext.declarative import declarative_base
from databases import Database
from typing import List
import uvicorn
import tempfile
import os
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from PyPDF2 import PdfReader
import nltk

nltk.download('punkt')
from nltk.tokenize import sent_tokenize

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Database setup
DATABASE_URL = "sqlite:///./documents.db"
database = Database(DATABASE_URL)
Base = declarative_base()


class Document(Base):
    __tablename__ = "documents"
    id = Column(Integer, primary_key=True, index=True)
    content = Column(Text)


# Create database engine and tables
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
Base.metadata.create_all(bind=engine)

# Initialize document embeddings storage
document_embeddings = {}


@app.on_event("startup")
async def startup():
    await database.connect()
    logger.info("Database connected successfully")


@app.on_event("shutdown")
async def shutdown():
    await database.disconnect()
    logger.info("Database disconnected")


def extract_text_from_pdf(file_path):
    try:
        reader = PdfReader(file_path)
        text = []
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text.append(page_text)
        if not text:
            raise ValueError("No text could be extracted from the PDF")
        return "\n".join(text)
    except Exception as e:
        logger.error(f"Error processing PDF: {str(e)}")
        raise HTTPException(status_code=400,
                            detail=f"Error processing PDF: {str(e)}")


def create_text_chunks(text, max_chunk_size=500):
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = ''
    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= max_chunk_size:
            current_chunk += ' ' + sentence
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks


def compute_embeddings(text_chunks: List[str]):
    try:
        if not text_chunks:
            raise ValueError(
                "No text chunks provided for embedding computation")
        vectorizer = TfidfVectorizer()
        embeddings = vectorizer.fit_transform(text_chunks)
        return embeddings, vectorizer
    except Exception as e:
        logger.error(f"Error computing embeddings: {str(e)}")
        raise HTTPException(status_code=500,
                            detail=f"Error computing embeddings: {str(e)}")


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/upload_pdf/")
async def upload_pdf(request: Request, file: UploadFile = File(...)):
    try:
        # Validate file type
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(status_code=400,
                                detail="Only PDF files are allowed")

        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            content = await file.read()
            if not content:
                raise HTTPException(status_code=400,
                                    detail="Empty file uploaded")
            tmp.write(content)
            tmp_path = tmp.name

        logger.info(f"Processing PDF file: {file.filename}")

        # Extract text
        text = extract_text_from_pdf(tmp_path)

        # Clean up temporary file
        os.unlink(tmp_path)

        if not text.strip():
            raise HTTPException(
                status_code=400,
                detail="No text could be extracted from the PDF")

        # Create text chunks using NLTK
        text_chunks = create_text_chunks(text)

        logger.info(f"Created {len(text_chunks)} text chunks")

        # Compute embeddings and store vectorizer
        embeddings, doc_vectorizer = compute_embeddings(text_chunks)

        # Save to database
        async with database.transaction():
            query = Document.__table__.insert().values(content=text)
            document_id = await database.execute(query)
            document_embeddings[document_id] = {
                'text_chunks': text_chunks,
                'embeddings': embeddings,
                'vectorizer': doc_vectorizer
            }

        logger.info(f"Document saved with ID: {document_id}")

        response = RedirectResponse(url="/", status_code=303)
        response.set_cookie(key="document_id", value=str(document_id))
        return response

    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error uploading PDF: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ask_question/")
async def ask_question(request: Request,
                       document_id: int = Form(...),
                       question: str = Form(...)):
    try:
        response = await process_question(document_id, question)
        logger.info(f"Processed question for document {document_id}")
        return templates.TemplateResponse(
            "index.html", {
                "request": request,
                "result": response,
                "document_id": document_id,
                "question": question
            })
    except Exception as e:
        logger.error(f"Error processing question: {str(e)}")
        return templates.TemplateResponse("index.html", {
            "request": request,
            "error": str(e)
        })


async def process_question(document_id: int, question: str):
    if not question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    if document_id not in document_embeddings:
        raise HTTPException(
            status_code=404,
            detail="Document not found. Please upload a PDF first.")

    try:
        text_chunks = document_embeddings[document_id]['text_chunks']
        embeddings = document_embeddings[document_id]['embeddings']
        doc_vectorizer = document_embeddings[document_id]['vectorizer']

        question_embedding = doc_vectorizer.transform([question])
        similarities = cosine_similarity(question_embedding, embeddings)[0]

        # Get top 3 most similar chunks
        top_indices = similarities.argsort()[-3:][::-1]

        # Combine the most relevant chunks
        response = "\n\n".join([text_chunks[i] for i in top_indices])

        return response

    except Exception as e:
        logger.error(f"Error in question processing: {str(e)}")
        raise HTTPException(status_code=500,
                            detail=f"Error processing question: {str(e)}")


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
