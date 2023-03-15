from pathlib import Path
from llama_index import download_loader, GPTSimpleVectorIndex, LLMPredictor, QuestionAnswerPrompt
from langchain import OpenAI
import os
import streamlit as st
import tempfile
from dotenv import load_dotenv
load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

def process_pdf(file):
   with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(file.getbuffer())
        temp_file.flush()

        # Process the PDF file
        with open(temp_file.name, "rb") as pdf_file:
            pdf_reader = PyPDF2.PdfFileReader(pdf_file)
            num_pages = pdf_reader.numPages
            st.write(f"Number of pages in the PDF: {num_pages}")

        # Remove the temporary file
        os.unlink(temp_file.name)


def main(query):
  PDFReader = download_loader("PDFReader")
  loader = PDFReader()
  documents = loader.load_data(file=Path('./holyverse-deck.pdf'))
  QA_PROMPT_TMPL = (
        "We have provided context information below. \n"
        "---------------------\n"
        "{context_str}"
        "\n---------------------\n"
        "Given this information, please answer the question in bullet point format with a new line after each point and cross reference any data cited in the document.\n"
        "warn the user if any information seems off: {query_str}\n"
    )
  QA_PROMPT = QuestionAnswerPrompt(QA_PROMPT_TMPL)

  llm_predictor = LLMPredictor(llm=OpenAI(temperature=0, model_name="text-davinci-003", openai_api_key=openai_api_key))
  index = GPTSimpleVectorIndex(documents)
  response = index.query(query, text_qa_template=QA_PROMPT)
  print(response)
  return response


st.header("Pitch Deck Qs")
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

user_input = st.text_input("Ask a question about the decks")
if uploaded_file is not None:
  pass
if st.button("Find out"):

      st.markdown(main(query=user_input))