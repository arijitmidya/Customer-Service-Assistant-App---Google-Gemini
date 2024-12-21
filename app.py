# Import Streamlit
import streamlit as st

# Standard library imports
import os
# import io
import getpass
import json

# Third-party library imports
import PyPDF2
import requests
import numpy as np
import ipywidgets as widgets
from IPython.display import display, Markdown

# Google generative AI imports
import google.generativeai as genai  

# LangChain imports
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from langchain_community.document_loaders import TextLoader
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from dotenv import load_dotenv

# Chromadb imports
import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings

load_dotenv()


# Method to generate chunks
def get_text_chunks_langchain(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=1)
    chunks = text_splitter.split_text(text)
    docs = [x for x in chunks]
    return docs


# Method to process the PDF
def process_pdf(file_path):
  # PDF processing
  with open(file_path, 'rb') as file:
      pdf_reader = PyPDF2.PdfReader(file)
      pdf_pages = pdf_reader.pages

      # Create chunks for each shoe detail page
      shoe_details = []
      for page_num, page in enumerate(pdf_pages, start=1):

          page_text = page.extract_text()
          # Split the text into lines and remove any empty lines
          lines = [line.strip() for line in page_text.splitlines() if line.strip()]

          # Join lines into continuous text
          consolidated_text = ' '.join(lines)
          shoe_details.append(consolidated_text)

          # PDF reading is done

        # Generate chunks for each shoe detail
      shoe_chunks = []
      for shoe_detail in shoe_details:
          chunks = get_text_chunks_langchain(shoe_detail)
          shoe_chunks.append(chunks)

      return shoe_chunks


# Method to generate embeddings only
def generate_embeddings(textt):

    embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    text_embeddings = [embedding_model.embed_query(text) for text in textt]
    return text_embeddings


# Text retrieval through RAG
def generation(retriever, input_query):
  llm_text = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest")
  template = """
  ```
  {context}
  ```

  {information}


  First greet!
  Display similar shoes from the provided "{context}" only strictly meeting the "{information}" criteria; say a message that you found and provide the following information in a bullet format about the shoe using the {context}: Shoes name, Brand name, Style, Style code, Original retail price, Store Location and description.
  If you don’t find relevant shoes from the provided "{context}", simply say that we have no such shoes. Apologies.
  """
  prompt = ChatPromptTemplate.from_template(template)

  rag_chain = (
      {"context": RunnablePassthrough(), "information": RunnablePassthrough()}
      | prompt
      | llm_text
      | StrOutputParser()
  )
  # Passing text as input data

  result = rag_chain.invoke({"context": retriever, "information": input_query})
  return result




st.title("Customer Service Assistant")  # Updated title
st.write("") 
st.write("") 
st.subheader("How can I help you today?")
st.write("") 


# Retrieve API key from environment variable
google_api_key = os.environ["GOOGLE_API_KEY"]

# Check if the API key is available
if google_api_key is None:
    st.warning("API key not found. Please set the google_api_key environment variable.")
    st.stop()



# Fetch the pdf
pdf_file_path = 'ShoesStore.pdf'
shoe_chunks=process_pdf(pdf_file_path)
# Text embedding
text_embeddings=[]
for shoe in shoe_chunks:
  embeddings = generate_embeddings(shoe)
  text_embeddings.append(embeddings)


# Create the chromadb client
client = chromadb.Client()
# Create db collection
collection_name = "products_embeddings_collection"
client.get_or_create_collection(
      name=collection_name,
      metadata={"hnsw:space": "cosine"}
  )

# Unique IDs for chromadb
ids = list(map(str, range(1, 21)))

# Extract embeddings list from the list of lists
embeddings_list=[]
for emb in text_embeddings:
  embeddings_list.append(emb[0])


# Extract document list from the list of lists
doc_list=[]
for shoe in shoe_chunks:
    doc_list.append(shoe[0])


# Store the documents and text embeddings
product_embeddings_collection  = client.get_collection(name=collection_name)
product_embeddings_collection.add(
    documents=doc_list,
    embeddings=embeddings_list,
    ids=ids
  )


# Get user's question
user_question = st.text_input("Ask a Question:")

if st.button("Get Answer"):
  if user_question:
    text_chunks = get_text_chunks_langchain(user_question)
    input_embeddings = generate_embeddings([chunk for chunk in text_chunks])
            
    # Retrieve relevant documents
    results = product_embeddings_collection.query(
              input_embeddings,
              n_results=2
              )

    result =generation(results, user_question )
    # Display the answer
    st.subheader("Answer:")
    st.write(result)

  else:
    st.warning("Please enter a question.")



# Sidebar
st.sidebar.title("Customer Support")
st.write("") 
st.write("") 
st.sidebar.markdown("Use this assistant to get help with:")
st.sidebar.markdown("- Shoe recommendations")
st.sidebar.markdown("- General inquiries")


# Footer
st.write("") 
st.write("") 
st.markdown("---")
st.markdown("### About Us")
st.markdown("We offer a wide range of shoes for all occasions. Feel free to browse our collection and reach out if you have any questions!")
st.markdown("Contact us at: [support@shoestore.com](mailto:support@shoestore.com)")