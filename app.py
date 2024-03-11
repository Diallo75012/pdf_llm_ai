#for env access and app
import streamlit as st
import os
from dotenv import load_dotenv
# langchain
from langchain_openai import OpenAI
# from langchain_community.llms import OpenAI # will be deprecated
from langchain_groq import ChatGroq # need to be installed not in community library as of 03/2024
from langchain_openai import ChatOpenAI # if needed to mimmic Openai endpoint
# from langchain.chat_models import HumanMessage
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader # will load text from document so no need python `with open , doc.read()`
from langchain_community.vectorstores.pgvector import PGVector
from langchain.vectorstores.pgvector import DistanceStrategy
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain.chains import RetrievalQA
from IPython.display import Markdown, display # we will just use st.markdown probably
# langchain custom library
import pgvector_langchain
from pgvector_langchain import (
  # chunk_doc, # we don't our library function but a cutomized one that is in the helper functions section 
  vector_db_create,
  vector_db_retrieve,
  add_document_and_retrieve,
  vector_db_override,
  # create_embedding_collection, # will create a custom one here
  similarity_search,
  MMR_search,
  ollama_embedding,
  answer_retriever
)
# pdf and txt file handling
from langchain_community.document_loaders import PyPDFLoader
from io import StringIO
import json
import time
# Agent team import
import agent_team


load_dotenv()

# env vars
OPENAI_API_BASE=os.getenv("OPENAI_API_BASE")
OPENAI_MODEL_NAME=os.getenv("OPENAI_MODEL_NAME")
OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")

connection_string = pgvector_langchain.CONNECTION_STRING
# collection_name --> we will use file name in function to define it

# for pgvector module
embeddings = pgvector_langchain.embeddings

#for llms
openai_api_key = st.sidebar.text_input('OpenAI API Key', type='password')
openai_llm = OpenAI(temperature=0.7, openai_api_key=openai_api_key)
custom_llm = ChatOpenAI(temperature=0.1, model=OPENAI_MODEL_NAME, max_tokens=1024)
groq_llm = ChatGroq(temperature=0.1, groq_api_key=OPENAI_API_KEY, model_name=OPENAI_MODEL_NAME) # mixtral-8x7b groq llm

# from agent_team
topic = agent_team.topic
result = agent_team.result

### HELPER FUNCTIONS

def generate_response(input_text, llm=ChatGroq(temperature=0.1, groq_api_key=OPENAI_API_KEY, model_name=OPENAI_MODEL_NAME)):
    st.info(llm.invoke(input_text)) # use llm.invoke(input_text) as llm(input_text) like in the documentation won't work

def store_embeddings(input_text, embedding_llm_engine):
    print("######################  TYPE OF 'input_text' from function: ", type(input_text))
    llm = embedding_llm_engine
    st.info() # use llm.invoke(input_text) as llm(input_text) like in the documentation won't work

def chunk_doc_text(text_of_doc: str, name_of_doc: str) -> list:
  if text_of_doc != "" and text_of_doc is not None:
    list_docs = []
    try:
      # using CharaterTextSplitter (use separator split text)
      # text_splitter = CharacterTextSplitter(separator="\n\n", chunk_size=200, chunk_overlap=20)
      # using RecursiveCharacterTextSplitter (maybe better)
      text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20) # instance create to chunk doc or text
      document = [Document(page_content=text_of_doc, metadata={"source": f"{name_of_doc}"})] # List is needed here for it to be a LangChain document as the '.Document'
      # print("DOCUMENT: ", document)
      # here we use '.split_documents' as we have formatted same as what is returned by 'TextLoader(f"{path}/{file}").load()' but keep in mind that '.split_text' can be used
      docs = text_splitter.split_documents(document)
      # print("DOCS: ", docs)
      print("DOCS TYPE: ", type(docs))
      list_docs.append(docs)
      count = 1
      st.info(f"# Number of chunks: {len(docs)}")
      if len(docs) > 3:
        st.info("### Overview of 2 first chunks parts")
        for doc in docs[0:2]:
          st.info(f"##### Part {count} chunk of document: {doc}\n")
          time.sleep(0.5)
          count += 1
        st.info("##### Please wait while other parts being processed...")
      else:
        for doc in docs:
          st.info(f"Part {count} chunk of document: {doc}\n")
          count += 1
          time.sleep(0.5)
        
    except Exception as e:
      print("Chunk Doc Error: ", e)
      return e
    print("LEN LIST DOCS: ", len(list_docs))
  return list_docs

#all_docs = chunk_doc_text(text_of_doc) # list_documents_txt
def create_embedding_collection(all_docs: list, file_name: str) -> st.delta_generator.DeltaGenerator:
  global connection_string
  collection_name = file_name
  connection_string = connection_string
  count = 0
  for doc in all_docs:
    print(f"Doc number: {count}")
    vector_db_create(doc, collection_name, connection_string) # this to create/ maybe override also
    # vector_db_override(doc, embeddings, collection_name, connection_string) # to override
    count += 1
  print("Type st.info: ", type(st.info("junko")), "\n" ,"Type st.success: ", type(st.success("shibuya")))
  if count <= 1:
    return st.info(f"Collection created with {count} document.")
  else:
    return st.info(f"Collection created with {count} documents.")

### DOC UPLOAD PART

from PyPDF2 import PdfReader
@st.cache_data
def load_docs(file_uploaded, file_name):
    all_text = ""
    print("file name end: ", file_name.split(".")[1])
    if file_name.split(".")[1] == "pdf":
      pdf_reader = PdfReader(file_uploaded)
      text = ""
      for page in pdf_reader.pages:
        text += page.extract_text()
        all_text += text
    elif file_name.split(".")[1] == "txt":
      stringio = StringIO(file_uploaded.getvalue().decode("utf-8"))
      text = stringio.read()
      all_text += text
    else:
      st.warning('Please provide txt or pdf.', icon="⚠️")
    print("All Text: ", all_text)
    return all_text

### APP START ####
st.title('🦜🔗 Quickstart App')

## UPLOAD DOCUMENT ANd EMBEDDING
# User uploads document and embedded before querying
doc_uploaded = st.file_uploader("Upload your file", type=[".pdf",".txt"], accept_multiple_files=False,)

if doc_uploaded is not None:
  file_name = doc_uploaded.name
  with st.spinner("Wait while file is being uploaded..."):
    st.info(f"File uploaded: {file_name}")
    doc = load_docs(doc_uploaded, file_name)

    on = st.toggle("Check Document's head (first 400 characters...)")
    if on:
      #st.write(doc)
      st.subheader(doc[0:400])
    
    tutorial = st.markdown(
                 """
                 __________________________________________
                 ### Tutorial
                 * Use the above toggle button to have a quick preview of the first 400 characters of the document uploaded.
                 * Use the under button to store(embed) your document.
                 * Wait for the embedding process to complete and enter your query to get insight about the document. 
                 """
               )
    
    if doc:
      with st.form("Embbed_doc"):
        tutorial = st.info("Use this button to store document before querying. Make sure you wait the process to be done before starting queries.")
        doc_submitted = st.form_submit_button("Store Document")

        if doc_submitted:
          with st.spinner("Wait while file is being chunked..."):
            all_docs = chunk_doc_text(doc, file_name) # returns list of doc chuncks
          print("TYPE ALL DOCS: ", type(all_docs))
          # print("ALL DOCS: ", [str(doc)+"\n" for doc in all_docs])
          with st.spinner("Wait while file is being embedded..."):
            create_embedding_collection(all_docs, file_name) # make sure embedding server is running if local otherwise provide the engine env vars
            st.success("Document successfully embedded. You can start asking your questions.")
            extra_info = st.markdown(
                 """
                 _______________________________________
                 ### Add API Key Before Query
                 #### Easy!
                 * Put Your Groq API Key on the left Side Panel Field
                 * Then Use the field just under to ask your question
                 * The LLM will retrieve the most relevant data to answer your query
                 """
               )

  # User uploaded document head is shown to have an overview of it (call also use pd dataframe to have more control on how many row to show if csv file for eg.)
  #with st.spinner("Wait while file is being uploaded..."):
    #loader = PyPDFLoader(json.load(doc_uploaded).upload_url)
    #pages = loader.load_and_split()
    #st.write(pages[0])
    # bytes_data = doc_uploaded.getvalue()
    # .subheader or use .write
    # head_doc_uploaded = st.subheader(bytes_data.decode("utf-8")) # .decode("utf-8") for .txt files
    #st.success("Now Your Can Query To Get Insights About The Document")
    #st.warning("Make sure the uploaded document type is supported (ppdf, txt)", icon="🔥")

#### FORM WITH USER PROMPTING LOGIC TO LLM/AGENT
# need to add logic to query embeddings and maybe use agent crew to work
with st.form('my_form'):
    ### USER PROMPT PART
    # User entering text
    text = st.text_area("Enter text:", "What are the main findings in this paper?")
    submitted = st.form_submit_button("Submit")

    # IF LOGIC FOR GROQ TO HAVE THE API_KEY ENTERED .startswith("gsk_"), CAN CHANGE CODE FOR OTHER LLM
    # message warning if openai_api_key hasn't been provider by User
    if not openai_api_key.startswith("gsk_"): # 'gsk-' for groq and 'sk-' for openai
        st.warning("Please enter your OpenAI API key!", icon="⚠")
        junko = "SIBUYAAAAA"
        st.info(junko, icon="🔥") # success, warning, info, exception(only in try except and no 'icon')
        #st.balloons()
        #st.snow()

    #### LLM ANSWER PART
    # LLM answer to User prompt
    if submitted and openai_api_key.startswith("gsk_"):
        #message = HumanMessage(content=f"{text}")
        #print("######################  TYPE OF 'message': ", type(message))
        # this to query groq for chat
        # with st.spinner("Wait while file is being embedded..."):
            # generate_response(text) # can add 'llm' if need to change it, default is groq_llm
        # this to query database for embedding retrieval
        with st.spinner("Wait while embedding are being analyzed..."):
          print("FILE NAME: ", file_name)
          # Ollama retriever using 'RetrievalQA.from_chain_type': tried this retriever but it is not efficient at all. returns type str
          retrieve_answer = answer_retriever(text, file_name, connection_string, embeddings)["Response"] # can add an extra argument for 'llm' used if wan't to change, default is ChatOllama(model="mistral:7b")
          # Similarity search retrieval. returns type AiMessage
          # retrieve_answer = similarity_search(text, file_name, connection_string, embeddings)
          # MMR search retrieval. returns type AiMessage
          # retrieve_answer = MMR_search(text, file_name, connection_string, embeddings)
          
          # this is only for similarity search and mmr as it returns a list of dictionaries
          score_list = []
          try:
            for elem in retrieve_answer:
              score_list.append(elem["score"])
            # get smallest score so shortest vector so closest relationship
            response = ''.join([elem["content"] for elem in retrieve_answer if elem["score"] == min(score_list)])
          except Exception as e:
            print("Error: ", e)
            response = retrieve_answer
            print("Type response (retrieve_answer): ", type(response))

            # get llm check the answer and provide more insight
            llm_insight_on_response = dict(groq_llm.invoke(response))["content"]
            st.info(llm_insight_on_response)

        # Agent work result
        st.info(result)








