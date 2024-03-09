from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader # will load text from document so no need python `with open , doc.read()`
from langchain_community.vectorstores.pgvector import PGVector
from langchain.vectorstores.pgvector import DistanceStrategy
# from langchain_openai import OpenAIEmbeddings # if needed to mimmic openai
from langchain_community.embeddings import OllamaEmbeddings
import os
from dotenv import load_dotenv
from langchain_community.chat_models import ChatOllama
from langchain.chains import RetrievalQA
from IPython.display import Markdown, display

### SET UP ENV VARS, OPENAIKEY
load_dotenv()
# Openai key
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')

### LANGCHAIN EMBEDDING AND RETRIVAL PART

# VAR; get doc/text and split in chunks cardano_meme_coin.txt, best_meme_coins_2024.txt, history_of_coins.txt
list_documents_txt = ["cardano_meme_coin.txt", "best_meme_coins_2024.txt", "history_of_coins.txt", "article.txt"]

# Use ollama to create embeddings
embeddings = OllamaEmbeddings(temperature=0)

# define connection to pgvector database
CONNECTION_STRING = PGVector.connection_string_from_db_params(
     driver=os.getenv("DRIVER"),
     host=os.getenv("HOST"),
     port=int(os.getenv("PORT")),
     database=os.getenv("DATABASE"),
     user=os.getenv("USER"),
     password=os.getenv("PASSWORD"),
)
# define collection name
COLLECTION_NAME = "test_embedding"


# HELPER functions , create collection, retrieve from collection, chunk documents
def chunk_doc(path: str, files: list) -> list:
  list_docs = []
  for file in files:
    loader = TextLoader(f"{path}/{file}")
    documents = loader.load()
    # using CharaterTextSplitter (use separator split text)
    # text_splitter = CharacterTextSplitter(separator="\n\n", chunk_size=200, chunk_overlap=20)
    # using RecursiveCharacterTextSplitter (maybe better)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=230, chunk_overlap=20)
    docs = text_splitter.split_documents(documents)
    list_docs.append(docs)
    print(f"Doc: {docs}\nLenght list_docs: {len(list_docs)}")
  return list_docs

# using PGVector
def vector_db_create(doc, collection, connection):
  db_create = PGVector.from_documents(
    embedding=embeddings,
    documents=doc,
    collection_name=collection, # must be unique
    connection_string=connection,
    distance_strategy = DistanceStrategy.COSINE,
  )
  return db_create

# PGVector retriever
def vector_db_retrieve(collection, connection, embedding):
  db_retrieve = PGVector(
    collection_name=collection,
    connection_string=connection,
    embedding_function=embedding,
  )
  return db_retrieve

# PGVector adding/updating doc and retriever . sotre parameter is a function "vector_db_retrieve" therefore create a variable with function and use it as store parameter
def add_document_and_retrieve(content, store):
  store.add_documents([Document(page_content=f"{content}")])
  docs_with_score = db.similarity_search_with_score("{content}")
  return doc_with_score

# PGVector update collection
def vector_db_override(doc, embedding, collection, connection):
  changed_db = PGVector.from_documents(
    documents=doc,
    embedding=embedding,
    collection_name=collection,
    connection_string=connection,
    distance_strategy = DistanceStrategy.COSINE,
    pre_delete_collection=True,
  )
  return changed_db


### USE OF EMBEDDING HELPER FOR BUSINESS LOGIC
## Creation of the collection
all_docs = chunk_doc("/home/creditizens/voice_llm", ["article.txt"]) # list_documents_txt
def create_embedding_collection(all_docs: list) -> str:
  collection_name = COLLECTION_NAME
  connection_string = CONNECTION_STRING
  count = 0
  for doc in all_docs:
    print(f"Doc number: {count} with lenght: {len(doc)}")
    vector_db_create(doc, collection_name, connection_string) # this to create/ maybe override also
    # vector_db_override(doc, embeddings, collection_name, connection_string) # to override
    count += 1
  return f"Collection created with documents: {count}"
# print(create_embedding_collection(all_docs))

##  similarity query
question = "What is the story of Azabujuuban ?"

def similarity_search(question):
  db = vector_db_retrieve(COLLECTION_NAME, CONNECTION_STRING, embeddings)
  docs_and_similarity_score = db.similarity_search_with_score(question)
  for doc, score in docs_and_similarity_score:
    print("-" * 80)
    print("Score: ", score)
    print(doc.page_content)
    print("-" * 80)
# print(similarity_search("What are the 9 Rules Rooms?"))

## MMR (Maximal Marginal Relevance) query
def MMR_search(question):
  db = vector_db_retrieve(COLLECTION_NAME, CONNECTION_STRING, embeddings)
  docs_and_MMR_score = db.max_marginal_relevance_search_with_score(question)
  for doc, score in docs_and_MMR_score:
    print("-" * 80)
    print("Score: ", score)
    print(doc.page_content)
    print("-" * 80)
# print(MMR_search("What are the Rules Rooms?"))

## OR use ollama query embedding
text = "How many needs are they in Chikara houses?"
def ollama_embedding(text):
  query_result = embeddings.embed_query(text)
  return query_result

def answer_retriever(query, collection, connection, embedding):
  db = vector_db_retrieve(collection, connection, embedding)
  llm = ChatOllama(model="mistral:7b")
  retriever = db.as_retriever(
    search_kwargs={"k": 3} # 3 best responses
  )
  retrieve_answer = RetrievalQA.from_chain_type(
    llm=llm, 
    chain_type="stuff", 
    retriever=retriever,
    verbose=True,
  )
  query = f"{query}"
  response = retrieve_answer.invoke(query)
  # return display(Markdown(response))
  print("RETRIEVAL RESPONSE Query: ", response["query"])
  print("RETRIEVAL RESPONSE Result: ", response["result"])
  return {
    "Question": query,
    "Response": response["result"],
  }


