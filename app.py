#for env access and app
import streamlit as st
import os
# import random
# import uuid
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
#from langchain_community.document_loaders import PyPDFLoader
from io import StringIO
import json
import time
from PyPDF2 import PdfReader

# for agents teams
from langchain_community.llms import Ollama
from langchain_community.tools import DuckDuckGoSearchRun
from crewai_tools import BaseTool
from langchain.agents import Tool
from crewai_tools import DirectoryReadTool, FileReadTool
from crewai import Agent, Task, Crew, Process

# for agents conversation logs from terminal forwarded to webui
import sys
from contextlib import contextmanager
# from io import StringIO # already imported in pdf file handling

load_dotenv()

#################
#### VARS #######
#################
# env vars
OPENAI_API_BASE=os.getenv("OPENAI_API_BASE")
OPENAI_MODEL_NAME=os.getenv("OPENAI_MODEL_NAME")
OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")

# vector database var
connection_string = pgvector_langchain.CONNECTION_STRING
# collection_name --> we will use file name in function to define it
embeddings = pgvector_langchain.embeddings

#for llms
#openai_api_key = st.sidebar.text_input('API_Key', type='password', key="api_key") # have different streamlit widget keys to not get an error when the same widget is reused. otherwise we get error from streamlit
#openai_llm = OpenAI(temperature=0.7, openai_api_key=openai_api_key)
#custom_llm = ChatOpenAI(temperature=0.1, model=OPENAI_MODEL_NAME, max_tokens=1024)
groq_llm = ChatGroq(temperature=0.1, groq_api_key=OPENAI_API_KEY, model_name=OPENAI_MODEL_NAME) # mixtral-8x7b groq llm

### agent team vars
#retrieved_answer_from_app = False
#topic_exist = False

########################################
####### HELPER FUNCTIONS ########
########################################

### For embeddings
def generate_response(input_text, llm=groq_llm):
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
      text_splitter = RecursiveCharacterTextSplitter(chunk_size=230, chunk_overlap=10) # instance create to chunk doc or text
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

### write terminal output to webui
@contextmanager
def capture_output():
    new_stdout, new_stderr = StringIO(), StringIO()
    old_stdout, old_stderr = sys.stdout, sys.stderr
    try:
        sys.stdout, sys.stderr = new_stdout, new_stderr
        yield new_stdout, new_stderr
    finally:
        sys.stdout, sys.stderr = old_stdout, old_stderr

# download final report
def download_report(final_report_dir, output_file_final_report):
  st.info("Report ready in the 'final_report' folder and you can download it now")
  try:
    with open(f"{final_report_dir}/{output_file_final_report}") as final_report:
      st.download_button("Download Report Now", final_report, f"{output_file_final_report}")
      st.write("Hope That This Report Will Help You During The Meeting!")
  except Exception as e:
    return f"No report file as been found at {final_report_dir}/{output_file_final_report}, error: {e}"


########################################
###### AGENT TEAM SETUP #######
########################################

### VARS

# LLMs
# ollama mistral7b
ollama_llm = Ollama(model="mistral:7b")
# LMSTUDIO
#OPENAI_API_BASE="http://localhost:1235/v1"
#OPENAI_MODEL_NAME=NA # no need model to be loaded in the webui
#OPENAI_API_KEY=NA # no need freee frreeeeeee freeeeeeeeeeeeeeee
lmstudio = ChatOpenAI(
    model_name="no-need",
    openai_api_key="no-need",
    openai_api_base="http://localhost:1235/v1",
)

# set variables for files and folders where agents will store those and read those
output_file_online_search = "online_search.txt"
output_file_database_search = "database_search.txt"
output_file_insight_advice_on_searches = "insight_advice_on_searches.txt"
output_file_final_report = "final_report.txt"
search_report_dir = "/home/creditizens/pdf_llm_app/agent_search_reports"
database_search_report_dir="/home/creditizens/pdf_llm_app/agent_database_search_reports"
advice_from_search_report_dir = "/home/creditizens/pdf_llm_app/agent_advice_from_search_reports"
final_report_dir = "/home/creditizens/pdf_llm_app/agent_final_report"


### SETUP TOOLS FOR AGENTS TO USE

# database retriever tools which need to match the retriever activated in main app, so here uncomment the right one and add it to agent tool
class BasicRetriever(BaseTool):
    name: str = "Basic Retriever"
    description: str = "This tools is going in the database to retrieve the embedded closest answers vectors. It will return a string that can be red and analyzed in order to provide an answers to the topic: {topic}"

    def _run(self, retrieved_answer: str) -> str:
      # Ollama retriever using 'RetrievalQA.from_chain_type': tried this retriever but it is not efficient at all. returns type str
      self.retrieved_answer = retrieved_answer
      return self.retrieved_answer

class SimilarityRetriever(BaseTool):
    name: str = "Similarity Retriever"
    description: str = "This tools is going in the database to retrieve the embedded closest answers vectors. It will return a string that can be red and analyzed in order to provide an answers to the topic: {topic}"
    def _run(self) -> str:
      # Similarity search retrieval. returns type AiMessage
      retrieve_answer = retrieved_answer
      score_list = []
      try:
        for elem in retrieve_answer:
          score_list.append(elem["score"])
        # get smallest score so shortest vector so closest relationship
        response = ''.join([elem["content"] for elem in retrieve_answer if elem["score"] == min(score_list)])
        return response
      except Exception as e:
        print("Error: ", e)
        response = retrieve_answer
        return f"we got an error: {e} and the response from database was {response}."

class MMRRetriever(BaseTool):
    name: str = "MMR Retriever"
    description: str = "This tools is going in the database to retrieve the embedded closest answers vectors. It will return a string that can be red and analyzed in order to provide an answers to the topic: {topic}"
    def _run(self) -> str:
      # MMR search retrieval. returns type AiMessage
      retrieve_answer = retrieved_answer
      score_list = []
      try:
        for elem in retrieve_answer:
          score_list.append(elem["score"])
        # get smallest score so shortest vector so closest relationship
        response = ''.join([elem["content"] for elem in retrieve_answer if elem["score"] == min(score_list)])
        return response
      except Exception as e:
        print("Error: ", e)
        response = retrieve_answer
        return f"we got an error: {e} and the response from database was {response}."

# instantiation of the database retriever tool
# here choose the tool that you want to run using the 'func' field equal to '<your_class>.run'
# we will defile it in the class instead as we need to pass in the retrieved_answer variable which here will throw an error variable not defined
#database_retriever_tool = Tool(
  #name="database embedding information retriver",
  #func=BasicRetriever(retrieved_answer).run, # change the retriever Here using the class you want (Basic, Similarity, MMR..)
  #description="Useful for database embedding search info. Get answer insight from database embeddings stored information and get best answer from database",
#)

# file reader tool
file_tool_read_search_report = FileReadTool(file_path=f"{search_report_dir}/{output_file_online_search}")
file_tool_read_database_report = FileReadTool(file_path=f"{database_search_report_dir}/{output_file_database_search}")
file_tool_read_advice_report = FileReadTool(file_path=f"{advice_from_search_report_dir}/{output_file_insight_advice_on_searches}")

#class FileReader(BaseTool):
    #name: str = "File Reader"
    #description: str = "This tools is going read all files present in the folder one by one, analyse their content and provide advice to enrich the future report about that topic: {self.topic}. It takes as argument the name of the file present in the folder so the path folder name and file name. Try file name only if you have already access to the folder. If there is no file ask the agent responsable of producing the file to make it so that you can read it."
    #def _run(self, text_file: str) -> str:
      #with open(text_file, "r") as f:
        #file_content = f.read()
        #return file_content

#file_reader_tool = Tool(
  #name="File Reader",
  #func=FileReader.run,
  #description="Read the content of a file to provide you information to analyse from and be able to provide insights and advice, specially around the topic: {self.topic}",
#)

# Directory and files tools
search_report_dir_docs_tool = DirectoryReadTool(directory=f"{search_report_dir}")
database_search_report_dir_docs_tool = DirectoryReadTool(directory=f"{database_search_report_dir}")
advice_from_search_report_dir_docs_tool = DirectoryReadTool(directory=f"{advice_from_search_report_dir}")
# final_report_dir_docs_tool = DirectoryReadTool(directory=f"{final_report_dir}")


#internet search tool
search_tool = DuckDuckGoSearchRun()
       
### AGENTS DEFINITION
# Topic for agent coming from user will be passed as argument in business logic function process

class AgentTeam:

  def __init__(self, topic, retrieved_answer):
    self.output_file_online_search = "online_search.txt"
    self.output_file_database_search = "database_search.txt"
    self.output_file_insight_advice_on_searches = "insight_advice_on_searches.txt"
    self.output_file_final_report = "final_report.txt"
    self.search_report_dir = "./agent_search_reports"
    self.database_search_report_dir="./agent_database_search_reports"
    self.advice_from_search_report_dir = "./agent_advice_from_search_reports"
    self.final_report_dir = "./agent_final_report"
    self.topic = topic
    self.retrieved_answer = retrieved_answer

  # agent 1:
  def OnlineSearcher(self):
    self.online_searcher = Agent(
      role="Online Searcher",
      goal=f"Go online using the search_tool in order to gather as much information as you can about external sources to have more insight about the '{self.topic}'. Find information about '{self.topic}'. Visit different aritcles, search papers, publications, wikipedia also if you want to design a well rounded view on the '{self.topic}'. You will produce a report using markdown.",
      verbose=True,
      memory=True,
      backstory="""You are a specialist of online search and can find information and gather it in a well organized report format using markdown. You are trying to find different view in order to have a well rounded view on the matter you are searching for.""",
      tools=[search_tool],
      allow_delegation=True,
      #llm=ollama_llm,
      #llm=lmstudio,
      llm=groq_llm,
      max_rpm=5,
      max_iter=3,
    )
    return self.online_searcher

  # Agent 2
  def DatabaseRetriever(self):
    self.database_retriever = Agent(
      role="Database Retriever",
      goal=f"You will analyze the information retrieved from the database about the '{self.topic}' and analyze the topic and produce a report for the user to have insights about the '{self.topic}' and be ready for its next meeting with some information about the '{self.topic}'. Here is the the information retrieved from the database about '{self.topic}': '{self.retrieved_answer}'",
      verbose=True,
      memory=True,
      backstory="""You are an experienced report writer for people who look for information from database and get insights about a subject to be ready for the next meeting in which they will need to have something to say about it and be pertinent on the subject. You are helping people to find the right information from databases.""",
      #tools=[self.database_retriever_tool],
      allow_delegation=True,
      #llm=ollama_llm,
      #llm=lmstudio,
      llm=groq_llm,
      max_rpm=5,
      max_iter=3,
    )
    return self.database_retriever

  # Agent 3 
  def CriticalResults(self):
    self.critical_results = Agent(
      role="Search Report Judge and Enhancer",
      goal=f"The user have required more information about '{self.topic}'. You will read the present reports produced by 'Database Retriever' collegue and 'Online Searcher' collegue. You have a 'file_tool_read_search_report' to read the reports from the directory 'search_report_dir_docs_tool' from 'Online Searcher' work. After having analyzed the content of those reports and the user needs: '{self.topic}', you will produce a more elaborated report with a very attractive and true title, paragraphs with titles to explain the different view points, a paragraph called 'Answer to user' in which there will be your critical thinking about what is the answer of user request, 3 Q&A about the '{self.topic}'. You can get more information if you find that the reports are missing some points by using the 'search_tool' to enrich your answer with examples. Use markdown to write the report.",
      verbose=True,
      memory=True,
      backstory="""You are an experienced crititcal reviewer about numerous subjects in which you find the gapes and enrich the reports made by others by providing another view on the subject treated. You use also if needed external sources from the internet to prove your point by providing links as referral to what other specialists say about the subject. You reports are very professional and help people to have a full insight on a subject with interesting pertinent view.""",
      tools=[file_tool_read_search_report, file_tool_read_database_report, search_tool],
      allow_delegation=False,
      #llm=ollama_llm,
      #llm=lmstudio,
      llm=groq_llm,
      max_rpm=5,
      max_iter=3,
    )
    return self.critical_results

  # Agent 4
  def ReportCreator(self):
    self.report_creator = Agent(
      role="Report Creator",
      goal=f"The user need insights about '{self.topic}'. You will create a report that the user will be able to use for the next meeting with the board management. Some other collegues made some research online and in the internal organization database about the '{self.topic}' and have produced a full report that you can read from 'Search Report Judge and Enhancer' collegue using the tool 'file_tool_read_advice_report'. You will read the present reports produced by 'Database Retriever' collegue and 'Online Searcher' collegue. You have a 'file_tool_read_search_report' to read the reports from 'Online Searcher' work. And, you have a 'file_tool_read_database_report' to read the reports from 'Database Retriever' work. After having analyzed the content of those reports and the user needs: The file written by your collegue 'Search Report Judge and Enhancer' is to be found in this folder '{self.advice_from_search_report_dir}' with the name of file '{self.output_file_insight_advice_on_searches}', and, the 'file_tool_read_advice_report' can help you to read it. Read all those files and your job is to create a report for the user to be able to talk about the subject during the board management meeting. It is important to have a report writtem in a professional way and using markdown to have all organized with titles, bullet points, links if available, pertinent to know information, important points to consider. Your produced report should be store in '{self.final_report_dir}' with the name '{self.output_file_final_report}'.",
      verbose=True,
      memory=True,
      backstory="""You are an experience report producer who have worked in the top 10 US companies and your reports have helped board maangers to make good decisions. You are helping people to succeed during their meetings by having very professional reports that provide pertinent insights on subjects. You helping in the decison making process and are well known for being the best in that task getting ceveral Nobel Prices.""",
      tools=[file_tool_read_search_report, file_tool_read_database_report, file_tool_read_advice_report],
      allow_delegation=False,
      #llm=ollama_llm,
      #llm=lmstudio,
      llm=groq_llm,
      max_rpm=5,
      max_iter=3,
    )
    return self.report_creator

# custom values can be added to agents: allow_delegation (ability to delegate tasks or ask questions), max_rpm (rate limiting request to llm), verbose (True to have logs of agent interactions), max_iter (maximum iterations to avoid loops and have best answer after the number of defined iterations of the task). eg:
#max_rpm=10, # Optinal: Limit requests to 10 per minute, preventing API abuse
#max_iter=5, # Optional: Limit task iterations to 5 before the agent tried to gives its best answer


### DEFINE TASKS

class TaskGroups:

  def __init__(self, topic, retrieved_answer):
    self.output_file_online_search = "online_search.txt"
    self.output_file_database_search = "database_search.txt"
    self.output_file_insight_advice_on_searches = "insight_advice_on_searches.txt"
    self.output_file_final_report = "final_report.txt"
    self.search_report_dir = "./agent_search_reports"
    self.database_search_report_dir="./agent_database_search_reports"
    self.advice_from_search_report_dir = "./agent_advice_from_search_reports"
    self.final_report_dir = "./agent_final_report"
    self.topic = topic
    self.retrieved_answer = retrieved_answer

  # Task 1:
  def OnlineSearchTask(self):
    self.online_search = Task(
      description=f"""Searching online about the user requested information gathering request about: '{self.topic}'. It can be used as well to search more information about reports that you have red and to enrich your report with external internet links and point of views. """,
      expected_output="A markdown report on your answer about the user request after having made some search online abou it. the report need to be rich with a professional tone. The report purpose is to have insights for a board manager meeting, therefore, do not invent any fake information, it has to be grounded with referrals of where you foudn the information. Make sure that the report is having titles, paragraphs, links and answer to the question about user topic ('{self.topic}'). Output a report using markdown at '{self.search_report_dir}/{self.output_file_online_search}'",
      agent=AgentTeam(topic=self.topic, retrieved_answer=self.retrieved_answer).OnlineSearcher(),
      async_execution=False,
      output_file=f"{self.search_report_dir}/{self.output_file_online_search}"
    )
    return self.online_search

  # Task 2:
  def DatabaseEmbeddingsRetrievalTaks(self):
    self.database_embeddings_retrieval = Task(
      description=f"""Retrieving from the database information by running a tool that will provide information about the best vector form the embedded database data. The tool is already set with user topic '{self.topic}' so will retrieve the information and return an answer that can be used.""",
      expected_output=f"After retrieving information returned by the database tool about the topc '{self.topic}', you will produce a report with insights answering to user question '{self.topic}'. The report is for the presentation before the board managers, therefore, need to be written with a professional tone and be very organized. The report shoudl have titles, parapgraphs and the answer to the user request ('{self.topic}'). The report shoudl highlight what was the returned answer fromt he database. If the database doesn't retrieve  any information because the database doesn't have the information or because of an error, mention it in the report and make it very short jsut to tell what is the error or that the database retrieved answer is too far from the subject meaning that the database doesn't have any answer about it. output a file report using markdown at '{self.database_search_report_dir}/{self.output_file_database_search}'",
      agent=AgentTeam(topic=self.topic, retrieved_answer=self.retrieved_answer).DatabaseRetriever(),
      async_execution=False,
      output_file=f"{self.database_search_report_dir}/{self.output_file_database_search}" 
    )
    return self.database_embeddings_retrieval

  # Task 3:
  def JudgeDataTask(self):
    self.judge_data = Task(
      description=f"""Judging the reports made by collegues 'Online Searcher' and 'Database Retriever' about user topic: '{self.topic}'. Providing critical thinking about those reports findings and making search online to validate or enrich those answers in order to have a well rounded report answering the topic ('{self.topic}').""",
      expected_output=f"You will produce a more elaborated report with a very attractive and true title, paragraphs with titles to explain the different view points, a paragraph called 'Answer to user' in which there will be your critical thinking about what is the answer of user request, 3 Q&A about the topic: '{self.topic}'. The report is for the next meeting with the borad management to make decision, make sure it is pertinent and detailed enough with proof like links or other points of view. Output the report in file using markdown '{self.advice_from_search_report_dir}/{self.output_file_insight_advice_on_searches}'",
      agent=AgentTeam(topic=self.topic, retrieved_answer=self.retrieved_answer).CriticalResults(),
      async_execution=False,
      output_file=f"{self.advice_from_search_report_dir}/{self.output_file_insight_advice_on_searches}"
    )
    return self.judge_data

  # Task 4:
  def ProduceReportTask(self):
    self.produce_report = Task(
      description=f"""Produces a final report based on the report made by your collegues and gathered and formatted by 'Search Report Judge and Enhancer' about the user topic: '{self.topic}'.""",
      expected_output=f"Produce a very detailed and professional report using markdown at '{self.final_report_dir}/{self.output_file_final_report}'. The report is for the board management to have pertinent insight on thesubject so it should answer in a professional and realistic way. It should be SMART. The report shoudl have, title and subtitles, paragraphs, 5 Q&A, answer to topic: '{self.topic}', pertinent view section to provide a new view on the topic, warning section to have like a contengency plan for the topic and anytthing else that you judge that  should be added to the report in order to be ready for a meeting. As you use markdown, you can add links, bullet points to make it nice to read as well.",
      agent=AgentTeam(topic=self.topic, retrieved_answer=self.retrieved_answer).ReportCreator(),
      async_execution=False,
      output_file=f"{self.final_report_dir}/{self.output_file_final_report}" 
    )
    return self.produce_report


### COMBINE THE AGENT AND SET WORKFLOW

class ProjectAgents:

  def __init__(self, topic, online_search, database_embeddings_retrieval, judge_data, produce_report, online_searcher, database_retriever, critical_results, report_creator):
    # topic
    self.topic = topic
    # tasks
    self.online_search = online_search
    self.database_embeddings_retrieval = database_embeddings_retrieval
    self.judge_data = judge_data
    self.produce_report = produce_report
    # agents
    self.online_searcher = online_searcher
    self.database_retriever = database_retriever
    self.critical_results = critical_results
    self.report_creator = report_creator

  def AgentsCrew(self):
    self.project_agents = Crew(
      tasks=[self.online_search, self.database_embeddings_retrieval, self.judge_data, self.produce_report],  # Tasks to be delegated and executed under the manager's supervision. they use ollama (mistral:7b)
      agents=[self.online_searcher, self.database_retriever, self.critical_results, self.report_creator],
      manager_llm=ChatOpenAI(temperature=0.1, model="mixtral-8x7b-32768", max_tokens=1024),  # Defines the manager's decision-making engine, here it is openai but use the custom llm you want . here uses Groq (mixtral-8x7b-32768)
      # tools=[], # not sure if manager can use tools ... have to test
      process=Process.hierarchical,  # Specifies the hierarchical management approach
      verbose=2, # like in the documentation
    )
    return self.project_agents


### START THE TEAM WORK ####
# result = project_agents.kickoff()
# print(result)
# ProjectAgents(topic=f"{topic}").AgentsCrew()

### TEST PRINT STDOUT OR STDERR TO THE WEBUI SO GET TERMINAL OUTPUT FORWARDED TO WEBUI

#with st.form("test_form"):
  #text = st.text_area("Enter text:", "What are the main findings in this paper?", key='test')
  #submition = st.form_submit_button("Read Stdout")
  #if submition:
    #import sys
    #from contextlib import contextmanager
    #from io import StringIO

    #@contextmanager
    #def capture_output():
        #new_stdout, new_stderr = StringIO(), StringIO()
        #old_stdout, old_stderr = sys.stdout, sys.stderr
        #try:
            #sys.stdout, sys.stderr = new_stdout, new_stderr
            #yield sys.stdout
        #finally:
            #sys.stdout, sys.stderr = old_stdout, old_stderr

    # Example usage
    #with capture_output() as captured:
        #print("This is the text output from terminal: ", text)

        #st.write(captured.getvalue())



############################################################################
###### APP WITH BUSINESS LOGIC STARTS ######
############################################################################

st.title('🦜🔗 Creditizens PDF App (LangChain Groq Ollama/LMStudio PGVector)')

### UPLOAD DOCUMENT ANd EMBEDDING
# User uploads document and embedded before querying
def upload_and_chunk():
  doc_uploaded = st.file_uploader("Upload your file", type=[".pdf",".txt"], accept_multiple_files=False, key="upload_one_file")
  status_and_file_name = {"upload_status": "upload done", "file": ""}
  if doc_uploaded is not None:
    file_name = doc_uploaded.name
    status_and_file_name["file"] = file_name
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
                 ### Report Generation
                 #### Easy!
                 * Agents will search for it online
                 * Agents will retrieve the most relevant data to answer your query
                 * Agents will then work on data in order to produce insightful report for the meeting
                 """
                 )

  return status_and_file_name

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
def ask_question_get_agent_team_make_report(file_name):
  with st.form('my_form'):
    ### USER PROMPT PART
    # User entering text
    text = st.text_area("Enter text:", "What are the main findings in this paper?", key='user_question')
    submitted = st.form_submit_button("Submit")

    # IF LOGIC FOR GROQ TO HAVE THE API_KEY ENTERED .startswith("gsk_"), CAN CHANGE CODE FOR OTHER LLM
    # message warning if openai_api_key hasn't been provider by User
    #if not openai_api_key.startswith("gsk_"): # 'gsk-' for groq and 'sk-' for openai
      #st.warning("Please enter your OpenAI API key!", icon="⚠")
      #junko = "SIBUYAAAAA"
      #st.info(junko, icon="🔥") # success, warning, info, exception(only in try except and no 'icon')
      #st.balloons()
      #st.snow()

    #### LLM ANSWER PART
    # LLM answer to User prompt
    if submitted: # and openai_api_key.startswith("gsk_"):
      #message = HumanMessage(content=f"{text}")
      #print("######################  TYPE OF 'message': ", type(message))
      # this to query groq for chat
      # with st.spinner("Wait while file is being embedded..."):
          # generate_response(text) # can add 'llm' if need to change it, default is groq_llm
      # this to query database for embedding retrieval
      with st.spinner("Wait while embedding are being analyzed..."):
        # print("FILE NAME: ", file_name)
          
        # Ollama retriever using 'RetrievalQA.from_chain_type': tried this retriever but it is not efficient at all. returns type str
        retrieve_answer = answer_retriever(text, file_name, connection_string, embeddings)["Response"] # can add an extra argument for 'llm' used if wan't to change, default is ChatOllama(model="mistral:7b")
          
        # Similarity search retrieval. returns type AiMessage
        # retrieve_answer = similarity_search(text, file_name, connection_string, embeddings)
          
        # MMR search retrieval. returns type AiMessage
        # retrieve_answer = MMR_search(text, file_name, connection_string, embeddings)
        
        # this is only for similarity search and mmr as it returns a list of dictionaries
        #score_list = []
        #try:
          #for elem in retrieve_answer:
            #score_list.append(elem["score"])
          # get smallest score so shortest vector so closest relationship
          #response = ''.join([elem["content"] for elem in retrieve_answer if elem["score"] == min(score_list)])
        #except Exception as e:
          #print("Error: ", e)
          #response = retrieve_answer
          #print("Type response (retrieve_answer): ", type(response))

          # get llm check the answer and provide more insight
          #llm_insight_on_response = dict(groq_llm.invoke(response))["content"]
          #st.info(llm_insight_on_response)

        # topic taken from question
        topic = text
        # Agent work result
        st.header('Agents are working for you', divider='green')
        st.subheader('See under agents workflow and actions while waiting for the report.')
          
        # set to true the variables to retrieved_answer_from_app and topic_exist
        #retrieved_answer_from_app = True
        #topic_exist = True
        # print("is answer retrieved?: ", retrieved_answer_from_app, "Does topic exist?: ", topic_exist)
        # tasks 
        online_search = TaskGroups(topic, retrieve_answer).OnlineSearchTask()
        database_embeddings_retrieval = TaskGroups(topic, retrieve_answer).DatabaseEmbeddingsRetrievalTaks()
        judge_data = TaskGroups(topic, retrieve_answer).JudgeDataTask()
        produce_report = TaskGroups(topic, retrieve_answer).ProduceReportTask()
        # agents
        online_searcher = AgentTeam(topic, retrieve_answer).OnlineSearcher()
        database_retriever = AgentTeam(topic, retrieve_answer).DatabaseRetriever()
        critical_results = AgentTeam(topic, retrieve_answer).CriticalResults()
        report_creator = AgentTeam(topic, retrieve_answer).ReportCreator()
          
        project_agent_team_work = ProjectAgents(topic, online_search, database_embeddings_retrieval, judge_data, produce_report, online_searcher, database_retriever, critical_results, report_creator).AgentsCrew()

        # get agents conversation logs output in webui
        #with capture_output() as captured:
          #project_agent_team_work.kickoff()
        #st.write(captured.getvalue())
        # Using capture_output in Streamlit
        with st.spinner("Go have a coffee (OUTSIDE lol ) while agent team are working on producing the best report ever for your meeting..."):
          with capture_output() as (out, err):      
            result = project_agent_team_work.kickoff()
            # Retrieve and display the captured stdout and stderr
            stdout = out.getvalue()
            stderr = err.getvalue()
          if stdout:
            st.text(f"Working hard for you....Please be patient\n{stdout}")
          if stderr:
            st.text(f"Errors/Warnings:\n{stderr}")
  return f""


#### 
def business_logic():
  start_upload = upload_and_chunk()
  if start_upload["upload_status"] == "upload done":
    filename = start_upload["file"]
    print("filename: ", filename)
    report = ask_question_get_agent_team_make_report(filename)
  
  file_exist = os.path.exists(f"{final_report_dir}/{output_file_final_report}")
  if file_exist == True: 
    # download report
    download_report(final_report_dir, output_file_final_report)

business_logic()


