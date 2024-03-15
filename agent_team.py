#### SETUP LLM ####
import streamlit as st
import os
#os.environ["OPENAI_API_KEY"] = "Your Key"

from dotenv import load_dotenv
from app import topic_exist

load_dotenv()

#### SETUP LLMS ####
# OLLAMA OR GROQ (uncomment in .env file)
# important!!! : for GROQ the field "tools" is not supported so we need to use tools of llms with another llm setup openai or ollama
OPENAI_API_BASE=os.getenv("OPENAI_API_BASE")
OPENAI_MODEL_NAME=os.getenv("OPENAI_MODEL_NAME")  # Adjust based on available model
OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")

# Or just OLLAMA using langchain if using GROQ for the crew manager as GROQ doesn't support the field "tools" of the agents
from langchain_community.llms import Ollama

ollama_llm = Ollama(model="mistral:7b")

# LMSTUDIO
#OPENAI_API_BASE="http://localhost:1235/v1"
#OPENAI_MODEL_NAME=NA # no need model to be loaded in the webui
#OPENAI_API_KEY=NA # no need freee frreeeeeee freeeeeeeeeeeeeeee

# so need to create a client for ollama or lmstudio as it mimics openai endpoint logic already
# but if you want you can create client as crewai is langchain compatible using langchian imports of those like : from langchain_community.llms import Ollama, from langchain_community.llms import LlamaCpp ...


### OPTION TO HAVE HUMAN INTERACTION IN THE CREW PROCESS ####
#from langchain.agents import load_tools
#human_tools = load_tools(["human"])
# then when creating an agent you pass in the human tool to have human interact at that level. eg:
#digital_marketer = Agent(
  #role='...',
  #goal='...',
  #backstory="""...""",
  #verbose=..., # True or False
  #allow_delegation=...,
  #tools=[search_tool]+human_tools # Passing human tools to the agent
  #max_rpm=..., # int
  #max_iter=..., # int
  #llm=...
#)

### VARS
#from app import retrieved_answer_from_app, topic_exist

# get the retrieved answers from database to get agent working and choose from app.py if you want to basic, similarity or mmr search
#def run_retriever():
  #if retrieved_answer_from_app == True:
    #from app import retrieve_answer
    #return retrieve_answer
#retrieve_answer = run_retriever()

# get the topic from the user entered question in the application
#def get_topic():
  #if topic_exist == True:
    #from app import topic
    #return topic
#topic = get_topic()

# set variables for files and folders where agents will store those and read those
output_file_online_search = "online_search.txt"
output_file_database_search = "database_search.txt"
output_file_insight_advice_on_searches = "insight_advice_on_searches.txt"
output_file_final_report = "final_report.txt"
search_report_dir = "./agent_search_reports"
database_search_report_dir="./agent_database_search_reports"
advice_from_search_report_dir = "./agent_advice_from_search_reports"
final_report_dir = "./agent_final_report"


#### SETUP TOOLS FOR AGENTS TO USE ####
from langchain_community.tools import DuckDuckGoSearchRun

search_tool = DuckDuckGoSearchRun()

import pgvector_langchain
from crewai_tools import BaseTool

# database retriever tools which need to match the retriever activated in main app, so here uncomment the right one and add it to agent tool
class BasicRetriever(BaseTool):
    name: str = "Basic Retriever"
    description: str = "This tools is going in the database to retrieve the embedded closest answers vectors. It will return a string that can be red and analyzed in order to provide an answers to the topic: {topic}"

    def _run(self) -> str:
      # Ollama retriever using 'RetrievalQA.from_chain_type': tried this retriever but it is not efficient at all. returns type str
      retrieve_answer = retrieved_answer # can add an extra argument for 'llm' used if wan't to change, default is ChatOllama(model="mistral:7b")
      return retrieve_answer

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
from langchain.agents import Tool

# here choose the tool that you want to run using the 'func' field equal to '<your_class>.run'
database_retriever_tool = Tool(
  name="database embedding information retriver",
  func=BasicRetriever.run,
  description="Useful for database embedding search info. Get answer insight from database embeddings stored information and get best answer from database",
)

# file reader tool
class FileReader(BaseTool):
    name: str = "File Reader"
    description: str = "This tools is going read all files present in the folder one by one, analyse their content and provide advice to enrich the future report about that topic: {topic}"
    def _run(self, text_file: str) -> str:
      with open(text_file, "r") as f:
        file_content = f.read()
        return file_content

file_reader_tool = Tool(
  name="File Reader",
  func=FileReader.run,
  description="Read the content of a file to provide you information to analyse from and be able to provide insights and advice, specially around the topic: {topic}",
)

from crewai_tools import DirectoryReadTool, FileReadTool

search_report_dir_docs_tool = DirectoryReadTool(directory=f"{search_report_dir}")
database_search_report_dir_docs_tool = DirectoryReadTool(directory=f"{database_search_report_dir}")
advice_from_search_report_dir_docs_tool = DirectoryReadTool(directory=f"{advice_from_search_report_dir}")
final_report_dir_docs_tool = DirectoryReadTool(directory=f"{final_report_dir}")
file_tool = FileReadTool()
       
#### AGENTS DEFINITION ####

from crewai import Agent

# Topic for the crew run
# topic = st.text_area("Enter question:", "What are the main findings in this paper?")

# agent 1:
online_searcher = Agent(
  role="Online Searcher",
  goal=f"Go online using the search_tool in order to gather as much information as you can about external sources to have more insight about the '{topic}'. Find information about '{topic}'. Visit different aritcles, search papers, publications, wikipedia also if you want to design a well rounded view on the '{topic}'. You will produce a report using markdown.",
  verbose=True,
  memory=True,
  backstory="""You are a specialist of online search and can find information and gather it in a well organized report format using markdown. You are trying to find different view in order to have a well rounded view on the matter you are searching for.""",
  tools=[search_tool],
  allow_delegation=True,
  llm=ollama_llm,
  max_rpm=2,
  max_iter=3,
)

# Agent 2
database_retriever = Agent(
  role="Database Retriever",
  goal=f"You will find information in the database using the database_retriever_tool about the '{topic}'. After getting the answer from the database, you will analyze it and analyze the topic and produce a report for the user to have insights about the '{topic}' and be ready for its next meeting with some information about the '{topic}'",
  verbose=True,
  memory=True,
  backstory="""You are an experienced report writer for people who look for information from database and get insights about a subject to be ready for the next meeting in which they will need to have something to say about it and be pertinent on the subject. You are helping people to find the right information from databases.""",
  tools=[database_retriever_tool],
  allow_delegation=True,
  llm=ollama_llm,
  max_rpm=2,
  max_iter=3,
)

# Agent 3 add here that agent can read from a folder where the other agents are going to produce their file reports so that it will have access to those files. I will create a function with open.... so that this agent will have a tool to read files.
critical_results = Agent(
  role="Search Report Judge and Enhancer",
  goal=f"The user have required more information about '{topic}'. You will read the present reports produced by 'Database Retriever' collegue and 'Online Searcher' collegue. You have a 'file_reader_tool' and 'file_tool' to read those reports from the directory 'search_report_dir_docs_tool' and 'database_search_report_dir_docs_tool'. Their reports are located respectively in the files '{output_file_online_search}' and  '{output_file_database_search}'. After having analyzed the content of those reports and the user need '{topic}', you will produce a more elaborated report with a very attrctive and true title, paragraphs with titles to explain the different view points, a paragraph called 'Answer to user' in which there will be your critical thinking about what is the answer of user request, 3 Q&A about the '{topic}'. You can get more information if you find that the reports are missing some points by using the 'search_tool' to enrich your answer with examples. Use markdown to write the report.",
  verbose=True,
  memory=True,
  backstory="""You are an experienced crititcal reviewer about numerous subjects in which you find the gapes and enrich the reports made by others by providing another view on the subject treated. You use also if needed external sources from the internet to prove your point by providing links as referral to what other specialists say about the subject. You reports are very professional and help people to have a full insight on a subject with interesting pertinent view.""",
  tools=[file_reader_tool, search_report_dir_docs_tool, database_search_report_dir_docs_tool, file_tool, search_tool],
  allow_delegation=False,
  llm=ollama_llm,
  max_rpm=2,
  max_iter=3,
)

# Agent 4
report_creator = Agent(
  role="Report Creator",
  goal=f"The user need insights about '{topic}'. You will create a report that the user will be able to use for the next meeting with the board management. Some other collegues made some research online and in the internal organization database about the '{topic}' and have produced a full report that you can read from 'Search Report Judge and Enhancer' collegue using the tool 'file_reader_tool' or 'file_tool'. The file written by your collegue 'Search Report Judge and Enhancer' is to be found in this folder '{advice_from_search_report_dir}' with the name of file '{output_file_insight_advice_on_searches}', and, the 'advice_from_search_report_dir_docs_tool' can help you to find it. Read that file and your job is to create a report for the user to be able to talk about the subject during the board management meeting. It is important to have a report writtem in a professional way and using markdown to have all organized with titles, bullet points, links if available, pertinent to know information, important points to consider. Your produced report should be store in '{final_report_dir}' with the name 'output_file_final_report'",
  verbose=True,
  memory=True,
  backstory="""You are an experience report producer who have worked in the top 10 US companies and your reports have helped board maangers to make good decisions. You are helping people to succeed during their meetings by having very professional reports that provide pertinent insights on subjects. You helping in the decison making process and are well known for being the best in that task getting ceveral Nobel Prices.""",
  tools=[file_reader_tool, final_report_dir_docs_tool, advice_from_search_report_dir_docs_tool, file_tool],
  allow_delegation=False,
  llm=ollama_llm,
  max_rpm=2,
  max_iter=3,
)

# custom values can be added to agents: allow_delegation (ability to delegate tasks or ask questions), max_rpm (rate limiting request to llm), verbose (True to have logs of agent interactions), max_iter (maximum iterations to avoid loops and have best answer after the number of defined iterations of the task). eg:
#max_rpm=10, # Optinal: Limit requests to 10 per minute, preventing API abuse
#max_iter=5, # Optional: Limit task iterations to 5 before the agent tried to gives its best answer

#### DEFINE TASKS ####

from crewai import Task


# Task 1:
online_search = Task(
  description=f"""Searching online about the user requested information gathering request about: '{topic}'. It can be used as well to search more information about reports that you have red and to enrich your report with external internet links and point of views. """,
  expected_output="A markdown report on your answer about the user request after having made some search online abou it. the report need to be rich with a professional tone. The report purpose is to have insights for a board manager meeting, therefore, do not invent any fake information, it has to be grounded with referrals of where you foudn the information. Make sure that the report is having titles, paragraphs, links and answer to the question about user topic ('{topic}'). Output a report using markdown at '{search_report_dir}/{output_file_online_search}'",
  agent=online_searcher,
  async_execution=False,
  output_file=f"{search_report_dir}/{output_file_online_search}"
)

# Task 2:
database_embeddings_retrieval = Task(
  description=f"""Retrieving from the database information by running a tool that will provide information about the best vector form the embedded database data. The tool is already set with user topic '{topic}' so will retrieve the information and return an answer that can be used.""",
  expected_output=f"After retrieving information returned by the database tool about the topc '{topic}', you will produce a report with insights answering to user question '{topic}'. The report is for the presentation before the board managers, therefore, need to be written with a professional tone and be very organized. The report shoudl have titles, parapgraphs and the answer to the user request ('{topic}'). The report shoudl highlight what was the returned answer fromt he database. If the database doesn't retrieve  any information because the database doesn't have the information or because of an error, mention it in the report and make it very short jsut to tell what is the error or that the database retrieved answer is too far from the subject meaning that the database doesn't have any answer about it. output a file report using markdown at '{database_search_report_dir}/{output_file_database_search}'",
  agent=database_retriever,
  async_execution=False,
  output_file=f"{database_search_report_dir}/{output_file_database_search}" 
)

# Task 3:
judge_data = Task(
  description=f"""Judging the reports made by collegues 'Online Searcher' and 'Database Retriever' about user topic: '{topic}'. Providing critical thinking about those reports findings and making search online to validate or enrich those answers in order to have a well rounded report answering the topic ('{topic}').""",
  expected_output=f"You will produce a more elaborated report with a very attractive and true title, paragraphs with titles to explain the different view points, a paragraph called 'Answer to user' in which there will be your critical thinking about what is the answer of user request, 3 Q&A about the topic: '{topic}'. The report is for the next meeting with the borad management to make decision, make sure it is pertinent and detailed enough with proof like links or other points of view. Output the report in file using markdown '{advice_from_search_report_dir}/{output_file_insight_advice_on_searches}'",
  agent=critical_results,
  async_execution=False,
  output_file=f"{advice_from_search_report_dir}/{output_file_insight_advice_on_searches}"
)

# Task 4:
produce_report = Task(
  description=f"""Produces a final report based on the report made by your collegues and gathered and formatted by ''Search Report Judge and Enhancer' about the user topic: '{topic}'.""",
  expected_output=f"Produce a very detailed and professional report using markdown at '{final_report_dir}/{output_file_final_report}'. The report is for the board management to have pertinent insight on thesubject so it should answer in a professional and realistic way. It should be SMART. The report shoudl have, title and subtitles, paragraphs, 5 Q&A, answer to topic: '{topic}', pertinent view section to provide a new view on the topic, warning section to have like a contengency plan for the topic and anytthing else that you judge that  should be added to the report in order to be ready for a meeting. As you use markdown, you can add links, bullet points to make it nice to read as well.",
  agent=report_creator,
  async_execution=False,
  output_file=f"{final_report_dir}/{output_file_final_report}" 
)


#### COMBINE THE AGENT AND SET WORKFLOW ####

from crewai import Crew, Process
from langchain_openai import ChatOpenAI # but will be using our environment variables set to Groq's API

project_agents = Crew(
  tasks=[online_search, database_embeddings_retrieval, judge_data, produce_report],  # Tasks to be delegated and executed under the manager's supervision. they use ollama (mistral:7b)
  agents=[online_searcher, database_retriever, critical_results, report_creator],
  manager_llm=ChatOpenAI(temperature=0.1, model="mixtral-8x7b-32768", max_tokens=1024),  # Defines the manager's decision-making engine, here it is openai but use the custom llm you want . here uses Groq (mixtral-8x7b-32768)
  # tools=[], # not sure if manager can use tools ... have to test
  process=Process.hierarchical,  # Specifies the hierarchical management approach
  verbose=2, # like in the documentation
)


### START THE TEAM WORK ####
# result = project_agents.kickoff()
# print(result)







