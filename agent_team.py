#### SETUP LLM ####
import streamlit as st
import os
#os.environ["OPENAI_API_KEY"] = "Your Key"

from dotenv import load_dotenv


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
import app
# get the retrieved answers from database to get agent working and choose from app.py if you want to basic, similarity or mmr search
if app.retrieve_answer:
  retrieved_answer = app.retrieve_answer
# get the topic from the user entered question in the application
if app.topic:
  topic = app.topic
# set variables for files and folders where agents will store those and read those
output_file_online_search = "online_search.txt"
output_file_database_search = "database_search.txt"
output_file_insight_advice_on_searches = "insight_advice_on_searches"
output_file_final_report = "final_report"
search_report_dir = "agent_search_reports"
advice_from_search_report_dir = "agent_advice_from_search_reports"
final_report_dir = "agent_final_report"


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
  func=MMRRetriever.run,
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
       
#### AGENTS DEFINITION ####

from crewai import Agent

# Topic for the crew run
# topic = st.text_area("Enter question:", "What are the main findings in this paper?")

# agent 1:
online_searcher = Agent(
  role='',
  goal=f' {topic}',
  verbose=True,
  memory=True,
  backstory=""" """,
  tools=[search_tool],
  allow_delegation=True,
  llm=ollama_llm,
  max_rpm=2,
  max_iter=3,
)

# Agent 2
database_retriever = Agent(
  role='',
  goal=f'{topic} ',
  verbose=True,
  memory=True,
  backstory=""" """,
  tools=[database_retriever_tool],
  allow_delegation=True,
  llm=ollama_llm,
  max_rpm=2,
  max_iter=3,
)

# Agent 3 add here that agent can read from a folder where the other agents are going to produce their file reports so that it will have access to those files. I will create a function with open.... so that this agent will have a tool to read files.
critical_results = Agent(
  role='',
  goal=f' {topic}',
  verbose=True,
  memory=True,
  backstory=""" """,
  tools=[file_reader_tool],
  allow_delegation=False,
  llm=ollama_llm,
  max_rpm=2,
  max_iter=3,
)

# Agent 4
report_creator = Agent(
  role='',
  goal=f' {topic}',
  verbose=True,
  memory=True,
  backstory=""" """,
  tools=[],
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
  description=f"""{topic}""",
  expected_output='',
  agent=,
  async_execution=False,
  output_file=''  # Example of output customization
)

# Task 2:
database_embeddings_retrieval = Task(
  description=f""" """,
  expected_output=f'{topic}',
  agent=,
  async_execution=False,
  output_file=''  # Example of output customization
)

# Task 3:
judge_data = Task(
  description=f""" topic}.""",
  expected_output=f'',
  agent=,
  async_execution=False,
  output_file=''  # Example of output customization
)

# Task 4:
produce_report = Task(
  description=f""" topic}.""",
  expected_output=f'',
  agent=,
  async_execution=False,
  output_file=''  # Example of output customization
)


#### COMBINE THE AGENT AND SET WORKFLOW ####

from crewai import Crew, Process
from langchain_openai import ChatOpenAI # but will be using our environment variables set to Groq's API

project_agents = Crew(
  tasks=[],  # Tasks to be delegated and executed under the manager's supervision. they use ollama (mistral:7b)
  agents=[],
  manager_llm=ChatOpenAI(temperature=0.1, model="mixtral-8x7b-32768", max_tokens=1024),  # Defines the manager's decision-making engine, here it is openai but use the custom llm you want . here uses Groq (mixtral-8x7b-32768)
  tools=[], # not sure if manager can use tools ... have to test
  process=Process.hierarchical,  # Specifies the hierarchical management approach
  verbose=2, # like in the documentation
)


### START THE TEAM WORK ####
result = project_agents.kickoff()
print(result)







