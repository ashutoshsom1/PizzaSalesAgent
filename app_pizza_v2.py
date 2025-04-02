import langchain
langchain.verbose = False
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
import time
import random
import matplotlib
# matplotlib.use('TkAgg')
matplotlib.use('agg')
import matplotlib.pyplot as plt
#
import streamlit as st
st.set_option('deprecation.showPyplotGlobalUse', False)
import fetch_bot_answers
import os

import re
#
import speech_recognition as sr
import pyttsx3
#
import pandas as pd
from langchain.chains import ConversationalRetrievalChain
from langchain_community.agent_toolkits import create_sql_agent
from langchain_openai import AzureChatOpenAI
from langchain_openai import ChatOpenAI
from typing import List
from pydantic import BaseModel
import sqlite3
from langchain.tools import Tool
from langchain_experimental.utilities import PythonREPL
from sqlalchemy import create_engine
from langchain.agents.agent_toolkits import create_retriever_tool
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain_core.prompts.chat import MessagesPlaceholder
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent, SQLDatabaseToolkit
from langchain_openai import ChatOpenAI
from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain_core.messages import AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from PIL import Image
from streamlit_searchbox import st_searchbox


load_dotenv(r"ae.env")
local_web_server_uri = "http://localhost:8501"  # Replace with your local web server URI where you host the file pdf to navigate to the source pages


KEY = os.getenv("KEY")
ENDPOINT = os.getenv("ENDPOINT")
API_VERSION = os.getenv("API_VERSION")
LLM_MODEL_NAME= os.getenv("LLM_MODEL_NAME")




st.set_page_config(page_title="Sales Chatbot")

llm = AzureChatOpenAI( 
    openai_api_key=KEY,
    azure_endpoint=ENDPOINT,
    openai_api_version= API_VERSION,
    deployment_name=LLM_MODEL_NAME,
    temperature = 0,
)

#db = SQLDatabase.from_uri(r"sqlite:///C:/PizzaSalesInsight/all_pizza_sales_insightsv6.1.db")

db = SQLDatabase.from_uri(r"sqlite:///all_pizza_sales_insightsv6.1.db")

toolkit = SQLDatabaseToolkit(db=db, llm=llm)
context = toolkit.get_context()
tools = toolkit.get_tools()



embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-base-en-v1.5")
vector_db = FAISS.load_local(r"C:/PizzaSalesInsight/pizza_vector_database_v3", embeddings, allow_dangerous_deserialization = True)
retriever = vector_db.as_retriever(search_kwargs={"k": 2})
description = """Used for quering the documents directly or quering the documents after output from sql"""
retriever_tool = create_retriever_tool(
    retriever,
    name="query_documents",
    description=description,
)

##
# Initialize the Python REPL
python_repl = PythonREPL()

def run_python_code(code):
  """
  Runs the provided Python code with the necessary imports for Streamlit visualization.

  Args:
      code (str): The Python code to be executed.

  Returns:
      The output of the Python code execution, or an error message if there's an issue.
  """

  # Add two newlines and import statements for Streamlit
  code_string = code.replace("plt.show()", "")
  modified_code = f"{code_string}\n\nimport streamlit as st\nst.pyplot()"
# st.altair_chart
# st.pyplot()
  try:
    return python_repl.run(modified_code)
  except Exception as e:
    return f"Error running code:\n{str(e)}"


# Define a function to run Python code using the REPL
# def run_python_code(code):
    
#     return python_repl.run(code)

description__2 = """A Python shell. Use this to execute Python commands. Input should be a valid Python command. If you want to see the output visualized, ensure it results in data suitable for a chart. For example, to visualize sales growth per product category, you can use Streamlit functions like st.bar_chart to create a bar chart directly within this app."""
# description__2 = "A Python shell. Use this to execute Python commands. Input should be a valid Python command. If you want to see the output, ensure it's printed. For example, to visualize data in a chart, you can use the provided code snippet to create a bar chart showing the sales growth for each product category."
# Create a Tool instance
repl_tool = Tool(
    name="python_repl",
    description= description__2,
    func=run_python_code,
)

def get_top_key_information_of_store(data):
    try:
        store_id = data['store_id']
        pattern = r'^(SRT)[-\s](\d{4,5})$'
        match = re.match(pattern, store_id)
        store_id = f"{match.group(1)}-{match.group(2)}"

        conn = sqlite3.connect(r'all_pizza_sales_insightsv6.1.db')
        cursor = conn.cursor()

        #Total revenue in the last quarter
        sql_query = """SELECT 
            round(SUM(revenue)) AS total_revenue
        FROM 
            pizza_data
        WHERE 
            store_id = ? 
            AND quarter = 'Quarter4';"""
        cursor.execute(sql_query, (store_id,))
        total_revenue_last_quarter = cursor.fetchall()
        total_revenue_last_quarter = [item for sublist in total_revenue_last_quarter for item in sublist][0]
        
        #Total Profit in the last quarter
        sql_query = """SELECT 
            round(SUM(profit)) AS total_profit
        FROM 
            pizza_data
        WHERE 
            store_id = ? 
            AND quarter = 'Quarter4';"""
        cursor.execute(sql_query, (store_id,))
        total_profit_last_quarter = cursor.fetchall()
        total_profit_last_quarter = [item for sublist in total_profit_last_quarter for item in sublist][0]

        #Total number of orders in the last quarter
        sql_query = """SELECT 
            count(distinct(order_id)) AS total_orders
        FROM 
            pizza_data
        WHERE 
            store_id = ? 
            AND quarter = 'Quarter4';"""
        cursor.execute(sql_query, (store_id,))
        total_orders_last_quarter = cursor.fetchall()
        total_orders_last_quarter = [item for sublist in total_orders_last_quarter for item in sublist][0]

        #Top 3 profitable pizzas sold in the last quarter
        sql_query = """SELECT 
            name,
            round(SUM(profit)) AS total_profit
        FROM 
            pizza_data
        WHERE store_id = ? AND quarter = 'Quarter4'
        GROUP BY 
            name
        ORDER BY 
            total_profit DESC
        LIMIT 3;"""
        cursor.execute(sql_query, (store_id,))
        top_3_profitable_pizzas = cursor.fetchall()
        top_3_profitable_pizzas1 = [item for sublist in top_3_profitable_pizzas for item in sublist][0]
        top_3_profitable_pizzas2 = [item for sublist in top_3_profitable_pizzas for item in sublist][2]
        top_3_profitable_pizzas3 = [item for sublist in top_3_profitable_pizzas for item in sublist][4]
    
        #top_3_profitable_pizzas_sold = top_3_profitable_pizzas1 + " " + top_3_profitable_pizzas2  + " " + top_3_profitable_pizzas3
        top_3_profitable_pizzas_sold = [top_3_profitable_pizzas1,top_3_profitable_pizzas2,top_3_profitable_pizzas3]
        
        #Store ranking in terms of profit in the last quarter
        sql_query = """WITH
        store_profits AS (
            SELECT
                store_id,
                SUM(profit) AS total_profit
            FROM
                pizza_data
            WHERE
                quarter = 'Quarter4'
            GROUP BY
                store_id
        ), ranked_stores AS (
            SELECT
                store_id,
                total_profit,
                RANK() OVER (ORDER BY total_profit DESC) AS rank
            FROM
                store_profits
        )SELECT
            rank
        FROM
            ranked_stores
        WHERE
            store_id = ?;"""
        cursor.execute(sql_query, (store_id,))
        store_ranking_profitable_last_quarter = cursor.fetchall()
        store_ranking_profitable_last_quarter = [item for sublist in store_ranking_profitable_last_quarter for item in sublist][0]
        
       #Peak Selling Day
        sql_query = """SELECT 
            day AS order_date,
            COUNT(distinct(order_id)) AS total_orders
        FROM pizza_data
        WHERE store_id = ?
        GROUP BY day
        ORDER BY total_orders DESC
        LIMIT 1;
        """
        cursor.execute(sql_query, (store_id,))
        peak_selling_day = cursor.fetchall()
        peak_selling_day = [item for sublist in peak_selling_day for item in sublist][0]

        conn.close()

        data_req = {'Store ID': store_id ,\
                    'Total Orders Served in the last quarter': total_orders_last_quarter,\
                    'Peak Selling Day' : peak_selling_day,\
                    'Total Revenue in the last quarter' : total_revenue_last_quarter,\
                    'Total Profit in the last quarter' : total_profit_last_quarter,\
                    'Top 3 Most Profitable Pizzas in the last quarter': top_3_profitable_pizzas_sold,\
                    'Last Quarter Profitability Rank out of Total 30 Stores':store_ranking_profitable_last_quarter
                    }
        return data_req
    except:
        "Store ID not found!"

description = '''Given the store_id, this function will find out the store_id's top key information, form your input query in the format {"store_id":store_id}'''

from langchain.tools import BaseTool, StructuredTool, tool
store_top_data_search = StructuredTool.from_function(
    func=get_top_key_information_of_store,
    name="Get_Top_Key_Information_For_Store",
    description=description,
)

def get_pizza_recommendation_for_store(data):
    try:
        store_id = data['store_id']
        pattern = r'^(SRT)[-\s](\d{4,5})$'
        match = re.match(pattern, store_id)
        store_id = f"{match.group(1)}-{match.group(2)}"

        conn = sqlite3.connect(r'all_pizza_sales_insightsv6.1.db')
        cursor = conn.cursor()

        sql_query = """SELECT 
                store_format, 
                region
            FROM pizza_data
            WHERE store_id = ?;
            """
        
        cursor.execute(sql_query, (store_id,))
        results = cursor.fetchall()
        store_format = [item for sublist in results for item in sublist][0]
        store_region = [item for sublist in results for item in sublist][1]

        sql_query ="""SELECT 
        name, 
        SUM(quantity) AS total_sales_quantity
        FROM pizza_data
        WHERE store_id = ?
        GROUP BY name
        ORDER BY total_sales_quantity DESC
        LIMIT 1;"""
        cursor.execute(sql_query, (store_id,))
        results = cursor.fetchall()
        highest_selling_pizza = [item for sublist in results for item in sublist][0]

        #Select a region randomly which is not same as given store region
        sql_query = """SELECT region
        FROM pizza_data
        WHERE region != ?
        GROUP BY region
        ORDER BY RANDOM()
        LIMIT 1;
        """
        cursor.execute(sql_query, (store_region,))
        other_selected_region = cursor.fetchall()
        other_selected_region = [item for sublist in other_selected_region for item in sublist][0]
        

        sql_query = """WITH StoreInfo AS (
            SELECT 
                store_format, 
                region
            FROM pizza_data
            WHERE store_id = ?
        ),
        RegionalSales AS (
            SELECT 
                name, 
                region, 
                SUM(quantity) AS total_sales
            FROM pizza_data
            WHERE store_format = (SELECT store_format FROM StoreInfo)
            AND region = ?
            GROUP BY name, region
        )
        SELECT 
            name,
            SUM(total_sales) AS total_sales
        FROM RegionalSales
        GROUP BY name
        ORDER BY total_sales DESC
        LIMIT 2;
        """
        cursor.execute(sql_query, (store_id,other_selected_region))
        pizza_recommendations = cursor.fetchall()
        pizza_recommendation1 = [item for sublist in pizza_recommendations for item in sublist][0]
        pizza_recommendation2 = [item for sublist in pizza_recommendations for item in sublist][2]
        #pizza_recommendation3 = [item for sublist in pizza_recommendations for item in sublist][4]
        
        all_pizza_recommendations = [highest_selling_pizza,pizza_recommendation1,pizza_recommendation2]
        
        conn.close()

        data_req = {'Store ID': store_id ,\
                    #'Store Forrmat': store_format,\
                    #'Region': store_region,\
                    'Total Orders Served in the last quarter': all_pizza_recommendations,\
                    'Extra Information': 'Based on top selling pizza of this store and similar stores these are the recommendations' }
        return data_req
    except:
        "Store ID not found!"


description = '''Given the store_id, this function will find out the pizza recommedations \
    for the given store_id based on top selling pizza of different regions but same store format,\
        form your input query in the format {"store_id":store_id}. Don't add any other information in your output.'''

from langchain.tools import BaseTool, StructuredTool, tool
store_pizza_recommendations_search = StructuredTool.from_function(
    func=get_pizza_recommendation_for_store,
    name="Get_Pizza_Recommendation_For_Store",
    description=description,
)

def send_mail_tool(data):
    try:
        emailSubject = data.get('subject')
        emailBody = data.get('body')
        emailRecipient = data.get('recipient')
     
        api_url_token = fetch_bot_answers.api_url_token
        bot_secret_token = os.getenv("BOT_SECRET_TOKEN")
        base_url = fetch_bot_answers.base_url

        conversationId,new_bearer_token,stream_url = fetch_bot_answers.get_stream_url_and_token(api_url_token, bot_secret_token)
        api_url_conversation = '/'.join([base_url, 'conversations', conversationId, 'activities'])
        
        print(stream_url)

        message_text1 = "send mail"
        response_data1 = fetch_bot_answers.send_message(api_url_conversation, new_bearer_token, conversationId, message_text1)
        time.sleep(1)
        message_text2 = emailSubject
        response_data2 = fetch_bot_answers.send_message(api_url_conversation, new_bearer_token, conversationId, message_text2)
        time.sleep(1)
        message_text3 = emailBody
        response_data3 = fetch_bot_answers.send_message(api_url_conversation, new_bearer_token, conversationId, message_text3)
        time.sleep(1)
        message_text4 = emailRecipient
        print(message_text4)
        response_data4 = fetch_bot_answers.send_message(api_url_conversation, new_bearer_token, conversationId, message_text4)
        
        return "Success"
    except:
        "Incorrect Input!"

# def send_mail_tool(data):
#     try:
#         print(f"Data : {data}")
#         return "Sucess"
#     except:
#         "Incorrect Input!"

# description = '''Use this tool when the input is send mail, send email, mail, email'''

description = '''Given the subject,body,recipient name this function will fetch the email using the name \
    and send an email with the given subject and body to the given recipient.\
        Form your input query in the format {"subject":question,"body":answer,"recipient":recipient name}.'''

from langchain.tools import BaseTool, StructuredTool, tool
send_email_tool = StructuredTool.from_function(
    func=send_mail_tool,
    name="send_mail_tool",
    description=description,
)

tools.append(retriever_tool)
tools.append(repl_tool)
tools.append(store_top_data_search)
tools.append(store_pizza_recommendations_search)
tools.append(send_email_tool)

suggested_questions = pd.read_csv(r"sample_suggestion_qstns.csv")["Suggested_Questions"].tolist()

sytem_msg_prompt = open(r"prompt_pizzza_v2.txt").read()

messages = [
        SystemMessagePromptTemplate.from_template(sytem_msg_prompt),
        HumanMessagePromptTemplate.from_template("{input},{chat_history}"),
        AIMessage(content="{SQL_FUNCTIONS_SUFFIX}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
]

prompt = ChatPromptTemplate.from_messages(messages)
prompt = prompt.partial(**context)

agent = create_openai_tools_agent(llm, tools, prompt)

class InMemoryHistory(BaseChatMessageHistory, BaseModel):
    """In memory implementation of chat message history."""

    messages: List[BaseMessage] = Field(default_factory=list)

    def add_messages(self, messages: List[BaseMessage]) -> None:
        """Add a list of messages to the store"""
        self.messages.extend(messages)

    def clear(self) -> None:
        self.messages = []

# global variable to store the chat message history.
store = {}

def get_by_session_id(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryHistory()
    return store[session_id]


agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        return_intermediate_steps=True,
        return_source_documents=True
    )


agentic_executor = RunnableWithMessageHistory(agent_executor, get_by_session_id, output_messages_key="output", input_messages_key="input",
    history_messages_key="chat_history")

config = {'configurable': {'session_id': "AB322"}}
                        

# Streamlit App

custom_css = """

    <style>
    
    
    #custom-title {
        font-size: 36px;
        font-weight: bold;
        text-align: center;
        margin-bottom: 20px;
    }
    .stButton button[kind="secondary"] {
        background-color: #88888A;
        color: white;
        padding: 10px 20px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 25px;
        cursor: pointer;
        border-radius: 8px;
        border: none;
        transition-duration: 0.4s;
        margin-top: 30px;
        position: fixed;
        bottom: 120px;
        left: 400px; 
        # z-index:1000; 
                      
    }
    .stButton button[kind="primary"] {
        background-color: #88888A;
        color: white;
        padding: 10px 20px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 25px;
        cursor: pointer;
        border-radius: 8px;
        border: none;
        transition-duration: 0.4s;
        margin-top: 30px;
        position: fixed;
        bottom: 120px;
        left: 1100px; 
        # z-index:1000; 
                      
    }
    
    .stButton>button:hover {
        background-color: #45a049;
        color: white;
    }
    .stButton>button:active {
        background-color: #45a049;
        box-shadow: 0 5px #666;
        transform: translateY(4px);
    }
    .st-spinner>div>div {
        border-color: #4CAF50;
    }
    

    .golden-text {
        background: linear-gradient(45deg, #FFD700, #FFA500);
        -webkit-background-clip: text;
        color: transparent;
    }

    .green-text {
        color: #00FF00;
    }

    .gray-text {
        color: #808080;
    }
    
    .top-right {
        position: absolute;
        top: 10px;
        right: 10px;
    }
    
    </style>
    """
st.markdown(custom_css, unsafe_allow_html=True)
  
class BadRequestError(Exception):
    pass  


def recognize_speech():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.write("Calibrating for ambient noise...")
        recognizer.adjust_for_ambient_noise(source, duration=2) 
        st.write("Listening...")
        try:
            audio = recognizer.listen(source, timeout=10, phrase_time_limit=30)  # Adjusted phrase_time_limit for longer inputs
            text = recognizer.recognize_google(audio)
            return text
        except sr.UnknownValueError:
            return "Sorry, I couldn't understand what you said."
        except sr.RequestError:
            return "Sorry, I'm unable to access the Speech Recognition service."


# def recognize_from_microphone():

#         print("Speak into your microphone.")
#         st.write("Listening...")

#         audio_bytes = st_audiorec()
                          
        
#         # Set up the Azure Speech SDK
#         speech_config = SpeechConfig(subscription=os.getenv("SPEECH_KEY"), region=os.getenv("SPEECH_REGION"))
#         # audio_config = speechsdk.audio.AudioConfig(filename=audio_file_path)
        
#         # Create a recognizer with the given settings
#         recognizer = SpeechRecognizer(speech_config=speech_config, audio_config=audio_bytes)
    
#         # Perform speech recognition
#         result = recognizer.recognize_once_async().get()
        
#         # Display the result
#         if result.reason == speechsdk.ResultReason.RecognizedSpeech:
#             st.write("Recognized: {}".format(result.text))
#             return result.text
#         elif result.reason == speechsdk.ResultReason.NoMatch:
#             st.write("No speech could be recognized")
#         elif result.reason == speechsdk.ResultReason.Canceled:
#             st.write("Speech Recognition canceled: {}".format(result.cancellation_details.reason))
#             if result.cancellation_details.reason == speechsdk.CancellationReason.Error:
#                 st.write("Error details: {}".format(result.cancellation_details.error_details))

        



# Function to speak the response
def speak_response(response):
    engine = pyttsx3.init()
    engine.say(response)
    engine.runAndWait()

# st.set_page_config(page_title="Sales Chatbot", page_icon=None, layout="wide", initial_sidebar_state="auto")

# remove external links to images
def remove_markdown_images(text):
    return re.sub(markdown_image_regex, "", text)
 
# Regular expression to find Markdown image links
markdown_image_regex = r"!\[.*?\]\(.*?\)"

# # Function to reset or refresh the app state
def refresh_page():
    # Reset or clear the necessary session state or variables
    st.session_state.messages = []
    st.experimental_rerun()

#Main Streamlit app
def main():
    # st.markdown(custom_css, unsafe_allow_html=True)
    
    # st.title("📝 Give you query")
    
    # header = st.container()
    # with header:
    #     col1, col2 = st.columns([5, 10])  # Adjust column widths as needed
        
    with st.sidebar:
        
        
        
           
              
        st.markdown('''
            <style>
            custom-css
            {
                 text-align: center;
                 color: #3366FF;
                 font-size: 24px;
             }
             </style>

            <div id="custom-title"><span class="gray-text">Sales Insight Bot</span></div>

               
            ''', unsafe_allow_html=True)
    

    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # if "suggested_questions" not in st.session_state:
    #     st.session_state.suggested_questions = suggested_questions

    if st.button("Refresh",type="primary"):
        refresh_page()

    # def generate_questions(user_question, context_questions):
    #     """Use an AI model to generate 5 new questions based on user input and context."""
    #     prompt = f"""
    #     Given the user question: "{user_question}"
    #     and the context questions: {context_questions}

    #     Generate 5 related simple questions in a list.
    #     """
    #     # Call OpenAI API (or any other LLM you prefer)
    #     client = llm
    #     response = client.invoke(input=prompt)
    #     print(response)

    #     new_questions = response.content.strip().split("\n")

    #     return new_questions[:5]  # Ensure only 5 questions are returned

    # def update_question_list(user_question):
    #     """Replace 5 random questions in the session history with new AI-generated questions."""
    #     old_questions = st.session_state.suggested_questions
    #     new_questions = generate_questions(user_question, old_questions)

    #     # Select 5 random indices to replace
    #     indices_to_replace = random.sample(range(len(old_questions)), 5)
        
    #     for i, index in enumerate(indices_to_replace):
    #         old_questions[index] = new_questions[i]

    #     st.session_state.suggested_questions = old_questions  # Update session state

    # def search_suggestions(searchterm: str) -> list:
    #     return suggested_questions
    
    # selected_question = st_searchbox(
    #     search_suggestions,
    #     placeholder="Write your queries?...",
    #     key="my_key",
    # )
    #selected_question = st.selectbox("Check Suggestions", [""] + st.session_state.suggested_questions)
    KEYWORDS = ["revenue", "profit", "region", "store_format"]
    def extract_keywords(text):
        return [word for word in KEYWORDS if re.search(rf"\b{word}\b", text, re.IGNORECASE)]

    with st.form("my_form",clear_on_submit=True):
       
        prompt = st.text_input(label = "Write your queries")
        selected_question = st.selectbox(label="Check Suggestions", options=[""] + suggested_questions, label_visibility="collapsed", placeholder="Check Suggestions", key="my_key")
    
        submitted = st.form_submit_button("Submit")
    
    if  submitted:  #selected_question or prompt:
        
        if selected_question:
            qry = selected_question

            st.session_state.messages.append({"role": "user", "content": qry})
            with st.chat_message("user"):
                st.markdown(qry)
            print("chat =", [(message["role"], message["content"]) for message in st.session_state.messages])
            # print("chat_history=", [(message["role"], message["content"]) for message in st.session_state.messages])
            # recent_messages = st.session_state.messages[-2:]
                    
            try:
                #answer = agent_executor.invoke({"input": prompt,"chat_history":[(message["role"], message["content"]) for message in st.session_state.messages[-2:]]})
                answer = agentic_executor.invoke(  
                            {"input": qry},
                            config)
                
                print(answer)
    
                
                #result = answer['output']
                result_ = answer['output']
                result = remove_markdown_images(result_)
                #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
                intermediate_steps = answer["intermediate_steps"]
                tool_usage = []
                for steps in intermediate_steps:
                    tool_usage.append(steps[0].tool)
            
                if 'query_documents' in tool_usage:
                    results = vector_db.similarity_search_with_score(result)
                    list_of_dicts = []
                    for res in results:
                        score = res[1]
                        filename = res[0].metadata['source']
                        filename = filename.split("\\")[-1]
                        page_number = res[0].metadata['page'] + 1
    
                        dictionary = {"source": filename, "page": page_number, "score": score}
                        list_of_dicts.append(dictionary)
    
                    sorted_list_of_dicts = sorted(list_of_dicts, key=lambda x: x["score"])
                    
                    
                else:
                    sorted_list_of_dicts = []
                #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
                print("answer__=",answer)
                # result = sqldb_agent.run(final_prompt.format(prompt=prompt))
                # result = query_engine.query(prompt)   
                # result = agent_executor.invoke(final_prompt.format({"input": prompt}))
                # result = agent_executor.invoke({"input": prompt})
                print("prompt=", qry)
                # print("result=", result)
                print("answer=",result)

                with st.chat_message("assistant"):
                    message_placeholder = st.empty()
                    full_response = result
                    message_placeholder.markdown(full_response + "|")
                message_placeholder.markdown(full_response)
                #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
                if sorted_list_of_dicts:
                    for dictionary in sorted_list_of_dicts:
                        dictionary.pop("score", None)
                    st.markdown("<font size='2'>Sources:</font>", unsafe_allow_html=True)  # Add a smaller text header
                    st.markdown("<ul style='margin-top: 0; padding-left: 20px;'>", unsafe_allow_html=True)
                    i = 0
                    for item in sorted_list_of_dicts:
                        if i == 1:
                            break
                        # Create the source link with the local PDF path
                        source_link = f"<li><a href='{local_web_server_uri}/{item['source']}#page={item['page']}' target='_blank'>{item['source']} - Page {item['page']}</a></li>"
                        # st.markdown(f"<li><font size='2'>{source_link}</font></li>", unsafe_allow_html=True)
                        st.markdown(source_link, unsafe_allow_html=True)
                        i += 1
                    st.markdown("</ul>", unsafe_allow_html=True)
                #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
                print("result=",result)
                st.session_state.messages.append({"role": "assistant", "content": full_response})
        
            except Exception as e:
                print("exception=",e)
                note = "Please Try Again, Reframe the Query or Provide More Context and Try Again"
                with st.chat_message("assistant"):
                    message_placeholder = st.empty()
                    # information_after_llm =str(note)
                    message_placeholder.markdown(note + "|",unsafe_allow_html=True)  # Display the error message
                message_placeholder.markdown(note)
                # print("=",information_after_llm)
                st.session_state.messages.append({"role": "assistant", "content": note})
                

        else:
            if prompt:
                
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)
                print("chat =", [(message["role"], message["content"]) for message in st.session_state.messages])
                # print("chat_history=", [(message["role"], message["content"]) for message in st.session_state.messages])
                # recent_messages = st.session_state.messages[-2:]
                
                        
                try:
                    #answer = agent_executor.invoke({"input": prompt,"chat_history":[(message["role"], message["content"]) for message in st.session_state.messages[-2:]]})
                    answer = agentic_executor.invoke(  
                                {"input": prompt},
                                config)
                    
                    print(answer)

                    
                    #result = answer['output']
                    result_ = answer['output']
                    result = remove_markdown_images(result_)
                    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
                    intermediate_steps = answer["intermediate_steps"]
                    tool_usage = []
                    for steps in intermediate_steps:
                        tool_usage.append(steps[0].tool)
                
                    if 'query_documents' in tool_usage:
                        results = vector_db.similarity_search_with_score(result)
                        list_of_dicts = []
                        for res in results:
                            score = res[1]
                            filename = res[0].metadata['source']
                            filename = filename.split("\\")[-1]
                            page_number = res[0].metadata['page'] + 1

                            dictionary = {"source": filename, "page": page_number, "score": score}
                            list_of_dicts.append(dictionary)

                        sorted_list_of_dicts = sorted(list_of_dicts, key=lambda x: x["score"])
                        
                        
                    else:
                        sorted_list_of_dicts = []
                    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
                    print("answer__=",answer)
                    # result = sqldb_agent.run(final_prompt.format(prompt=prompt))
                    # result = query_engine.query(prompt)   
                    # result = agent_executor.invoke(final_prompt.format({"input": prompt}))
                    # result = agent_executor.invoke({"input": prompt})
                    print("prompt=", prompt)
                    # print("result=", result)
                    print("answer=",result)

                    with st.chat_message("assistant"):
                        message_placeholder = st.empty()
                        full_response = result
                        message_placeholder.markdown(full_response + "|")
                    message_placeholder.markdown(full_response)
                    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
                    if sorted_list_of_dicts:
                        for dictionary in sorted_list_of_dicts:
                            dictionary.pop("score", None)
                        st.markdown("<font size='2'>Sources:</font>", unsafe_allow_html=True)  # Add a smaller text header
                        st.markdown("<ul style='margin-top: 0; padding-left: 20px;'>", unsafe_allow_html=True)
                        i = 0
                        for item in sorted_list_of_dicts:
                            if i == 1:
                                break
                            # Create the source link with the local PDF path
                            source_link = f"<li><a href='{local_web_server_uri}/{item['source']}#page={item['page']}' target='_blank'>{item['source']} - Page {item['page']}</a></li>"
                            # st.markdown(f"<li><font size='2'>{source_link}</font></li>", unsafe_allow_html=True)
                            st.markdown(source_link, unsafe_allow_html=True)
                            i += 1
                        st.markdown("</ul>", unsafe_allow_html=True)
                    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
                    print("result=",result)
                    st.session_state.messages.append({"role": "assistant", "content": full_response})
                
                except Exception as e:
                    print("exception=",e)
                    note = "Please Try Again, Reframe the Query or Provide More Context and Try Again"
                    with st.chat_message("assistant"):
                        message_placeholder = st.empty()
                        # information_after_llm =str(note)
                        message_placeholder.markdown(note + "|",unsafe_allow_html=True)  # Display the error message
                    message_placeholder.markdown(note)
                    # print("=",information_after_llm)
                    st.session_state.messages.append({"role": "assistant", "content": note})

         
        # print("chat_history=", [(message["role"], message["content"]) for message in st.session_state.messages])

    
    #  Speech function implementation
    # if st.button("Ask your queries ?"):
    if st.button( "Ask"):
        
        prompt  = recognize_speech()
        # prompt = recognize_from_microphone()
        #prompt = transcribe_audio()
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        print("chat =", [(message["role"], message["content"]) for message in st.session_state.messages])
        # print("chat_history=", [(message["role"], message["content"]) for message in st.session_state.messages])
        # recent_messages = st.session_state.messages[-2:]
        
                
        try:
            with st.spinner("Processing..."):
                answer = agent_executor.invoke({"input":prompt ,"chat_history":[(message["role"], message["content"]) for message in st.session_state.messages[-2:]]})
                #result = answer['output']
                result_ = answer['output']
                result = remove_markdown_images(result_)
            #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
            intermediate_steps = answer["intermediate_steps"]
            tool_usage = []
            for steps in intermediate_steps:
                tool_usage.append(steps[0].tool)
           
            if 'query_documents' in tool_usage:
                results = vector_db.similarity_search_with_score(result)
                list_of_dicts = []
                for res in results:
                    score = res[1]
                    filename = res[0].metadata['source']
                    filename = filename.split("\\")[-1]
                    page_number = res[0].metadata['page'] + 1
 
                    dictionary = {"source": filename, "page": page_number, "score": score}
                    list_of_dicts.append(dictionary)
 
                sorted_list_of_dicts = sorted(list_of_dicts, key=lambda x: x["score"])
                   
            else:
                sorted_list_of_dicts = []
            #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#    
            print("answer__=",answer)
            print("prompt=", prompt)
            # print("result=", result)
            print("answer=",result)

            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = result
                message_placeholder.markdown(full_response + "|")
            message_placeholder.markdown(full_response)
            #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
            if sorted_list_of_dicts:
                for dictionary in sorted_list_of_dicts:
                    dictionary.pop("score", None)
                st.markdown("<font size='2'>Sources:</font>", unsafe_allow_html=True)  # Add a smaller text header
                st.markdown("<ul style='margin-top: 0; padding-left: 20px;'>", unsafe_allow_html=True)
                i = 0
                for item in sorted_list_of_dicts:
                    if i == 1:
                        break
                    st.markdown(f"<li><font size='2'>{item}</font></li>", unsafe_allow_html=True)
                    i += 1
                st.markdown("</ul>", unsafe_allow_html=True)
            #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
            print("result=",result)
            # speak_response(result)
            st.session_state.messages.append({"role": "assistant", "content": full_response})
        
        except Exception as e:
            print("exception=",e)
            note = "Please Try Again, Reframe the Query or Provide More Context and Try Again"
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                # information_after_llm =str(note)
                message_placeholder.markdown(note + "|",unsafe_allow_html=True)  # Display the error message
            message_placeholder.markdown(note)
            # print("=",information_after_llm)
            st.session_state.messages.append({"role": "assistant", "content": note})
         
        # print("chat_history=", [(message["role"], message["content"]) for message in st.session_state.messages])


        
if __name__ == "__main__":
    # st.set_option('server.enableCORS', True)
    main()

