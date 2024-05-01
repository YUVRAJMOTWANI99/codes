import streamlit as st 
import pandas as pd
import pymysql
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain import PromptTemplate
from langchain.chains import LLMChain
import re
from langchain_community.agent_toolkits import create_sql_agent
from langchain_openai import ChatOpenAI
# from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.agents.agent_types import AgentType
from langchain_community.utilities import SQLDatabase
import getpass
import os
from PIL import Image
from io import BytesIO
from lida import Manager, TextGenerationConfig , llm  
import base64

os.environ["OPENAI_API_KEY"] = 'PUT_YOUR_API_KEY'

db = SQLDatabase.from_uri("sqlite:///chinook.db")

st.subheader("Query Answer")
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
agent_executor = create_sql_agent(llm, db=db, agent_type="openai-tools", verbose=True, agent_executor_kwargs={"return_intermediate_steps": True})

query=st.text_area("Query your Data", height=200)
if st.button("answer"):
    response = agent_executor.invoke(query)
    st.subheader("ans from agent")
    st.success(response['output'])


    for i in range(len(response['intermediate_steps'])):
        string=response['intermediate_steps'][i][0].log


    generic_template = '''
        You are a helpful assistant that takes string which have SQL queries written in it,
        Please identify it and return it,only return sql query and nothing else, do not add anything 
        return 'No SQL Query' if there is no query in string.
        String :`{string}`
        '''
        
        # Initialize prompt template
    prompt = PromptTemplate(
            input_variables=['string'],
            template=generic_template
        )
        
        # Initialize LLMChain
    llm_chain = LLMChain(llm=llm, prompt=prompt)
        
        # Run the LLMChain with input values
    ref_query = llm_chain.run({'string': string})

    st.subheader("generated sql query")
    st.success(ref_query)

    db_name = "chinook"
    db_host = "localhost"
    db_username = "root"
    db_password = "PUT_YOUR_PASSWORD"

    try:
        conn = pymysql.connect(host = db_host,
                            port = int(3306),
                            user = "root",
                            password = db_password,
                            db = db_name)
    except :
        print ("error")
    if conn:
        print ("connection successful")
    else:
            print ("error")
    print(ref_query)
    df = pd.read_sql_query(ref_query, conn)
    st.subheader("table output from data base")
    st.table(df)


    def base64_to_image(base64_string):

        byte_data = base64.b64decode(base64_string)
        
    
        return Image.open(BytesIO(byte_data))

    df.to_csv("filename1.csv")
    lida = Manager(text_gen = llm("openai")) 
    textgen_config = TextGenerationConfig(n=1, temperature=0.2, use_cache=True)
    summary = lida.summarize("filename1.csv", summary_method="default", textgen_config=textgen_config)
    user_query = query
    charts = lida.visualize(summary=summary, goal=user_query, textgen_config=textgen_config, library='matplotlib')  
    charts[0]
    image_base64 = charts[0].raster
    img = base64_to_image(image_base64)
    st.subheader("generated graph")
    st.image(img)