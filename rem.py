import streamlit as st
from lida import Manager, TextGenerationConfig , llm  
from dotenv import load_dotenv
import os
import openai
from PIL import Image
from io import BytesIO
import base64
import pandas as pd
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_experimental.tools import PythonAstREPLTool
from langchain.output_parsers.openai_tools import JsonOutputToolsParser
from langchain_core.runnables import Runnable, RunnableLambda, RunnablePassthrough
from operator import itemgetter
from langchain_core.messages import ToolMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import MessagesPlaceholder
from llmx import llm

from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI

# load_dotenv()



class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource."""

    Pipeline: Literal["P1", "P2"] = Field(
        ...,
        description="Given a user question choose which Pipeline would be most relevant for answering their question",
    )


llm1 = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)
structured_llm = llm1.with_structured_output(RouteQuery)

system = """There are two code pipelines which are there after you give your verdict, 
You are an expert at routing a user question to the appropriate pipeline namely P1 and P2.

P1: it is the pipeline which generates the graph based on what is user asking
P2: it is the pipeline which gives answer of the question asked by user in text format

you have to decide which can be the better representation of the final answer, 

EXAMPLE:
Question: How many women died?
Answer:P2
Reason: As it is asking for just one number no need to generate a graph.

Based on your understanding, route it to the relevant pipeline."""
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "{question}"),
    ]
)

router = prompt | structured_llm


def base64_to_image(base64_string):
    # Decode the base64 string 
    byte_data = base64.b64decode(base64_string)
    
    # Use BytesIO to convert the byte data to image
    return Image.open(BytesIO(byte_data))


lida = Manager(text_gen = llm("openai"))
textgen_config = TextGenerationConfig(n=1, temperature=0.5, model="gpt-3.5-turbo-0301", use_cache=True)

menu = st.sidebar.selectbox("Choose an Option", ["Summarize", "Question Answer"])
# menu = st.sidebar.selectbox("Choose an Option", ["Question based Graph"])
file_uploader = st.file_uploader("Upload your CSV", type="csv")
if menu == "Summarize":
    st.subheader("Summarization of your Data")
    if file_uploader is not None:
        path_to_save = "filename.csv"
        with open(path_to_save, "wb") as f:
            f.write(file_uploader.getvalue())
        summary = lida.summarize("filename.csv", summary_method="default", textgen_config=textgen_config)
        st.write(summary)
        goals = lida.goals(summary, n=2, textgen_config=textgen_config)
        for goal in goals:
            st.write(goal)
        i = 0
        library = "matplotlib"
        textgen_config = TextGenerationConfig(n=1, temperature=0.2, use_cache=True)
        charts = lida.visualize(summary=summary, goal=goals[i], textgen_config=textgen_config, library=library)  
        img_base64_string = charts[0].raster
        img = base64_to_image(img_base64_string)
        st.image(img)
        
if menu == "Question Answer":
    st.subheader("Query Answer")
    if file_uploader is not None:
        path_to_save = "filename1.csv"
        with open(path_to_save, "wb") as f:
            f.write(file_uploader.getvalue())
        text_area = st.text_area("Query your Data to Generate Graph", height=200)
        if st.button("Answer"):
            if len(text_area) > 0:
                if router.invoke({"question":text_area}).Pipeline=="P1":
                    st.info("Your Query: " + text_area)
                    lida = Manager(text_gen = llm("openai")) 
                    textgen_config = TextGenerationConfig(n=1, temperature=0.2, use_cache=True)
                    summary = lida.summarize("filename1.csv", summary_method="default", textgen_config=textgen_config)
                    user_query = text_area
                    charts = lida.visualize(summary=summary, goal=user_query, textgen_config=textgen_config, library='matplotlib')  
                    charts[0]
                    image_base64 = charts[0].raster
                    img = base64_to_image(image_base64)
                    st.image(img)

                if router.invoke({"question":text_area}).Pipeline=='P2':
                    if file_uploader is not None:  
                        df=pd.read_csv(file_uploader)
                        tool = PythonAstREPLTool(locals={"df": df})

                        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
                    
                        llm_with_tools = llm.bind_tools([tool], tool_choice=tool.name)
                    
                        parser = JsonOutputToolsParser()
                    
                        system = f"""You have access to a pandas dataframe `df`. \
                        Here is the output of `df.head().to_markdown()`:
                    
                        ```
                        {df.head().to_markdown()}
                        ```
                    
                        Given a user question, write the Python code to answer it. \
                        Don't assume you have access to any libraries other than built-in Python ones and pandas.
                        Respond directly to the question once you have enough information to answer it."""
                        prompt = ChatPromptTemplate.from_messages(
                            [
                                (
                                    "system",
                                    system,
                                ),
                                ("human", "{question}"),
                                MessagesPlaceholder("chat_history", optional=True),
                            ]
                        )
                        
                        def _get_chat_history(x: dict) -> list:
                            """Parse the chain output up to this point into a list of chat history messages to insert in the prompt."""
                            ai_msg = x["ai_msg"]
                            tool_call_id = x["ai_msg"].additional_kwargs["tool_calls"][0]["id"]
                            tool_msg = ToolMessage(tool_call_id=tool_call_id, content=str(x["tool_output"]))
                            return [ai_msg, tool_msg]
                    
                            
                        chain = (
                                    RunnablePassthrough.assign(ai_msg=prompt | llm_with_tools)
                                    .assign(tool_output=itemgetter("ai_msg") | parser| {"query": lambda x: x[0]["args"]["query"]})
                                    .assign(chat_history=_get_chat_history)
                                    .assign(response=prompt | llm | StrOutputParser())
                                    .pick(["response"])
                                )
                            
                        ans=chain.invoke({"question": text_area})['response']
                        st.success(ans)