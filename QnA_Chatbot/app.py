import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

import os
from dotenv import load_dotenv

load_dotenv()

os.environ['LANGCHAIN_TRACING_V2'] = "true"
os.environ['LANGCHAIN_PROJECT'] = "Q&A Chatbot with Gemini"

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Please respond to the user queries."),
        ("user", "Question: {question}")
    ]
)

def generate_response(question, api_key, llm_model="gemini-1.5-flash", temperature=0.7, max_tokens=256):
    os.environ["GOOGLE_API_KEY"] = api_key  
    llm = ChatGoogleGenerativeAI(
        model=llm_model,
        temperature=temperature,
        max_output_tokens=max_tokens
    )
    output_parser = StrOutputParser()
    chain = prompt | llm | output_parser
    answer = chain.invoke({'question': question})
    return answer

st.title("Enhanced QnA Chatbot")
st.sidebar.title("Settings")
api_key = st.sidebar.text_input("Enter your own Gemini API key", type="password")

temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7)
max_tokens = st.slider("Max Tokens", min_value=50, max_value=300, value=150)

st.write("Go and ask any question")
user_input = st.text_input("You:")

if user_input and api_key:
    response = generate_response(user_input, api_key, temperature=temperature, max_tokens=max_tokens)
    st.write("**Answer:**", response)
elif user_input:
    st.write("Please provide your Gemini API key in the sidebar.")
else:
    st.write("Please provide the query.")