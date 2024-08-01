import streamlit as st 
from langchain_community.llms import Ollama 
from langchain.prompts import PromptTemplate 
from langchain.chains import LLMChain,LLMMathChain
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.agents.agent_types import AgentType 
from langchain.agents import Tool,initialize_agent 


st.title("Text-2-Math Solver")
question = st.text_area("Please enter your question here:")


def logical(question):
    api = WikipediaAPIWrapper()
    wiki = Tool(
        name = "wikipedia",
        func = api.run,
        description = "This tool will search the internet for the logical solution",
    )

    llm = Ollama(model='llama3',temperature = 0)

    math_chain = LLMMathChain.from_llm(llm=llm)

    math = Tool(
        name = "calculator",
        func = math_chain.run,
        description = "This tool will calculate the math problem",
    )

    prompt = """ you are an agent tasked for solving user mathematical question.Logically arrive 
    to a solution and display it in pointwise for the question below.
    Question: {question}
    Answer:

            """
    
    prompt_template = PromptTemplate(input_variables=['question'],template=prompt)

    chain = LLMChain(llm=llm,prompt=prompt_template)
    reason = Tool(
        name = "reasoning",
        func = chain.run,
        description = "This tool will logically provide answer for the solution",
    )

    agents = initialize_agent(
        tools = [wiki,math,reason],
        llm = llm,
        agent = AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose = True,
        handle_parsing_errors = True
    )

    response = agents.invoke({'input':question})
    return response

submit = st.button("Generate")
if submit:
    with st.spinner("Generating...."):
        st.write(logical(question))