import streamlit as st 
from langchain_community.llms import Ollama 
from langchain.prompts import PromptTemplate 
from langchain.chains import LLMChain,LLMMathChain
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.agents.agent_types import AgentType 
from langchain.agents import Tool,initialize_agent 
from langchain.callbacks import StreamlitCallbackHandler


st.title("Text-2-Math Solver")



api = WikipediaAPIWrapper()
wiki = Tool(
    name = "wikipedia",
    func = api.run,
    description = "This tool will search the internet for the logical solution",
)
llm = Ollama(model='llama3',temperature = 0.05)
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

if "messages" not in st.session_state:
    st.session_state["messages"]=[
    {"role":"assistant","content":"Hi, I'm a Math chatbot who can answer all your maths questions"}
]
    
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg['content'])


question=st.text_area("Enter your question:","I have 5 bananas and 7 grapes. I eat 2 bananas and give away 3 grapes. Then I buy a dozen apples and 2 packs of blueberries and each pack of blueberries contains 25 berries. How many total pieces of fruit do I have at the end?")
if st.button("find my answer"):
    if question:
        with st.spinner("Generate response.."):
            st.session_state.messages.append({"role":"user","content":question})
            st.chat_message("user").write(question)
            st_cb=StreamlitCallbackHandler(st.container(),expand_new_thoughts=False)
            response=agents.run(st.session_state.messages,callbacks=[st_cb]
                                         )
            st.session_state.messages.append({'role':'assistant',"content":response})
            st.write('### Response:')
            st.success(response)
    else:
        st.warning("Please enter the question")