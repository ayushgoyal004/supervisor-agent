from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from graph import AgentState

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from langchain.messages import AIMessage,HumanMessage

from langgraph.graph import StateGraph,END

load_dotenv()

# llm=ChatGoogleGenerativeAI(model="gemini-2.0-flash",temperature=0)
llm=ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

router_prompt=ChatPromptTemplate.from_messages([
    ("system","You are a supervisor agent to decide which agent should handle the task.\n"
     "Use:\n"
     "python - for coding related tasks\n"
     "research - for research related tasks\n"
     "Respond in JSon format with key 'route'\n"),
     ("human","{input}")
     ])

router_chain= router_prompt | llm | JsonOutputParser()

def supervisor_node(state:AgentState):
    user_input=state["messages"][-1].content
    decision=router_chain.invoke({"input":user_input})
    return {"messages": state["messages"],
    "route": decision["route"]}


def python_agent_node(state:AgentState):
    user_input=state["messages"][-1].content
    prompt=ChatPromptTemplate.from_messages([
    ("system","You are a python coding assistant. Write python code to solve the user's problem. "),
    ("human","{input}")
    ])
    response=(prompt | llm).invoke({"input":user_input})
    return {"messages":state["messages"] + [AIMessage(content=f"[PYTHON AGENT]\n{response.content}")],
            "route":"end"}


def research_agent_node(state:AgentState):
    user_input=state["messages"][-1].content
    prompt=ChatPromptTemplate.from_messages([
    ("system","You are a research assistant. Provide detailed research information to the user."),
    ("human","{input}")
    ])
    response=(prompt | llm).invoke({"input":user_input})
    return {"messages":state["messages"] + [AIMessage(content=f"[RESEARCH AGENT]\n{response.content}")],
            "route":"end"}

#Graph building
graph=StateGraph(AgentState)

graph.add_node("supervisor",supervisor_node)
graph.add_node("python_agent",python_agent_node)
graph.add_node("research_agent",research_agent_node)

graph.set_entry_point("supervisor")


graph.add_conditional_edges("supervisor",
                           lambda state:state["route"],
                           {"python":"python_agent",
                            "research":"research_agent"})

graph.add_edge("python_agent",END)
graph.add_edge("research_agent",END)

app=graph.compile()



# result = app.invoke({
#     "messages": [HumanMessage(content="Write Python code to reverse a list")]
#     # "messages": [HumanMessage(content="Explain how Retrieval-Augmented Generation works")]
# })

# print(result["messages"][-1].content)