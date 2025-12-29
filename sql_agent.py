from dotenv import load_dotenv
from typing import Literal

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_core.messages import AIMessage, SystemMessage,HumanMessage,ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

from langgraph.graph import START, END, StateGraph, MessagesState
from langgraph.prebuilt import ToolNode
from langgraph.types import interrupt
from langgraph.checkpoint.memory import InMemorySaver

# ------------------------------------------------
# Environment + Model
# ------------------------------------------------
load_dotenv()

# model = ChatGoogleGenerativeAI(
#     model="gemini-2.0-flash",
#     temperature=0
# )
model=ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

# ------------------------------------------------
# Agent Factory (DB comes ONLY from runtime)
# ------------------------------------------------
def build_sql_agent(db_url: str):
    """
    Build a SQL Agent for ANY database URL.
    No defaults. No auto-downloads.

    Examples:
      sqlite:///my.db
      postgresql://user:pass@host:5432/db
      mysql+pymysql://user:pass@host/db
    """

    # ---- DB + Toolkit ----
    db = SQLDatabase.from_uri(db_url)
    toolkit = SQLDatabaseToolkit(db=db, llm=model)
    sql_tools = toolkit.get_tools()

    get_schema_tool = next(t for t in sql_tools if t.name == "sql_db_schema")
    list_tables_tool = next(t for t in sql_tools if t.name == "sql_db_list_tables")
    sql_run_tool = next(t for t in sql_tools if t.name == "sql_db_query")

    # ------------------------------------------------
    # Human-in-the-loop Tool
    # ------------------------------------------------
    @tool
    def run_query_with_interrupt(config: RunnableConfig, query: str):
        """Execute SQL only after human approval."""
        confirmation = interrupt({
            "query": query,
            "message": "Please review or edit the SQL query."
        })

        if confirmation["type"] == "accept":
            return sql_run_tool.invoke({"query": query}, config)

        elif confirmation["type"] == "edit":
            return sql_run_tool.invoke(
                {"query": confirmation["query"]}, config
            )

        else:
            return "Query execution rejected by user."

    # ------------------------------------------------
    # Graph Nodes
    # ------------------------------------------------
    def list_tables_node(state: MessagesState):
        tables = list_tables_tool.invoke({})
        return {
            "messages": [
                AIMessage(
                    content=f"Connected successfully. Available tables: {tables}."
                            f"Fetching schema next."
                )
            ]
        }

    def schema_node(state: MessagesState):
        llm_with_schema = model.bind_tools([get_schema_tool])
        return {
            "messages": [llm_with_schema.invoke(state["messages"])]
        }


    generate_query_system_prompt="""
    You are an agent designed to interact with a SQL database.
    Given an input question, create a syntactically correct {dialect} query to run,
    then look at the results of the query and return the answer. Unless the user
    specifies a specific number of examples they wish to obtain, always limit your
    query to at most {top_k} results.

    You can order the results by a relevant column to return the most interesting
    examples in the database. Never query for all the columns from a specific table,
    only ask for the relevant columns given the question.

    DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.
    """.format(
        dialect=db.dialect,
        top_k=5,
    )


    def generate_query_node(state: MessagesState):
        system_msg = SystemMessage(
            # content=f"You are a {db.dialect} expert. "
            #         f"Generate a safe, read-only SQL query. "
            #         f"Limit results to 5."
            content= generate_query_system_prompt
        )
        llm_with_run = model.bind_tools([run_query_with_interrupt])
        return {
            "messages": [
                llm_with_run.invoke([system_msg] + state["messages"])
            ]
        }


    def answer_node(state: MessagesState):
        #To debug
        print(f"Messages in state: {len(state['messages'])}")
        system_msg = SystemMessage(
            content="Explain the SQL results clearly in plain English."
        )
        return {
            "messages": [model.invoke([system_msg] + state["messages"])]
        }


    def route_next(state: MessagesState) -> Literal[
        "get_schema_tool_node", "run_query", "answer"
    ]:
        last = state["messages"][-1]

        if not last.tool_calls:
            return "answer"

        tool_name = last.tool_calls[0]["name"]

        if tool_name == "run_query_with_interrupt":
            return "run_query"

        return "get_schema_tool_node"

    # ------------------------------------------------
    # Graph Construction
    # ------------------------------------------------
    builder = StateGraph(MessagesState)

    builder.add_node("list_tables", list_tables_node)
    builder.add_node("schema_node", schema_node)
    builder.add_node("get_schema_tool_node", ToolNode([get_schema_tool]))
    builder.add_node("generate_query", generate_query_node)
    builder.add_node("run_query", ToolNode([run_query_with_interrupt]))
    builder.add_node("answer", answer_node)

    builder.add_edge(START, "list_tables")
    builder.add_edge("list_tables", "schema_node")
    builder.add_edge("schema_node", "get_schema_tool_node")
    builder.add_edge("get_schema_tool_node", "generate_query")

    builder.add_conditional_edges("generate_query", route_next)

    builder.add_edge("run_query", "answer")
    builder.add_edge("answer", END)

    return builder.compile(
        checkpointer=InMemorySaver()
    )
