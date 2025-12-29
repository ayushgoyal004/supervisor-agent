from dotenv import load_dotenv
from typing import Literal, Optional
import uuid

from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

from langgraph.graph import START, END, StateGraph, MessagesState
from langgraph.prebuilt import ToolNode
from langgraph.types import interrupt
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore

from typing import Annotated
from langgraph.store.base import BaseStore
from langgraph.prebuilt import InjectedStore

load_dotenv()

model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# ------------------------------------------------
# 1. Memory Tools & Logic
# ------------------------------------------------

@tool
def save_user_preference(
    preference: str, 
    context: str, 
    config: RunnableConfig, 
    # Use Annotated and InjectedStore here:
    store: Annotated[BaseStore, InjectedStore()] 
):
    """
    Call this to remember a user's preference, favorite table, or specific SQL logic.
    """
    user_id = config["configurable"].get("user_id", "default_user")
    memory_id = str(uuid.uuid4())
    
    store.put(
        ("memories", user_id),
        memory_id,
        {"preference": preference, "context": context}
    )
    return f"Saved to memory: {preference}"

# ------------------------------------------------
# 2. Agent Factory
# ------------------------------------------------
def build_sql_agent(db_url: str):
    db = SQLDatabase.from_uri(db_url)
    toolkit = SQLDatabaseToolkit(db=db, llm=model)
    sql_tools = toolkit.get_tools()

    get_schema_tool = next(t for t in sql_tools if t.name == "sql_db_schema")
    list_tables_tool = next(t for t in sql_tools if t.name == "sql_db_list_tables")
    sql_run_tool = next(t for t in sql_tools if t.name == "sql_db_query")

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
            return sql_run_tool.invoke({"query": confirmation["query"]}, config)
        return "Query execution rejected by user."

    # --- Nodes ---

    def list_tables_node(state: MessagesState):
        tables = list_tables_tool.invoke({})
        return {
            "messages": [AIMessage(content=f"Connected. Tables: {tables}. Fetching schema...")]
        }

    def schema_node(state: MessagesState):
        llm_with_schema = model.bind_tools([get_schema_tool])
        return {"messages": [llm_with_schema.invoke(state["messages"])]}

    def generate_query_node(state: MessagesState, config: RunnableConfig, store: BaseStore):
        # SEARCH MEMORY: Fetch memories for this specific user
        user_id = config["configurable"].get("user_id", "default_user")
        memories = store.search(("memories", user_id))
        
        memory_str = "\n".join(
            [f"- {m.value['preference']} (Context: {m.value['context']})" for m in memories]
        ) if memories else "No previous preferences found."

        system_msg = SystemMessage(content=f"""
        You are a SQL expert for {db.dialect}. 
        
        USER PREFERENCES & MEMORIES:
        {memory_str}

        RULES:
        1. Use the memory above to tailor your queries (e.g., if user likes specific columns).
        2. If the user mentions a preference (e.g., 'I always want to see prices in USD'), 
           call 'save_user_preference' to remember it.
        3. Only perform SELECT queries. Max results: 5.
        """)

        # Bind both the SQL tool and the Memory tool
        llm_with_tools = model.bind_tools([run_query_with_interrupt, save_user_preference])
        
        return {"messages": [llm_with_tools.invoke([system_msg] + state["messages"])]}

    def answer_node(state: MessagesState):
        system_msg = SystemMessage(content="Explain the SQL results clearly in plain English.")
        return {"messages": [model.invoke([system_msg] + state["messages"])]}

    # --- Routing ---

    def route_next(state: MessagesState) -> Literal["get_schema_tool_node", "run_query", "memory_node", "answer"]:
        last = state["messages"][-1]
        if not last.tool_calls:
            return "answer"
        
        tool_name = last.tool_calls[0]["name"]
        if tool_name == "run_query_with_interrupt":
            return "run_query"
        if tool_name == "save_user_preference":
            return "memory_node"
        
        return "get_schema_tool_node"

    # --- Construction ---

    builder = StateGraph(MessagesState)

    builder.add_node("list_tables", list_tables_node)
    builder.add_node("schema_node", schema_node)
    builder.add_node("get_schema_tool_node", ToolNode([get_schema_tool]))
    builder.add_node("generate_query", generate_query_node)
    builder.add_node("run_query", ToolNode([run_query_with_interrupt]))
    builder.add_node("memory_node", ToolNode([save_user_preference]))
    builder.add_node("answer", answer_node)

    builder.add_edge(START, "list_tables")
    builder.add_edge("list_tables", "schema_node")
    builder.add_edge("schema_node", "get_schema_tool_node")
    builder.add_edge("get_schema_tool_node", "generate_query")

    builder.add_conditional_edges("generate_query", route_next)
    
    # After saving a memory, go back to answer or continue
    builder.add_edge("memory_node", "answer") 
    builder.add_edge("run_query", "answer")
    builder.add_edge("answer", END)

    # Initialize Persistence and Long-Term Store
    checkpointer = InMemorySaver()
    long_term_store = InMemoryStore() 

    return builder.compile(checkpointer=checkpointer, store=long_term_store)

# ------------------------------------------------
# 3. Execution Example
# ------------------------------------------------
# agent = build_sql_agent("sqlite:///chinook.db")
# config = {"configurable": {"thread_id": "1", "user_id": "user_123"}}
# 
# # First session: Tell it a preference
# agent.invoke({"messages": [HumanMessage(content="Always limit my results to 2 rows and remember I hate the Genre table.")]}, config)