# from dotenv import load_dotenv
# from typing import Literal, Optional
# import uuid

# from langchain_openai import ChatOpenAI
# from langchain_community.utilities import SQLDatabase
# from langchain_community.agent_toolkits import SQLDatabaseToolkit
# from langchain_core.messages import AIMessage, SystemMessage, HumanMessage, ToolMessage
# from langchain_core.runnables import RunnableConfig
# from langchain_core.tools import tool

# from langgraph.graph import START, END, StateGraph, MessagesState
# from langgraph.prebuilt import ToolNode
# from langgraph.types import interrupt
# from langgraph.checkpoint.memory import InMemorySaver
# from langgraph.store.base import BaseStore
# from langgraph.store.memory import InMemoryStore

# from typing import Annotated
# from langgraph.store.base import BaseStore
# from langgraph.prebuilt import InjectedStore

# load_dotenv()

# model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# # ------------------------------------------------
# # 1. Memory Tools & Logic
# # ------------------------------------------------

# @tool
# def save_user_preference(
#     preference: str, 
#     context: str, 
#     config: RunnableConfig, 
#     # Use Annotated and InjectedStore here:
#     store: Annotated[BaseStore, InjectedStore()] 
# ):
#     """
#     Call this to remember a user's preference, favorite table, or specific SQL logic.
#     """
#     user_id = config["configurable"].get("user_id", "default_user")
#     memory_id = str(uuid.uuid4())
    
#     store.put(
#         ("memories", user_id),
#         memory_id,
#         {"preference": preference, "context": context}
#     )
#     return f"Saved to memory: {preference}"

# # ------------------------------------------------
# # 2. Agent Factory
# # ------------------------------------------------
# def build_sql_agent(db_url: str):
#     db = SQLDatabase.from_uri(db_url)
#     toolkit = SQLDatabaseToolkit(db=db, llm=model)
#     sql_tools = toolkit.get_tools()

#     get_schema_tool = next(t for t in sql_tools if t.name == "sql_db_schema")
#     list_tables_tool = next(t for t in sql_tools if t.name == "sql_db_list_tables")
#     sql_run_tool = next(t for t in sql_tools if t.name == "sql_db_query")

#     @tool
#     def run_query_with_interrupt(config: RunnableConfig, query: str):
#         """Execute SQL only after human approval."""
#         confirmation = interrupt({
#             "query": query,
#             "message": "Please review or edit the SQL query."
#         })

#         if confirmation["type"] == "accept":
#             return sql_run_tool.invoke({"query": query}, config)
#         elif confirmation["type"] == "edit":
#             return sql_run_tool.invoke({"query": confirmation["query"]}, config)
#         return "Query execution rejected by user."

#     # --- Nodes ---

#     def list_tables_node(state: MessagesState):
#         tables = list_tables_tool.invoke({})
#         return {
#             "messages": [AIMessage(content=f"Connected. Tables: {tables}. Fetching schema...")]
#         }

#     def schema_node(state: MessagesState):
#         llm_with_schema = model.bind_tools([get_schema_tool])
#         return {"messages": [llm_with_schema.invoke(state["messages"])]}

#     def generate_query_node(state: MessagesState, config: RunnableConfig, store: BaseStore):
#         # SEARCH MEMORY: Fetch memories for this specific user
#         user_id = config["configurable"].get("user_id", "default_user")
#         memories = store.search(("memories", user_id))
        
#         memory_str = "\n".join(
#             [f"- {m.value['preference']} (Context: {m.value['context']})" for m in memories]
#         ) if memories else "No previous preferences found."

#         system_msg = SystemMessage(content=f"""
#         You are a SQL expert for {db.dialect}. 
        
#         USER PREFERENCES & MEMORIES:
#         {memory_str}

#         RULES:
#         1. Use the memory above to tailor your queries (e.g., if user likes specific columns).
#         2. If the user mentions a preference (e.g., 'I always want to see prices in USD'), 
#            call 'save_user_preference' to remember it.
#         3. Only perform SELECT queries. Max results: 5.
#         """)

#         # Bind both the SQL tool and the Memory tool
#         llm_with_tools = model.bind_tools([run_query_with_interrupt, save_user_preference])
        
#         return {"messages": [llm_with_tools.invoke([system_msg] + state["messages"])]}

#     def answer_node(state: MessagesState):
#         system_msg = SystemMessage(content="Explain the SQL results clearly in plain English.")
#         return {"messages": [model.invoke([system_msg] + state["messages"])]}

#     # --- Routing ---

#     def route_next(state: MessagesState) -> Literal["get_schema_tool_node", "run_query", "memory_node", "answer"]:
#         last = state["messages"][-1]
#         if not last.tool_calls:
#             return "answer"
        
#         tool_name = last.tool_calls[0]["name"]
#         if tool_name == "run_query_with_interrupt":
#             return "run_query"
#         if tool_name == "save_user_preference":
#             return "memory_node"
        
#         return "get_schema_tool_node"

#     # --- Construction ---

#     builder = StateGraph(MessagesState)

#     builder.add_node("list_tables", list_tables_node)
#     builder.add_node("schema_node", schema_node)
#     builder.add_node("get_schema_tool_node", ToolNode([get_schema_tool]))
#     builder.add_node("generate_query", generate_query_node)
#     builder.add_node("run_query", ToolNode([run_query_with_interrupt]))
#     builder.add_node("memory_node", ToolNode([save_user_preference]))
#     builder.add_node("answer", answer_node)

#     builder.add_edge(START, "list_tables")
#     builder.add_edge("list_tables", "schema_node")
#     builder.add_edge("schema_node", "get_schema_tool_node")
#     builder.add_edge("get_schema_tool_node", "generate_query")

#     builder.add_conditional_edges("generate_query", route_next)
    
#     # After saving a memory, go back to answer or continue
#     builder.add_edge("memory_node", "answer") 
#     builder.add_edge("run_query", "answer")
#     builder.add_edge("answer", END)

#     # Initialize Persistence and Long-Term Store
#     checkpointer = InMemorySaver()
#     long_term_store = InMemoryStore() 

#     return builder.compile(checkpointer=checkpointer, store=long_term_store)

# ------------------------------------------------
# 3. Execution Example
# ------------------------------------------------
# agent = build_sql_agent("sqlite:///chinook.db")
# config = {"configurable": {"thread_id": "1", "user_id": "user_123"}}
# 
# # First session: Tell it a preference
# agent.invoke({"messages": [HumanMessage(content="Always limit my results to 2 rows and remember I hate the Genre table.")]}, config)



# from dotenv import load_dotenv
# from typing import Literal, Optional, Annotated
# import uuid

# from langchain_openai import ChatOpenAI
# from langchain_community.utilities import SQLDatabase
# from langchain_community.agent_toolkits import SQLDatabaseToolkit
# from langchain_core.messages import AIMessage, SystemMessage, HumanMessage, ToolMessage
# from langchain_core.runnables import RunnableConfig
# from langchain_core.tools import tool

# from langgraph.graph import START, END, StateGraph, MessagesState
# from langgraph.prebuilt import ToolNode, InjectedStore
# from langgraph.types import interrupt
# from langgraph.checkpoint.memory import InMemorySaver
# from langgraph.store.base import BaseStore
# from langgraph.store.memory import InMemoryStore

# load_dotenv()

# model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# # ------------------------------------------------
# # 1. Memory Tool (with Upsert Logic)
# # ------------------------------------------------
# @tool
# def upsert_user_memory(content: str, topic: str, config: RunnableConfig, store: Annotated[BaseStore, InjectedStore()]):
#     """
#     Saves or updates important information about the user (e.g., preferences, role, or SQL habits).
#     Use a concise 'topic' (like 'formatting' or 'favorite_tables') to ensure information stays organized.
#     """
#     user_id = config["configurable"].get("user_id", "default_user")
#     namespace = ("memories", user_id)
    
#     existing_items = store.search(namespace, query=topic)
#     existing_memory = next((item for item in existing_items if item.value.get("topic") == topic), None)

#     if existing_memory:
#         # 1. Get old content
#         old_content = existing_memory.value.get("content", "")
#         # 2. Append new content to the old string (or list)
#         new_content = f"{old_content}\n- {content}" 
#         # 3. Overwrite the key with the cumulative data
#         store.put(namespace, existing_memory.key, {"content": new_content, "topic": topic})
#         return f"Appended to {topic}."
#     else:
#         # Create first entry
#         store.put(namespace, str(uuid.uuid4()), {"content": f"- {content}", "topic": topic})
#         return f"Created new memory for {topic}."


# # ------------------------------------------------
# # 2. Agent Factory
# # ------------------------------------------------

# def build_sql_agent(db_url: str):
#     db = SQLDatabase.from_uri(db_url)
#     toolkit = SQLDatabaseToolkit(db=db, llm=model)
#     sql_tools = toolkit.get_tools()

#     get_schema_tool = next(t for t in sql_tools if t.name == "sql_db_schema")
#     list_tables_tool = next(t for t in sql_tools if t.name == "sql_db_list_tables")
#     sql_run_tool = next(t for t in sql_tools if t.name == "sql_db_query")

#     @tool
#     def run_query_with_interrupt(config: RunnableConfig, query: str):
#         """Execute SQL only after human approval."""
#         confirmation = interrupt({
#             "query": query,
#             "message": "Please review or edit the SQL query."
#         })

#         if confirmation["type"] == "accept":
#             return sql_run_tool.invoke({"query": query}, config)
#         elif confirmation["type"] == "edit":
#             return sql_run_tool.invoke({"query": confirmation["query"]}, config)
#         return "Query execution rejected by user."

#     # --- Nodes ---

#     def list_tables_node(state: MessagesState):
#         tables = list_tables_tool.invoke({})
#         return {"messages": [AIMessage(content=f"Connected. Tables: {tables}. Fetching schema...")]}

#     def schema_node(state: MessagesState):
#         llm_with_schema = model.bind_tools([get_schema_tool])
#         return {"messages": [llm_with_schema.invoke(state["messages"])]}

#     def generate_query_node(state: MessagesState, config: RunnableConfig, store: BaseStore):
#         user_id = config["configurable"].get("user_id", "default_user")
        
#         # Retrieve all memories for this user
#         memories = store.search(("memories", user_id))
        
#         memory_str = "\n".join(
#             [f"- {m.value['topic'].upper()}: {m.value['content']}" for m in memories]
#         ) if memories else "No previous preferences found."

#         system_msg = SystemMessage(content=f"""
#         You are a SQL expert for {db.dialect}. 
        
#         LONG-TERM USER MEMORY:
#         {memory_str}

#         INSTRUCTIONS:
#         1. Use the memory above to tailor queries.
#         2. If you learn something new about the user, call 'upsert_user_memory'.
#            Choose a clear topic (e.g., 'formatting', 'preferred_columns', 'user_role').
#         3. Only perform SELECT queries. Max results: 5.
#         """)

#         llm_with_tools = model.bind_tools([run_query_with_interrupt, upsert_user_memory])
#         return {"messages": [llm_with_tools.invoke([system_msg] + state["messages"])]}

#     def answer_node(state: MessagesState):
#         system_msg = SystemMessage(content="Explain the SQL results clearly in plain English.")
#         return {"messages": [model.invoke([system_msg] + state["messages"])]}

#     # --- Graph Construction ---

#     builder = StateGraph(MessagesState)

#     builder.add_node("list_tables", list_tables_node)
#     builder.add_node("schema_node", schema_node)
#     builder.add_node("get_schema_tool_node", ToolNode([get_schema_tool]))
#     builder.add_node("generate_query", generate_query_node)
#     builder.add_node("run_query", ToolNode([run_query_with_interrupt]))
#     builder.add_node("memory_node", ToolNode([upsert_user_memory]))
#     builder.add_node("answer", answer_node)

#     builder.add_edge(START, "list_tables")
#     builder.add_edge("list_tables", "schema_node")
#     builder.add_edge("schema_node", "get_schema_tool_node")
#     builder.add_edge("get_schema_tool_node", "generate_query")

#     def route_next(state: MessagesState):
#         last = state["messages"][-1]
#         if not last.tool_calls: return "answer"
#         t_name = last.tool_calls[0]["name"]
#         if t_name == "run_query_with_interrupt": return "run_query"
#         if t_name == "upsert_user_memory": return "memory_node"
#         return "get_schema_tool_node"

#     builder.add_conditional_edges("generate_query", route_next)
    
#     builder.add_edge("memory_node", "answer") 
#     builder.add_edge("run_query", "answer")
#     builder.add_edge("answer", END)

#     return builder.compile(checkpointer=InMemorySaver(), store=InMemoryStore())



import os
import json
import uuid
from typing import Literal, Optional, Annotated
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

from langgraph.graph import START, END, StateGraph, MessagesState
from langgraph.prebuilt import ToolNode, InjectedStore
from langgraph.types import interrupt
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore

load_dotenv()

model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# ------------------------------------------------
# 1. Local Disk Persistence Helper
# ------------------------------------------------
MEMORY_FILE = "local_memories.json"

def save_to_disk(store: BaseStore, user_id: str):
    """Mirror the InMemoryStore to a local JSON file."""
    memories = store.search(("memories", user_id))
    data = {m.key: m.value for m in memories}
    
    # Load existing file to preserve other users
    all_data = {}
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, "r") as f:
            all_data = json.load(f)
    
    all_data[user_id] = data
    with open(MEMORY_FILE, "w") as f:
        json.dump(all_data, f, indent=4)

# ------------------------------------------------
# 2. Memory Tools (Manual & Automatic)
# ------------------------------------------------

@tool
def upsert_user_memory(content: str, topic: str, config: RunnableConfig, store: Annotated[BaseStore, InjectedStore()]):
    """Manual tool for the agent to save specific preferences."""
    user_id = config["configurable"].get("user_id", "default_user")
    namespace = ("memories", user_id)
    
    existing_items = store.search(namespace, query=topic)
    existing_memory = next((item for item in existing_items if item.value.get("topic") == topic), None)

    if existing_memory:
        new_content = f"{existing_memory.value.get('content', '')}\n- {content}" 
        store.put(namespace, existing_memory.key, {"content": new_content, "topic": topic})
    else:
        store.put(namespace, str(uuid.uuid4()), {"content": f"- {content}", "topic": topic})
    
    save_to_disk(store, user_id)
    return f"Updated memory for {topic}."


@tool
def clear_all_memories(config: RunnableConfig, store: Annotated[BaseStore, InjectedStore()]):
    """Clears all long-term memories for the current user."""
    user_id = config["configurable"].get("user_id", "default_user")
    namespace = ("memories", user_id)
    memories = store.search(namespace)
    for m in memories:
        store.delete(namespace, m.key)
    save_to_disk(store, user_id)
    return "All memories cleared."

# ------------------------------------------------
# 3. Agent Factory
# ------------------------------------------------

def build_sql_agent(db_url: str):
    db = SQLDatabase.from_uri(db_url)
    toolkit = SQLDatabaseToolkit(db=db, llm=model)
    sql_tools = toolkit.get_tools()

    get_schema_tool = next(t for t in sql_tools if t.name == "sql_db_schema")
    list_tables_tool = next(t for t in sql_tools if t.name == "sql_db_list_tables")
    sql_run_tool = next(t for t in sql_tools if t.name == "sql_db_query")

    # --- Automatic Memory Extraction Node ---
    # def automatic_memory_node(state: MessagesState, config: RunnableConfig, store: BaseStore):
    #     """Automatically called after every response to extract facts."""
    #     user_id = config["configurable"].get("user_id", "default_user")
        
    #     # Get the last conversation exchange
    #     last_user_msg = [m for m in state["messages"] if isinstance(m, HumanMessage)][-1].content
    #     last_ai_msg = [m for m in state["messages"] if isinstance(m, AIMessage)][-1].content

    #     extraction_prompt = f"""
    #     Review this exchange and extract one key preference or technical detail about the user's DB needs.
    #     User: {last_user_msg}
    #     AI: {last_ai_msg}
        
    #     If it's a new preference, return: TOPIC|FACT
    #     Example: formatting|User prefers uppercase SQL keywords.
    #     If nothing important, return: None
    #     """
        
    #     res = model.invoke(extraction_prompt).content
    #     if "|" in res:
    #         topic, fact = res.split("|", 1)
    #         # Use the tool logic internally
    #         upsert_user_memory.invoke({"content": fact.strip(), "topic": topic.strip()}, config)
        
    #     return state
    def automatic_memory_node(state: MessagesState, config: RunnableConfig, store: BaseStore):
        user_id = config["configurable"].get("user_id", "default_user")
        namespace = ("memories", user_id)
        
        # 1. Fetch current memories to provide context for updates/deletes
        existing_memories = store.search(namespace)
        memory_context = "\n".join([f"ID: {m.key} | {m.value['topic']}: {m.value['content']}" for m in existing_memories])

        extraction_prompt = f"""
        You are a memory management module. Review the latest exchange and determine if there is 
        information worth remembering for long-term SQL assistance.

        Current Memories:
        {memory_context}

        Latest Exchange:
        User: {state['messages'][-2].content if len(state['messages']) > 1 else ""}
        AI: {state['messages'][-1].content}

        Rules:
        1. If the user expresses a preference (e.g., "Always use LIMIT 10", "I prefer snake_case"), SAVE/UPDATE it.
        2. If the user corrects a previous preference, UPDATE it.
        3. If a memory is no longer true, DELETE it.
        4. If no new information is present, return "SKIP".

        Response format: ACTION|TOPIC|CONTENT|ID (ID only for Update/Delete)
        Example: UPDATE|formatting|Use uppercase SQL keywords|123-abc
        Example: SAVE|db_pref|User is working with a production schema|None
        """

        res = model.invoke(extraction_prompt).content
        if res == "SKIP":
            return state

        # Parse and execute
        parts = res.split("|")
        if len(parts) < 3: return state
        
        action, topic, content = parts[0], parts[1], parts[2]
        mem_id = parts[3] if len(parts) > 3 and parts[3] != "None" else str(uuid.uuid4())

        if action in ["SAVE", "UPDATE"]:
            store.put(namespace, mem_id, {"content": content, "topic": topic})
        elif action == "DELETE":
            store.delete(namespace, mem_id)

        save_to_disk(store, user_id)
        return state

    @tool
    def run_query_with_interrupt(config: RunnableConfig, query: str):
        """Execute SQL only after human approval."""
        confirmation = interrupt({"query": query, "message": "Review SQL."})
        if confirmation["type"] == "accept":
            return sql_run_tool.invoke({"query": query}, config)
        elif confirmation["type"] == "edit":
            return sql_run_tool.invoke({"query": confirmation["query"]}, config)
        return "Rejected."
    

    # --- Graph Nodes ---

    def list_tables_node(state: MessagesState):
        tables = list_tables_tool.invoke({})
        return {"messages": [AIMessage(content=f"Connected. Tables: {tables}")]}

    def schema_node(state: MessagesState):
        return {"messages": [model.bind_tools([get_schema_tool]).invoke(state["messages"])]}

    def generate_query_node(state: MessagesState, config: RunnableConfig, store: BaseStore):
        user_id = config["configurable"].get("user_id", "default_user")
        memories = store.search(("memories", user_id))
        
        memory_str = "\n".join(
            [f"- {m.value['topic'].upper()}: {m.value['content']}" for m in memories]
        ) if memories else "None"

        system_msg = SystemMessage(content=f"""
        SQL expert for {db.dialect}.
        USER MEMORY: {memory_str}
        1. Tailor queries based on memory.
        2. Max results: 5. SELECT only.
        """)

        # llm_with_tools = model.bind_tools([run_query_with_interrupt, upsert_user_memory,clear_all_memories])
        llm_with_tools = model.bind_tools([
                        run_query_with_interrupt
                    ])

        return {"messages": [llm_with_tools.invoke([system_msg] + state["messages"])]}

    def answer_node(state: MessagesState):
        system_msg = SystemMessage(content="Explain results clearly.")
        return {"messages": [model.invoke([system_msg] + state["messages"])]}

    # --- Graph Construction ---
    builder = StateGraph(MessagesState)

    builder.add_node("list_tables", list_tables_node)
    builder.add_node("schema_node", schema_node)
    builder.add_node("get_schema_tool_node", ToolNode([get_schema_tool]))
    builder.add_node("generate_query", generate_query_node)
    builder.add_node("run_query", ToolNode([run_query_with_interrupt]))
    # builder.add_node("memory_node", ToolNode([upsert_user_memory,clear_all_memories]))
    builder.add_node("answer", answer_node)
    builder.add_node("automatic_memory", automatic_memory_node) # NEW NODE

    builder.add_edge(START, "list_tables")
    builder.add_edge("list_tables", "schema_node")
    builder.add_edge("schema_node", "get_schema_tool_node")
    builder.add_edge("get_schema_tool_node", "generate_query")

    # def route_next(state: MessagesState):
    #     last = state["messages"][-1]
    #     if not last.tool_calls: return "answer"
    #     t_name = last.tool_calls[0]["name"]
    #     if t_name == "run_query_with_interrupt": return "run_query"
    #     # if t_name == "upsert_user_memory": return "memory_node"
    #     if t_name in ["upsert_user_memory", "clear_all_memories"]: 
    #         return "memory_node"
    #     return "get_schema_tool_node"
    def route_next(state: MessagesState):
        last = state["messages"][-1]

        if not last.tool_calls:
            return "answer"

        tool_name = last.tool_calls[0]["name"]

        if tool_name in ["upsert_user_memory", "clear_all_memories"]:
            return "memory_node"

        if tool_name == "run_query_with_interrupt":
            return "run_query"

        return "get_schema_tool_node"


    builder.add_conditional_edges("generate_query", route_next)
    
    # Pathing for memory and completion
    # builder.add_edge("memory_node", "answer") 
    builder.add_edge("run_query", "answer")
    builder.add_edge("answer", "automatic_memory") # Answer flows to memory extraction
    builder.add_edge("automatic_memory", END)

    # Initialize Long-Term Store
    long_term_store = InMemoryStore()
    
    # OPTIONAL: Seed the store from local_memories.json on startup
    if os.path.exists(MEMORY_FILE):
        try:
            with open(MEMORY_FILE, "r") as f:
                saved_data = json.load(f)
                for u_id, items in saved_data.items():
                    for k, v in items.items():
                        long_term_store.put(("memories", u_id), k, v)
        except: pass

    return builder.compile(checkpointer=InMemorySaver(), store=long_term_store)