# from dotenv import load_dotenv

# # Load .env variables before anything else
# load_dotenv()

# import streamlit as st
# import uuid
# from langchain_core.messages import HumanMessage, AIMessage
# from langgraph.types import Command

# # Import agent factories
# from sql_agent import build_sql_agent
# from supervisor import app as supervisor_graph 

# # -----------------------------
# # Streamlit Configuration
# # -----------------------------
# st.set_page_config(page_title="Agentic Workspace", layout="wide")

# # Initialize Session States
# if "user_id" not in st.session_state:
#     st.session_state.user_id = "default_user_123" # Or str(uuid.uuid4())
# if "chat_history" not in st.session_state:
#     st.session_state.chat_history = []
# if "thread_id" not in st.session_state:
#     st.session_state.thread_id = str(uuid.uuid4())
# if "sql_connected" not in st.session_state:
#     st.session_state.sql_connected = False
# if "db_url" not in st.session_state:
#     st.session_state.db_url = ""
# if "sql_graph" not in st.session_state:
#     st.session_state.sql_graph = None

# # -----------------------------
# # Sidebar
# # -----------------------------
# # with st.sidebar:
# #     st.title("Settings")
# #     agent_choice = st.radio("Select Agent Mode", ["SQL Agent", "Supervisor Agent"])
# #     st.divider()
# #     if agent_choice == "SQL Agent":
# #         st.subheader("Database Configuration")
# #         db_url_input = st.text_input(
# #             "Database Connection String",
# #             value=st.session_state.db_url or "sqlite:///Chinook.db"
# #         )

# #         if st.button("Connect"):
# #             try:
# #                 # Re-build the agent with the new URL
# #                 st.session_state.sql_graph = build_sql_agent(db_url_input)
# #                 st.session_state.db_url = db_url_input
# #                 st.session_state.sql_connected = True
# #                 st.success("Database connected!")
# #                 st.rerun()
# #             except Exception as e:
# #                 st.error(f"Connection failed: {e}")

# # -----------------------------
# # Sidebar
# # -----------------------------
# with st.sidebar:
#     st.title("Settings")
#     agent_choice = st.radio("Select Agent Mode", ["SQL Agent", "Supervisor Agent"])
#     st.divider()

#     # 1. User Identity (Crucial for Long-Term Memory)
#     st.subheader("User Profile")
#     user_id = st.text_input("User ID", value=st.session_state.get("user_id", "default_user"))
#     if user_id != st.session_state.get("user_id"):
#         st.session_state.user_id = user_id
#         # We don't necessarily need to rerun, but it ensures config is updated
#         st.rerun()

#     st.divider()

#     if agent_choice == "SQL Agent":
#         st.subheader("Database Configuration")
#         db_url_input = st.text_input(
#             "Database Connection String",
#             value=st.session_state.db_url or "sqlite:///Chinook.db"
#         )

#         if st.button("Connect"):
#             try:
#                 # build_sql_agent now returns a compiled graph with an internal Store
#                 st.session_state.sql_graph = build_sql_agent(db_url_input)
#                 st.session_state.db_url = db_url_input
#                 st.session_state.sql_connected = True
#                 st.success("Database connected!")
#                 st.rerun()
#             except Exception as e:
#                 st.error(f"Connection failed: {e}")

#         # 2. Memory Visualization
#         if st.session_state.sql_connected and st.session_state.sql_graph:
#             st.divider()
#             st.subheader("🧠 Long-Term Memory")
            
#             # Access the store from the compiled graph
#             store = st.session_state.sql_graph.store
#             # Search for memories in the specific user's namespace
#             memories = store.search(("memories", st.session_state.user_id))
            
#             if memories:
#                 for m in memories:
#                     with st.expander(f"💡 {m.value['preference'][:30]}..."):
#                         st.write(f"**Preference:** {m.value['preference']}")
#                         st.caption(f"**Context:** {m.value['context']}")
                
#                 if st.button("🗑️ Clear All Memories"):
#                     # Delete each memory in the namespace
#                     for m in memories:
#                         store.delete(("memories", st.session_state.user_id), m.key)
#                     st.toast("Memories cleared!")
#                     st.rerun()
#             else:
#                 st.info("The agent hasn't saved any user preferences yet.")
    


# # -----------------------------
# # Chat UI Rendering
# # -----------------------------
# st.title(f"{agent_choice}")

# # Display historical messages from session state
# for msg in st.session_state.chat_history:
#     with st.chat_message(msg["role"]):
#         st.markdown(msg["content"])

# config = {"configurable": 
#           {"thread_id": st.session_state.thread_id,
#             "user_id": st.session_state.user_id
#             }
#         }

# # -----------------------------
# # Chat Input & Processing
# # -----------------------------
# user_input = st.chat_input("Enter your request")

# if user_input:
#     # 1. Store and display user message
#     st.session_state.chat_history.append({"role": "user", "content": user_input})
#     with st.chat_message("user"):
#         st.markdown(user_input)

#     # 2. Generate Assistant Response
#     with st.chat_message("assistant"):
#         if agent_choice == "SQL Agent":
#             if not st.session_state.sql_connected or st.session_state.sql_graph is None:
#                 st.error("Please connect a database first.")
#             else:
#                 final_response = ""
#                 with st.status("Querying Database...", expanded=True) as status:
#                     # stream_mode="values" gives us the list of messages at each step
#                     for event in st.session_state.sql_graph.stream(
#                         {"messages": [HumanMessage(content=user_input)]},
#                         config,
#                         stream_mode="values",
#                     ):
#                         last_msg = event["messages"][-1]
#                         if isinstance(last_msg, AIMessage):
#                             # if last_msg.tool_calls:
#                             #     status.write(f"🔧 Tool Call: {last_msg.tool_calls[0]['name']}")
#                             if last_msg.tool_calls:
#                                 t_name = last_msg.tool_calls[0]['name']
#                                 # Add specific feedback for memory saving
#                                 if t_name == "save_user_preference":
#                                     status.write(f"🧠 Learning preference: {last_msg.tool_calls[0]['args'].get('preference')}")
#                                 else:
#                                     status.write(f"🔧 Tool Call: {t_name}")
                            
#                             else:
#                                 final_response = last_msg.content
                    
#                     status.update(label="Complete", state="complete", expanded=False)
                
#                 # Display and Save the final response
#                 if final_response:
#                     st.markdown(final_response)
#                     st.session_state.chat_history.append({"role": "assistant", "content": final_response})
                
#                 # Check for interrupts immediately after streaming
#                 st.rerun()

#         else:
#             # Supervisor Agent (Functional Implementation)
#             if supervisor_graph is None:
#                 st.error("Supervisor Agent graph is not imported or defined.")
#             else:
#                 with st.status("Supervisor routing task...", expanded=True) as status:
#                     final_response = ""
#                     # We use streaming to see the "thought process" of the supervisor
#                     for event in supervisor_graph.stream(
#                         {"messages": [HumanMessage(content=user_input)]},
#                         config,
#                         stream_mode="values",
#                     ):
#                         last_msg = event["messages"][-1]
#                         if isinstance(last_msg, AIMessage):
#                             if last_msg.tool_calls:
#                                 # Show which sub-agent the supervisor is calling
#                                 status.write(f"🤝 Delegating to: {last_msg.tool_calls[0]['name']}")
#                             else:
#                                 final_response = last_msg.content
                    
#                     status.update(label="Task Complete", state="complete", expanded=False)

#                 if final_response:
#                     st.markdown(final_response)
#                     st.session_state.chat_history.append({"role": "assistant", "content": final_response})

# # -----------------------------
# # Human-in-the-Loop (SQL)
# # -----------------------------
# if agent_choice == "SQL Agent" and st.session_state.sql_connected and st.session_state.sql_graph:
#     state = st.session_state.sql_graph.get_state(config)

#     # If the graph is waiting for human input (interrupt)
#     if state.next and state.tasks and state.tasks[0].interrupts:
#         interrupt_data = state.tasks[0].interrupts[0].value
        
#         st.divider()
#         st.subheader("🛠️ SQL Review Required")
#         st.info("The agent needs your approval to run this query:")
#         st.code(interrupt_data["query"], language="sql")

#         edited_query = st.text_area("Edit query (optional):", value=interrupt_data["query"])
#         col1, col2, col3 = st.columns(3)

#         # ---------------------------------------------------------
#         # Helper to process the Command and update history
#         # ---------------------------------------------------------
#         def process_approval(command):
#             final_response = ""
#             # 1. Resume the graph with the specific command (accept/edit/reject)
#             for event in st.session_state.sql_graph.stream(command, config, stream_mode="values"):
#                 last_msg = event["messages"][-1]
#                 if isinstance(last_msg, AIMessage) and not last_msg.tool_calls:
#                     final_response = last_msg.content
            
#             # 2. Save the final response (even if it's a rejection message) to history
#             if final_response:
#                 st.session_state.chat_history.append({"role": "assistant", "content": final_response})
            
#             # 3. Force rerun to clear the interrupt UI and show the answer in chat
#             st.rerun()

#         with col1:
#             if st.button("✅ Approve & Run", use_container_width=True):
#                 process_approval(Command(resume={"type": "accept"}))
            
#         with col2:
#             if st.button("📝 Run Edited Query", use_container_width=True):
#                 process_approval(Command(resume={"type": "edit", "query": edited_query}))
            
#         with col3:
#             if st.button("❌ Reject", use_container_width=True):
#                 # This moves the graph to the 'answer' node with a 'rejected' status
#                 process_approval(Command(resume={"type": "reject"}))


from dotenv import load_dotenv

# Load .env variables before anything else
load_dotenv()

import streamlit as st
import uuid
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.types import Command

# Import agent factories
from sql_agent import build_sql_agent
from supervisor import app as supervisor_graph 

# -----------------------------
# Streamlit Configuration
# -----------------------------
st.set_page_config(page_title="Agentic Workspace", layout="wide")

# Initialize Session States
if "user_id" not in st.session_state:
    st.session_state.user_id = "default_user_123"
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "sql_connected" not in st.session_state:
    st.session_state.sql_connected = False
if "db_url" not in st.session_state:
    st.session_state.db_url = ""
if "sql_graph" not in st.session_state:
    st.session_state.sql_graph = None

# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.title("Settings")
    agent_choice = st.radio("Select Agent Mode", ["SQL Agent", "Supervisor Agent"])
    st.divider()

    # User Identity
    st.subheader("User Profile")
    user_id = st.text_input("User ID", value=st.session_state.user_id)
    if user_id != st.session_state.user_id:
        st.session_state.user_id = user_id
        st.rerun()

    st.divider()

    if agent_choice == "SQL Agent":
        st.subheader("Database Configuration")
        db_url_input = st.text_input(
            "Database Connection String",
            value=st.session_state.db_url or "sqlite:///Chinook.db"
        )

        if st.button("Connect"):
            try:
                st.session_state.sql_graph = build_sql_agent(db_url_input)
                st.session_state.db_url = db_url_input
                st.session_state.sql_connected = True
                st.success("Database connected!")
                st.rerun()
            except Exception as e:
                st.error(f"Connection failed: {e}")

        # --- UPDATED: Memory Visualization for Upsert Logic ---
        if st.session_state.sql_connected and st.session_state.sql_graph:
            st.divider()
            st.subheader("🧠 Long-Term Memory")
            
            store = st.session_state.sql_graph.store
            memories = store.search(("memories", st.session_state.user_id))
            
            if memories:
                # Group/Sort by topic
                for m in memories:
                    topic = m.value.get('topic', 'General').upper()
                    content = m.value.get('content', 'No content')
                    
                    # with st.expander(f"📌 {topic}"):
                    #     st.write(content)
                    #     st.caption(f"Key: {m.key}") # Helpful for debugging UUIDs
                    with st.expander(f"📌 {topic}"):
                        col_text, col_del = st.columns([0.85, 0.15])
                        with col_text:
                            st.write(content)
                        with col_del:
                            # Use a unique key for each button based on m.key
                            if st.button("❌", key=f"del_{m.key}"):
                                store.delete(("memories", st.session_state.user_id), m.key)
                                st.rerun()
                
                if st.button("🗑️ Clear All Memories"):
                    for m in memories:
                        store.delete(("memories", st.session_state.user_id), m.key)
                    st.toast("Memories cleared!")
                    st.rerun()
            else:
                st.info("No learned preferences yet.")

# -----------------------------
# Chat UI Rendering
# -----------------------------
st.title(f"{agent_choice}")

for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

config = {
    "configurable": {
        "thread_id": st.session_state.thread_id,
        "user_id": st.session_state.user_id
    }
}

# -----------------------------
# Chat Input & Processing
# -----------------------------
user_input = st.chat_input("Enter your request")

if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        if agent_choice == "SQL Agent":
            if not st.session_state.sql_connected or st.session_state.sql_graph is None:
                st.error("Please connect a database first.")
            else:
                final_response = ""
                if memories:
                    st.caption(f"✨ Using {len(memories)} saved preferences to tailor results.")
                with st.status("Thinking...", expanded=True) as status:
                    for event in st.session_state.sql_graph.stream(
                        {"messages": [HumanMessage(content=user_input)]},
                        config,
                        stream_mode="values",
                    ):
                        last_msg = event["messages"][-1]
                        if "automatic_memory" in event:
                            st.toast("💡 Memory updated based on our conversation!", icon="🧠")
                        if isinstance(last_msg, AIMessage):
                            if last_msg.tool_calls:
                                t_name = last_msg.tool_calls[0]['name']
                                
                                # --- UPDATED: Match the new Tool Name ---
                                if t_name == "upsert_user_memory":
                                    topic = last_msg.tool_calls[0]['args'].get('topic')
                                    status.write(f"🧠 Updating memory for: **{topic}**")
                                elif t_name == "clear_all_memories":
                                    status.write("🗑️ Clearing all long-term memories...")
                                else:
                                    status.write(f"🔧 Tool Call: {t_name}")
                            else:
                                final_response = last_msg.content
                    
                    status.update(label="Complete", state="complete", expanded=False)
                
                if final_response:
                    st.markdown(final_response)
                    st.session_state.chat_history.append({"role": "assistant", "content": final_response})
                
                    st.rerun()
        else:
            # Supervisor Agent logic remains same...
            pass

# -----------------------------
# Human-in-the-Loop (SQL)
# -----------------------------
if agent_choice == "SQL Agent" and st.session_state.sql_connected and st.session_state.sql_graph:
    state = st.session_state.sql_graph.get_state(config)

    # If the graph is waiting for human input (interrupt)
    if state.next and state.tasks and state.tasks[0].interrupts:
        interrupt_data = state.tasks[0].interrupts[0].value
        
        st.divider()
        st.subheader("🛠️ SQL Review Required")
        st.info("The agent needs your approval to run this query:")
        st.code(interrupt_data["query"], language="sql")

        edited_query = st.text_area("Edit query (optional):", value=interrupt_data["query"])
        col1, col2, col3 = st.columns(3)

        # ---------------------------------------------------------
        # Helper to process the Command and update history
        # ---------------------------------------------------------
        def process_approval(command):
            final_response = ""
            # 1. Resume the graph with the specific command (accept/edit/reject)
            for event in st.session_state.sql_graph.stream(command, config, stream_mode="values"):
                last_msg = event["messages"][-1]
                if isinstance(last_msg, AIMessage) and not last_msg.tool_calls:
                    final_response = last_msg.content
            
            # 2. Save the final response (even if it's a rejection message) to history
            if final_response:
                st.session_state.chat_history.append({"role": "assistant", "content": final_response})
            
            # 3. Force rerun to clear the interrupt UI and show the answer in chat
            st.rerun()

        with col1:
            if st.button("✅ Approve & Run", use_container_width=True):
                process_approval(Command(resume={"type": "accept"}))
            
        with col2:
            if st.button("📝 Run Edited Query", use_container_width=True):
                process_approval(Command(resume={"type": "edit", "query": edited_query}))
            
        with col3:
            if st.button("❌ Reject", use_container_width=True):
                # This moves the graph to the 'answer' node with a 'rejected' status
                process_approval(Command(resume={"type": "reject"}))

