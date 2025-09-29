import os
import tempfile
from datetime import datetime

import streamlit as st

from llm import build_agent_for_db_path

# Configure page
st.set_page_config(page_title="SQL Chat Agent", page_icon="💬", layout="wide")

# Initialize session state
def init_session_state():
    """Initialize session state variables."""
    defaults = {
        "messages": [],
        "db_path": None,
        "agent": None,
        "db": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

@st.cache_resource(show_spinner=False)
def get_agent(db_path: str):
    """Get cached agent and database for the given path."""
    return build_agent_for_db_path(db_path)

def setup_database():
    """Handle database upload and initialization."""
    uploaded = st.file_uploader("Upload SQLite .db file", type=["db", "sqlite", "sqlite3"])
    
    if uploaded is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".db") as tmp:
            tmp.write(uploaded.getbuffer())
            if st.session_state.db_path != tmp.name:
                st.session_state.db_path = tmp.name
                st.session_state.agent = None
                st.session_state.messages = []
                st.success("Database uploaded!")
    
    # Use default database if available
    if st.session_state.db_path is None:
        default_path = os.path.join(os.getcwd(), "Chinook.db")
        if os.path.exists(default_path):
            st.session_state.db_path = default_path
            st.info("Using bundled Chinook.db")
    
    # Initialize agent
    if st.session_state.db_path and st.session_state.agent is None:
        try:
            with st.spinner("Loading database..."):
                st.session_state.agent, st.session_state.db = get_agent(st.session_state.db_path)
            st.success("Database loaded!")
        except Exception as e:
            st.error(f"Failed to load database: {e}")

def display_database_info():
    """Display database information in sidebar."""
    if st.session_state.db:
        st.subheader("📊 Database Info")
        st.write(f"**Dialect:** {st.session_state.db.dialect}")
        with st.expander("Available Tables", expanded=False):
            tables = sorted(st.session_state.db.get_usable_table_names())
            for table in tables:
                st.write(f"• {table}")

def parse_step_info(message, step_count):
    """Parse step information from agent message."""
    step_info = {"step": step_count}
    
    if hasattr(message, 'tool_calls') and message.tool_calls:
        tool_call = message.tool_calls[0]
        step_info["tool_call"] = {
            "name": tool_call['name'],
            "args": tool_call.get('args', {})
        }
    elif message.content:
        step_info["content"] = message.content
    else:
        step_info["type"] = type(message).__name__
    
    return step_info

def display_execution_steps(steps):
    """Display execution steps in an expandable section."""
    with st.expander(f"🔍 View execution steps ({len(steps)} steps)", expanded=False):
        for i, step in enumerate(steps, 1):
            st.markdown(f"**Step {i}:**")
            
            if step.get("tool_call"):
                tool_call = step["tool_call"]
                st.markdown(f"🔧 **Tool:** `{tool_call['name']}`")
                if tool_call.get('args'):
                    st.json(tool_call['args'])
            elif step.get("content"):
                st.markdown(f"💬 **Response:** {step['content']}")
            else:
                st.markdown(f"📝 **Type:** {step.get('type', 'Unknown')}")
            
            if i < len(steps):
                st.markdown("---")

# Initialize app
init_session_state()

# Sidebar
with st.sidebar:
    st.header("🗄️ Database Setup")
    setup_database()
    display_database_info()
    
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# Main chat interface
st.title("💬 SQL Chat Agent")
st.caption("Ask questions about your database in natural language")

if not st.session_state.agent:
    st.warning("Please upload a database file or place 'Chinook.db' in the project root to start chatting.")
else:
    # Display chat history
    for message in st.session_state.messages:
        if message["role"] == "user":
            with st.chat_message("user"):
                st.write(message["content"])
        else:
            with st.chat_message("assistant"):
                st.write(message["content"])
                if "steps" in message:
                    display_execution_steps(message["steps"])

    # Chat input
    if prompt := st.chat_input("Ask a question about your database..."):
        # Add and display user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)
        
        # Generate assistant response
        with st.chat_message("assistant"):
            thinking_placeholder = st.empty()
            response_placeholder = st.empty()
            
            thinking_placeholder.info("🤔 Thinking...")
            
            try:
                messages = []
                steps = []
                step_count = 0
                
                # Stream agent execution
                for step in st.session_state.agent.stream(
                    {"messages": [{"role": "user", "content": prompt}]}, 
                    stream_mode="values"
                ):
                    step_count += 1
                    last_message = step["messages"][-1]
                    messages.append(last_message)
                    steps.append(parse_step_info(last_message, step_count))
                    thinking_placeholder.info(f"🔄 Step {step_count}: Processing...")
                
                thinking_placeholder.empty()
                
                if messages and messages[-1].content:
                    final_answer = messages[-1].content
                    response_placeholder.success(final_answer)
                    
                    # Add to chat history
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": final_answer,
                        "steps": steps,
                        "timestamp": datetime.now()
                    })
                else:
                    response_placeholder.error("No response generated")
                    
            except Exception as e:
                thinking_placeholder.empty()
                response_placeholder.error(f"Error: {str(e)}")
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": f"Sorry, I encountered an error: {str(e)}",
                    "steps": [],
                    "timestamp": datetime.now()
                })


