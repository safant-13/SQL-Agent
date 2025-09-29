import os
import tempfile
import streamlit as st
from llm import build_agent_for_db_path


st.set_page_config(page_title="SQL Agent POC", page_icon="🗄️")
st.title("SQL Agent POC")
st.caption("Upload a SQLite .db file and ask questions in natural language.")


@st.cache_resource(show_spinner=False)
def get_agent(db_path: str):
    agent, db = build_agent_for_db_path(db_path)
    return agent, db


uploaded = st.file_uploader("Upload SQLite .db file", type=["db", "sqlite", "sqlite3"], accept_multiple_files=False)

db_path = None
if uploaded is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".db") as tmp:
        tmp.write(uploaded.getbuffer())
        db_path = tmp.name
        st.success(f"Uploaded database saved to: {db_path}")

example_path = os.path.join(os.getcwd(), "Chinook.db")
if db_path is None and os.path.exists(example_path):
    st.info("No file uploaded. Using bundled Chinook.db for demo.")
    db_path = example_path

if db_path:
    try:
        agent, db = get_agent(db_path)
        st.write(f"Dialect: {db.dialect}")
        with st.expander("Tables", expanded=False):
            st.write(sorted(list(db.get_usable_table_names())))

        question = st.text_input("Ask a question about your data", value="Which genre on average has the longest tracks?")
        run = st.button("Run")

        if run and question.strip():
            st.subheader("Agent Execution (Live)")
            
            # Create containers for live updates
            status_container = st.container()
            steps_container = st.container()
            answer_container = st.container()
            
            with status_container:
                status = st.empty()
                
            with steps_container:
                step_placeholder = st.empty()
                
            messages = []
            step_count = 0
            
            try:
                for step in agent.stream({"messages": [{"role": "user", "content": question}]}, stream_mode="values"):
                    step_count += 1
                    last = step["messages"][-1]
                    messages.append(last)
                    
                    # Update status
                    status.info(f"Step {step_count}: Processing...")
                    
                    # Show current step live
                    with step_placeholder.container():
                        st.markdown(f"### Step {step_count}")
                        
                        if hasattr(last, 'tool_calls') and last.tool_calls:
                            tool_call = last.tool_calls[0]
                            st.markdown(f"**🔧 Tool Call:** `{tool_call['name']}`")
                            if tool_call.get('args'):
                                st.json(tool_call['args'])
                        elif last.content:
                            st.markdown(f"**💬 Response:** {last.content}")
                        else:
                            st.markdown(f"**📝 Message Type:** {type(last).__name__}")
                            
                        st.markdown("---")
                    
                    # Small delay to make it visible
                    import time
                    time.sleep(0.5)
                
                # Final status and answer
                status.success(f"✅ Completed in {step_count} steps")
                
                if messages and messages[-1].content:
                    with answer_container:
                        st.subheader("🎯 Final Answer")
                        st.success(messages[-1].content)
                        
            except Exception as e:
                status.error(f"❌ Error: {str(e)}")
                st.error(f"Agent execution failed: {e}")

            # Optional: Show full trace in expander
            if messages:
                with st.expander("🔍 Full Debug Trace", expanded=False):
                    for i, m in enumerate(messages, start=1):
                        st.markdown(f"**Step {i}:** {type(m).__name__}")
                        st.code(str(m))
    except Exception as e:
        st.error(f"Failed to initialize agent: {e}")
else:
    st.warning("Upload a database file to begin, or place `Chinook.db` in the project root.")


