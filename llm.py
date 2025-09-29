import os
from typing import Literal, Tuple

import dotenv
from langchain.chat_models import init_chat_model
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from langchain_core.messages import AIMessage
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

dotenv.load_dotenv()


def build_agent_for_db_path(db_path: str, model: str = "groq:meta-llama/llama-4-scout-17b-16e-instruct") -> Tuple[object, SQLDatabase]:
    """Build a SQL agent for the given database path."""
    llm = init_chat_model(model, api_key=os.getenv("GROQ_API_KEY"))
    db = SQLDatabase.from_uri(f"sqlite:///{db_path}")
    
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    tools = toolkit.get_tools()
    
    # Get required tools
    get_schema_tool = next(t for t in tools if t.name == "sql_db_schema")
    run_query_tool = next(t for t in tools if t.name == "sql_db_query")
    list_tables_tool = next(t for t in tools if t.name == "sql_db_list_tables")
    
    # Create tool nodes
    get_schema_node = ToolNode([get_schema_tool], name="get_schema")
    run_query_node = ToolNode([run_query_tool], name="run_query")
    
    def list_tables(state: MessagesState):
        """List all available tables in the database."""
        tool_call = {"name": "sql_db_list_tables", "args": {}, "id": "list_tables", "type": "tool_call"}
        tool_call_message = AIMessage(content="", tool_calls=[tool_call])
        tool_message = list_tables_tool.invoke(tool_call)
        response = AIMessage(f"Available tables: {tool_message.content}")
        return {"messages": [tool_call_message, tool_message, response]}

    def call_get_schema(state: MessagesState):
        """Force model to get schema for relevant tables."""
        llm_with_tools = llm.bind_tools([get_schema_tool], tool_choice="any")
        response = llm_with_tools.invoke(state["messages"])
        return {"messages": [response]}

    def generate_query(state: MessagesState):
        """Generate SQL query based on user question."""
        system_prompt = f"""You are an agent designed to interact with a SQL database.
Given an input question, create a syntactically correct {db.dialect} query to run,
then look at the results of the query and return the answer. Unless the user
specifies a specific number of examples they wish to obtain, always limit your
query to at most 5 results.

You can order the results by a relevant column to return the most interesting
examples in the database. Never query for all the columns from a specific table,
only ask for the relevant columns given the question.

DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database."""
        
        system_message = {"role": "system", "content": system_prompt}
        llm_with_tools = llm.bind_tools([run_query_tool])
        response = llm_with_tools.invoke([system_message] + state["messages"])
        return {"messages": [response]}

    def check_query(state: MessagesState):
        """Double-check the generated query for common mistakes."""
        system_prompt = f"""You are a SQL expert with a strong attention to detail.
Double check the {db.dialect} query for common mistakes, including:
- Using NOT IN with NULL values
- Using UNION when UNION ALL should have been used
- Using BETWEEN for exclusive ranges
- Data type mismatch in predicates
- Properly quoting identifiers
- Using the correct number of arguments for functions
- Casting to the correct data type
- Using the proper columns for joins

If there are any of the above mistakes, rewrite the query. If there are no mistakes,
just reproduce the original query.

You will call the appropriate tool to execute the query after running this check."""
        
        system_message = {"role": "system", "content": system_prompt}
        tool_call = state["messages"][-1].tool_calls[0]
        user_message = {"role": "user", "content": tool_call["args"]["query"]}
        llm_with_tools = llm.bind_tools([run_query_tool], tool_choice="any")
        response = llm_with_tools.invoke([system_message, user_message])
        response.id = state["messages"][-1].id
        return {"messages": [response]}

    def should_continue(state: MessagesState) -> Literal[END, "check_query"]:
        """Decide whether to continue to query checking or end."""
        last_message = state["messages"][-1]
        return "check_query" if getattr(last_message, "tool_calls", None) else END

    # Build the graph
    builder = StateGraph(MessagesState)
    
    # Add nodes
    nodes = [
        ("list_tables", list_tables),
        ("call_get_schema", call_get_schema),
        ("get_schema", get_schema_node),
        ("generate_query", generate_query),
        ("check_query", check_query),
        ("run_query", run_query_node),
    ]
    for name, func in nodes:
        builder.add_node(name, func)
    
    # Add edges
    edges = [
        (START, "list_tables"),
        ("list_tables", "call_get_schema"),
        ("call_get_schema", "get_schema"),
        ("get_schema", "generate_query"),
        ("check_query", "run_query"),
        ("run_query", "generate_query"),
    ]
    for source, target in edges:
        builder.add_edge(source, target)
    
    builder.add_conditional_edges("generate_query", should_continue)
    
    return builder.compile(), db


if __name__ == "__main__":
    # Minimal CLI demo (optional): python llm.py <db_path> "<question>"
    import sys
    if len(sys.argv) >= 3:
        db_path = sys.argv[1]
        question = " ".join(sys.argv[2:])
        agent, _ = build_agent_for_db_path(db_path)
        for step in agent.stream({"messages": [{"role": "user", "content": question}]}, stream_mode="values"):
            print(step["messages"][-1])
    else:
        print("Usage: python llm.py <db_path> '<question>'")