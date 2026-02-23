from typing import List, Literal, Annotated
import json
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.runnables import RunnableConfig

from app.agent.state import FilmState
from app.agent.planner import NarrativePlanner
from app.agent.tools import list_tables_tool, get_schema_tool, db_query_tool, get_column_values_tool, validate_clips, trigger_video_search_tool
from app.core.config import settings
from app.utils.logger import setup_logging

logger = setup_logging("agent_graph")

# --- Initialize Components ---

# 1. Tools
tools = [list_tables_tool, get_schema_tool, get_column_values_tool, db_query_tool]
tool_node = ToolNode(tools)
supervisor_tool_node = ToolNode(tools + [trigger_video_search_tool])

# 2. LLM (Agent)
# Using vLLM as OpenAI Compatible Endpoint
llm = ChatOpenAI(
    model="Qwen/Qwen3-VL-8B-Instruct-FP8", # Or the model name served by vLLM
    openai_api_base=settings.VLLM_BASE_URL,
    openai_api_key="token-is-ignored",
    temperature=0
)

# Bind tools to LLM
llm_editor = llm.bind_tools(tools)
llm_supervisor = llm.bind_tools(tools + [trigger_video_search_tool])

# 3. Planner
planner = NarrativePlanner()
memory = MemorySaver()

# --- System Prompt ---
SYSTEM_PROMPT = """You are an expert Video Editor Agent with direct access to a SQL database of video clips.
Your goal is to find the best video clips matching the user's request and then pass them to the Planner to create an edit.

**Your Workflow:**
1.  **Analyze Request**: Understand what the user wants (e.g., "b-roll of nature").
2.  **Inspect Schema**: Use `list_tables_tool` and `get_schema_tool` to understand the database.
    *   Table `clips` usually contains `visual_description`, `category`, `shot_type`, `duration` (FLOAT), etc.
3.  **Explore Data**: Use `get_column_values_tool` if you need to know valid categories (e.g., is it 'A-Roll' or 'a_roll'?).
4.  **Query**: Write and execute a SQL query using `db_query_tool`.
    *   Use `ILIKE` for case-insensitive matching.
    *   Examples: `SELECT * FROM clips WHERE visual_description ILIKE '%sunset%'`
5.  **Refine**: If the query fails or returns 0 results, correct your SQL and try again.
6.  **Finish**: When you have found relevant clips, STOP calling tools. Just respond with a text summary like "I found X clips."

**Important:**
- **Do not LIMIT results** unless the user explicitly asks (e.g. "Find me 5 clips"). return all matches.
- Do not make up clip names.
- Always verify your query results.
"""

SUPERVISOR_PROMPT = """You are an Assistant Director. You brainstorm video narratives with the user.
Your goal is to help the user figure out what kind of edit they want to make.
You have access to the database to answer questions about available clips (using list_tables_tool, get_schema_tool, db_query_tool, etc.). You can count clips, find categories, and summarize what's available.
When the user is ready to create an edit and you have agreed on the direction, use the `trigger_video_search_tool`
passing specific instructions on what clips the Video Editor should find (e.g., 'Find clips with beaches and sunsets').
Do NOT use the trigger_video_search_tool if the user is just asking questions or brainstorming.
"""

# --- Nodes ---

def agent_node(state: FilmState):
    """
    ReAct Agent: Decides to call a tool or end the search.
    """
    messages = state["messages"]
    
    # --- Logging Context Separation ---
    # Only log at the start of a turn (when the last message is from the user)
    if messages and isinstance(messages[-1], HumanMessage):
        logger.info("\n" + "="*40 + "\n=== SYSTEM PROMPT ===\n" + SYSTEM_PROMPT)
        logger.info("\n=== USER INPUT ===\n" + str(messages[-1].content))


    # Prepend System Prompt
    # Truncate large tool outputs in the context to avoid hitting max tokens
    # We keep the original messages in 'state', this is just for the LLM view.
    filtered_messages = []
    for msg in messages:
        if isinstance(msg, ToolMessage) and len(str(msg.content)) > 2000:
            content_str = str(msg.content)
            new_content = None
            
            # Try parsing as JSON list (common for db_query_tool)
            try:
                data = json.loads(content_str)
                if isinstance(data, list) and len(data) > 0:
                    # Keep first item to show structure/schema
                    subset = data[:1]
                    # Add summary item
                    subset.append({
                        "summary": f"... {len(data) - 1} more items truncated to save context ...",
                        "note": "The full list is available to the Planner and Parser."
                    })
                    new_content = json.dumps(subset)
            except Exception:
                # Not JSON or failed to parse
                pass
            
            if not new_content:
                 # Fallback to string slicing
                new_content = content_str[:2000] + "... (truncated)"

            # Create a copy with truncated content
            # Note: We must preserve the ID and other attributes so the LLM knows which tool call it matches
            msg_copy = ToolMessage(
                content=new_content,
                tool_call_id=msg.tool_call_id,
                name=msg.name,
                artifact=msg.artifact,
                status=msg.status
            )
            filtered_messages.append(msg_copy)
        else:
            filtered_messages.append(msg)

    input_messages = [SystemMessage(content=SYSTEM_PROMPT)] + filtered_messages
    
    response = llm_editor.invoke(input_messages)
    
    return {"messages": [response], "active_agent": "editor"}

def supervisor_node(state: FilmState):
    """
    Supervisor Agent: Chats with user, brainstorms, and triggers Editor when ready.
    """
    messages = state["messages"]
    
    # --- Logging Context Separation ---
    # Only log at the start of a turn (when the last message is from the user)
    if messages and isinstance(messages[-1], HumanMessage):
        logger.info("\n" + "="*40 + "\n=== SUPERVISOR PROMPT ===\n" + SUPERVISOR_PROMPT)
        logger.info("\n=== USER INPUT ===\n" + str(messages[-1].content))

    input_messages = [SystemMessage(content=SUPERVISOR_PROMPT)] + messages
    
    response = llm_supervisor.invoke(input_messages)
    
    return {"messages": [response], "active_agent": "supervisor"}

def result_parser_node(state: FilmState):
    """
    Extracts the clips from the last successful SQL query in the conversation history
    and populates state["bin"].
    """
    messages = state["messages"]
    bin_clips = []
    
    # Iterate backwards to find the last db_query_tool output
    for msg in reversed(messages):
        if isinstance(msg, ToolMessage) and msg.name == "db_query_tool":
            try:
                # The tool returns a JSON string
                data = json.loads(msg.content)
                if isinstance(data, list):
                    bin_clips = data
                    logger.info(f"Parsed {len(bin_clips)} clips from history.")
                    break
            except Exception as e:
                logger.warning(f"Failed to parse db_query_tool output: {e}")
    
    if not bin_clips:
        logger.warning("No clips found in history to pass to planner.")
        
    return {"bin": bin_clips}


def planner_node(state: FilmState):
    """
    Generate Edit Plan using DSPy based on 'bin' results.
    """
    intent = state.get("user_intent", "")
    bin_clips = state.get("bin", [])
    
    # --- Logging Context Separation ---
    logger.info("\n" + "="*40 + "\n=== PLANNER INTENT ===\n" + str(intent))
    logger.info("\n=== PLANNER CONTEXT (Search Results) ===\n" + json.dumps(bin_clips, indent=2))
    
    logger.info(f"Generating Plan for {len(bin_clips)} clips.")
    
    if not bin_clips:
        return {
            "timeline": [],
            "messages": [AIMessage(content="I couldn't find any clips matching your request.")]
        }

    try:
        # Call DSPy module
        result = planner(user_intent=intent, search_results=bin_clips)
        
        plan_text = result.get("edit_plan", "")
        reasoning = result.get("reasoning", "")
        
        timeline = [{"description": plan_text}] 
        return {
            "timeline": timeline,
            # Pass reasoning in additional_kwargs for Streamlit UI
            "messages": [AIMessage(content=plan_text, additional_kwargs={"reasoning": reasoning})]
        }
        
    except Exception as e:
        logger.error(f"Planning Error: {e}")
        return {
            "timeline": [],
            "messages": [AIMessage(content=f"Error generating plan: {e}")]
        }

# --- Router ---

def editor_router(state: FilmState) -> Literal["tools", "parser"]:
    messages = state["messages"]
    last_message = messages[-1]
    
    # If the LLM making a tool call?
    if last_message.tool_calls:
        return "tools"
    
    # If no tool call, it means the agent is done searching (or gave up).
    # Move to parsing results -> planner
    return "parser"

def supervisor_router(state: FilmState) -> Literal["supervisor_tools", END]:
    messages = state["messages"]
    last_message = messages[-1]
    
    if last_message.tool_calls:
        return "supervisor_tools"
    return END

def supervisor_tools_router(state: FilmState) -> Literal["agent", "supervisor"]:
    messages = state["messages"]
    last_message = messages[-1]
    
    if isinstance(last_message, ToolMessage) and last_message.name == "trigger_video_search_tool":
        logger.info(f"Supervisor triggering Editor via tool call.")
        return "agent"
        
    return "supervisor"

# --- Graph Definition ---

workflow = StateGraph(FilmState)

workflow.add_node("supervisor", supervisor_node)
workflow.add_node("agent", agent_node)
workflow.add_node("tools", tool_node)
workflow.add_node("supervisor_tools", supervisor_tool_node)
workflow.add_node("parser", result_parser_node)
workflow.add_node("planner", planner_node)

workflow.add_edge(START, "supervisor")

workflow.add_conditional_edges("supervisor", supervisor_router)
workflow.add_conditional_edges("supervisor_tools", supervisor_tools_router)
workflow.add_conditional_edges("agent", editor_router)
workflow.add_edge("tools", "agent")
workflow.add_edge("parser", "planner")
workflow.add_edge("planner", END)

app = workflow.compile(checkpointer=memory)
