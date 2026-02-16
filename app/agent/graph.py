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
from app.agent.tools import list_tables_tool, get_schema_tool, db_query_tool, get_column_values_tool, validate_clips
from app.core.config import settings
from app.utils.logger import setup_logging

logger = setup_logging("agent_graph")

# --- Initialize Components ---

# 1. Tools
tools = [list_tables_tool, get_schema_tool, get_column_values_tool, db_query_tool]
tool_node = ToolNode(tools)

# 2. LLM (Agent)
# Using vLLM as OpenAI Compatible Endpoint
llm = ChatOpenAI(
    model="Qwen/Qwen3-VL-8B-Instruct-FP8", # Or the model name served by vLLM
    openai_api_base=settings.VLLM_BASE_URL,
    openai_api_key="token-is-ignored",
    temperature=0
)

# Bind tools to LLM
llm_with_tools = llm.bind_tools(tools)

# 3. Planner
planner = NarrativePlanner()
memory = MemorySaver()

# --- System Prompt ---
SYSTEM_PROMPT = """You are an expert Video Editor Agent with direct access to a SQL database of video clips.
Your goal is to find the best video clips matching the user's request and then pass them to the Planner to create an edit.

**Your Workflow:**
1.  **Analyze Request**: Understand what the user wants (e.g., "b-roll of nature").
2.  **Inspect Schema**: Use `list_tables_tool` and `get_schema_tool` to understand the database.
    *   Table `clips` usually contains `visual_description`, `category`, `shot_type`, etc.
3.  **Explore Data**: Use `get_column_values_tool` if you need to know valid categories (e.g., is it 'A-Roll' or 'a_roll'?).
4.  **Query**: Write and execute a SQL query using `db_query_tool`.
    *   Use `ILIKE` for case-insensitive matching.
    *   Examples: `SELECT * FROM clips WHERE visual_description ILIKE '%sunset%'`
5.  **Refine**: If the query fails or returns 0 results, correct your SQL and try again.
6.  **Finish**: When you have found relevant clips, STOP calling tools. Just respond with a text summary like "I found X clips."

**Important:**
- Do not make up clip names.
- Always verify your query results.
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


    # Prepend System Prompt if not present (or as a separate message manipulation)
    input_messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages
    
    response = llm_with_tools.invoke(input_messages)
    
    return {"messages": [response]}


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

def router(state: FilmState) -> Literal["tools", "parser"]:
    messages = state["messages"]
    last_message = messages[-1]
    
    # If the LLM making a tool call?
    if last_message.tool_calls:
        return "tools"
    
    # If no tool call, it means the agent is done searching (or gave up).
    # Move to parsing results -> planner
    return "parser"

# --- Graph Definition ---

workflow = StateGraph(FilmState)

workflow.add_node("agent", agent_node)
workflow.add_node("tools", tool_node)
workflow.add_node("parser", result_parser_node)
workflow.add_node("planner", planner_node)

workflow.add_edge(START, "agent")

workflow.add_conditional_edges("agent", router)
workflow.add_edge("tools", "agent")
workflow.add_edge("parser", "planner")
workflow.add_edge("planner", END)

app = workflow.compile(checkpointer=memory)
