from typing import List
from langgraph.graph import StateGraph, START
from langgraph.checkpoint.memory import MemorySaver
from app.agent.state import FilmState
from app.agent.planner import NarrativePlanner
from app.agent.tools import sql_search, validate_clips
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from app.utils.logger import setup_logging

logger = setup_logging("agent_graph")

# Initialize Tools & Planner
planner = NarrativePlanner()
memory = MemorySaver()

# --- Nodes ---

def agent_node(state: FilmState):
    """
    Decides the next step based on the conversation history and current intent.
    Simple logic:
    - If user intent is new, Search.
    - If search results exist, Plan.
    - If plan exists, Validate.
    - If validated, respond to user.
    """
    messages = state["messages"]
    last_message = messages[-1]
    
    # Very basic routing logic for now. 
    # In a real agent, an LLM would decide this dynamically.
    bin_val = state.get("bin")
    # If bin is None, we haven't searched yet.
    if bin_val is None:
        logger.info("Agent Decides: Search required (bin is None).")
        return {"next": "search"}
    # If bin is empty list, we searched but found nothing.
    elif len(bin_val) == 0:
         logger.info("Agent Decides: Search yielded no results. Ending.")
         return {
             "next": "end",
             "messages": [AIMessage(content="I couldn't find any clips matching that description. Try broader keywords.")]
         }
    
    if not state.get("timeline"):
        logger.info("Agent Decides: Planning required.")
        return {
            "next": "planner",
            "messages": [AIMessage(content=f"I found {len(bin_val)} potential clips. I'm creating an edit plan now...")]
        }

    logger.info("Agent Decides: Done.")
    return {"next": "end"}

def search_node(state: FilmState):
    """
    Execute SQL search based on user intent.
    """
    intent = state.get("user_intent", "")
    if not intent:
        logger.warning("Warning: No user_intent found in state.")
        return {"bin": []}
    logger.info(f"Executing Search for: {intent}")
    
    # Construct a query using LLM ideally, but for MVP let's use a broad keyword search
    # We'll use the 'sql_search' tool logic directly here or via tool node if complex
    stop_words = {"find", "show", "me", "clips", "clip", "video", "videos", "footage", "of", "with", "searching", "looking", "for", "lets", "let's", "craft", "make", "create", "edit", "focussed", "focused"}
    keywords = [w for w in intent.lower().split() if w not in stop_words and len(w) > 2]
    
    if keywords:
        search_term = keywords[0] 
    else:
        search_term = intent # Fallback

    query = f"SELECT * FROM clips WHERE visual_description ILIKE '%{search_term}%' OR category ILIKE '%{search_term}%'"
    logger.info(f"Executing Search Query: {query}")
    # StructuredTool must be called with invoke
    try:
        results = sql_search.invoke({"query": query})
        logger.info(f"Search returned {len(results) if isinstance(results, list) else 'Error/Empty'} results.")
    except Exception as e:
        logger.error(f"Search Tool Error: {e}")
        results = []
    
    if isinstance(results, str):
        # Error or empty
        logger.warning(f"Search Result (String): {results}")
        return {"bin": []}
        
    return {"bin": results}

def planner_node(state: FilmState):
    """
    Generate Edit Plan using DSPy based on search results.
    """
    intent = state["user_intent"]
    bin_clips = state["bin"]
    
    logger.info(f"Generating Plan for {len(bin_clips)} candidate clips. User Intent: {intent}")
    if not bin_clips:
        logger.warning("No clips to plan with.")
        return {"timeline": []}

    try:
        # Call DSPy module
        # Note: dspy.settings.configure needs to be called somewhere globally before this
        # We assume it is configured in service.py
        plan_text = planner.forward(user_intent=intent, search_results=bin_clips)
        
        logger.debug(f"Raw Plan Text (Length: {len(plan_text)}): {plan_text[:100]}...") # Log first 100 chars
        
        # Parse text into structured timeline (simplistic parsing for MVP)
        # Assuming the LLM returns a list or readable text we can display
        # We store the raw text or structured dict if possible.
        # For MVP, we'll store a mock structure based on the text.
        timeline = [{"description": plan_text}] 
        return {
            "timeline": timeline,
            "messages": [AIMessage(content=plan_text)]
        }
        
    except Exception as e:
        logger.error(f"Planning Error: {e}")
        return {"timeline": []}

def validation_node(state: FilmState):
    """
    Validate if clips in timeline exist.
    """
    timeline = state.get("timeline", [])
    if not timeline:
        return {}
        
    # Extract clip names from timeline (if structured)
    # Since our planner output is text for now, this step is tricky without parsing.
    # We will skip deep validation in this pass or implement a regex extract if needed.
    return {}

# --- Graph Definition ---

workflow = StateGraph(FilmState)

workflow.add_node("agent", agent_node)
workflow.add_node("search", search_node)
workflow.add_node("planner", planner_node)
workflow.add_node("validation", validation_node)

workflow.add_edge(START, "agent")

# Conditional edges based on agent decision
def router(state):
    # Route based on the agent's decision stored in 'next'
    next_step = state.get("next")
    if next_step == "end":
        return "__end__"
    return next_step

workflow.add_conditional_edges("agent", router)
workflow.add_edge("search", "agent") 
workflow.add_edge("planner", "validation")
workflow.add_edge("validation", "agent")

app = workflow.compile(checkpointer=memory)
