from typing import TypedDict, List, Annotated
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage

class FilmState(TypedDict):
    """
    Represents the state of the film editing session.
    """
    messages: Annotated[List[BaseMessage], add_messages]
    user_intent: str
    bin: List[dict]  # List of clips found via search (candidates)
    timeline: List[dict]  # Ordered list of clips selected for the edit (The Edit Plan)
    next: str # control flow field
    sql_query: str # The last generated SQL query
    sql_error: str # Any error from execution
    active_agent: str # tracks which agent is currently active
