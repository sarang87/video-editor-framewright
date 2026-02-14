import dspy
from typing import List
from pydantic import BaseModel, Field
from app.utils.logger import setup_logging

logger = setup_logging("agent_planner")

# Define the input/output signatures
class NarrativePlannerSignature(dspy.Signature):
    """
    Given a user's intent and a list of available clips (search results),
    create a cohesive Edit Plan (Timeline) that tells a story.
    """
    user_intent = dspy.InputField(desc="The narrative goal or story the user wants to tell.")
    search_results = dspy.InputField(desc="List of available clips with their metadata (JSON/Dict format).")
    
    edit_plan = dspy.OutputField(desc="A list of selected clips in order, including why they were chosen.")

class EditStep(BaseModel):
    clip_name: str
    reasoning: str = Field(desc="Why this clip fits here in the sequence.")
    
class EditPlan(BaseModel):
    steps: List[EditStep]

# Define the DSPy Module
class NarrativePlanner(dspy.Module):
    def __init__(self):
        super().__init__()
        # ChainOfThought for reasoning
        self.generate_plan = dspy.ChainOfThought(NarrativePlannerSignature)
        
    def forward(self, user_intent: str, search_results: List[dict]):
        # Convert list of dicts to string for context
        context_str = str(search_results)
        
        # Limit context if too large (naive truncation)
        if len(context_str) > 10000:
             context_str = context_str[:10000] + "... (truncated)"
             
        prediction = self.generate_plan(user_intent=user_intent, search_results=context_str)
        
        # Log Full Prediction Object for Debugging
        logger.debug(f"Full Prediction Object: {prediction}")
        # Validating that correct field is accessed
        logger.debug(f"Reasoning: {getattr(prediction, 'reasoning', 'No reasoning found')}")
        logger.info(f"Generated Plan: {prediction.edit_plan}")
        
        return prediction.edit_plan
