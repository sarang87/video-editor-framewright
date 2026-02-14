import dspy
from app.agent.graph import app as agent_app
from app.agent.state import FilmState
from app.core.config import settings

# Configure DSPy global settings
# We'll use vLLM as the LM for DSPy
lm = dspy.LM(model="openai/Qwen/Qwen3-VL-8B-Instruct-FP8", api_base=settings.VLLM_BASE_URL, api_key="token-is-ignored")
# Remove global configuration to avoid threading issues in Streamlit
# dspy.settings.configure(lm=lm)

class AgentService:
    def __init__(self):
        self.app = agent_app
        self.lm = lm

    def stream_chat(self, user_input: str, thread_id: str = "default_thread"):
        """
        Stream chat with the agent.
        """
        config = {"configurable": {"thread_id": thread_id}}
        
        # Initial state or update
        # For simplicity, we just pass the user intent as the input
        inputs = {"user_intent": user_input, "messages": [("user", user_input)]}
        
        # Run the graph with DSPy context
        # Using .stream() to get updates
        with dspy.context(lm=self.lm):
            for event in self.app.stream(inputs, config=config):
                yield event
            
    def get_state(self, thread_id: str = "default_thread"):
        config = {"configurable": {"thread_id": thread_id}}
        return self.app.get_state(config)
