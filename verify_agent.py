
import os
import sys

# Ensure app is in path
sys.path.append(os.getcwd())

from app.agent.service import AgentService
from app.core.config import settings

def main():
    print("Initializing AgentService...")
    try:
        service = AgentService()
    except Exception as e:
        print(f"Failed to initialize service: {e}")
        return

    query = "Lets craft a mountain focussed fast cut edit"
    print(f"Streaming chat for query: '{query}'")
    
    try:
        for event in service.stream_chat(query):
            print(f"Event: {event}")
            if "messages" in event:
                print(f"Message: {event['messages'][-1].content}")
            elif "planner" in event:
                 print(f"Planner Output: {event['planner']}")
    except Exception as e:
        print(f"Error during streaming: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
