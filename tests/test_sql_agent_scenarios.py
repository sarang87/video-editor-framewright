
import unittest
import os
import sys
import shutil
import warnings

# Add project root to path
sys.path.append(os.getcwd())

# Set env var for DuckDB path to a test file
TEST_DB_PATH = "test_clips_scenarios.duckdb"
os.environ["DUCKDB_PATH"] = TEST_DB_PATH

from app.agent.graph import app
from app.models import ClipMetadata
from app.services.database import DuckDBManager
from langchain_core.messages import HumanMessage
from app.agent import tools
import dspy

# Suppress ResourceWarnings about unclosed sockets/files if possible, 
# though they often come from third-party libs.
warnings.filterwarnings("ignore", category=ResourceWarning)

class TestSQLAgent(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        # Configure DSPy for the planner
        # Using the same vLLM endpoint as the main app
        vllm_base = os.getenv("VLLM_BASE_URL", "http://vllm:8000/v1")
        lm = dspy.LM(model="openai/Qwen/Qwen3-VL-8B-Instruct-FP8", api_base=vllm_base, api_key="token-is-ignored")
        dspy.configure(lm=lm)

        # Initialize DuckDB Manager
        cls.db = DuckDBManager(TEST_DB_PATH)
        
        # Inject our test db instance into tools
        # app/agent/tools.py has a global `db_manager`
        tools.db_manager = cls.db

    def setUp(self):
        # Reset DB state before each test
        self.db._get_connection().execute("DROP TABLE IF EXISTS clips")
        self.db.init_db()
        
        # Populate with mock data
        self.populate_mock_data()

    def populate_mock_data(self):
        clips = [
            ClipMetadata(
                clip_name="mountain_drone.mp4",
                category="B-Roll",
                visual_description="Drone shot of snowy mountains at sunset",
                shot_type="Wide",
                motion_detected="High",
                narrative_utility="Establishing",
                transition_point="Start",
                duration=5.0
            ),
            ClipMetadata(
                clip_name="forest_hike.mp4",
                category="B-Roll",
                visual_description="POV walking through a dense green forest",
                shot_type="POV",
                motion_detected="Medium",
                narrative_utility="Action",
                transition_point="Middle",
                duration=12.0
            ),
            ClipMetadata(
                clip_name="interview_subject.mp4",
                category="A-Roll",
                visual_description="Subject talking about climate change",
                shot_type="Medium Close-Up",
                motion_detected="Low",
                narrative_utility="Dialogue",
                transition_point="None",
                duration=45.0
            ),
             ClipMetadata(
                clip_name="city_lapse.mp4",
                category="B-Roll",
                visual_description="Time lapse of city traffic at night",
                shot_type="Time Lapse",
                motion_detected="High",
                narrative_utility="Transition",
                transition_point="End",
                duration=8.0
            )
        ]
        
        for clip in clips:
            self.db.insert_clip(clip)

    @classmethod
    def tearDownClass(cls):
        # Close connection
        cls.db.close()
        # Remove test file
        if os.path.exists(TEST_DB_PATH):
            try:
                os.remove(TEST_DB_PATH)
            except PermissionError:
                print(f"Warning: Could not remove {TEST_DB_PATH} (probably open in another process)")
            except Exception as e:
                print(f"Warning: Error removing test db: {e}")

    def run_agent(self, query: str):
        inputs = {
            "messages": [HumanMessage(content=query)],
            "user_intent": query
        }
        
        final_state = app.invoke(inputs, config={"configurable": {"thread_id": "test_thread"}})
        return final_state

    def test_find_clips_by_visual_description(self):
        print("\n--- Test: Find clips by description (mountains) ---")
        query = "Find me a drone shot of mountains."
        state = self.run_agent(query)
        
        # Check if 'bin' (parsed results) contains the correct clip
        bin_clips = state.get("bin", [])
        found_names = [c['clip_name'] for c in bin_clips]
        
        self.assertIn("mountain_drone.mp4", found_names)
        self.assertNotIn("interview_subject.mp4", found_names)
        
        # Check planner output
        timeline = state.get("timeline", [])
        if not timeline:
             print("\nDEBUG: State at failure:", state)
        self.assertTrue(len(timeline) > 0, "Planner should generate a timeline")

    def test_find_clips_with_duration_filter(self):
        print("\n--- Test: Find clips longer than 10 seconds ---")
        query = "Find clips that are longer than 10 seconds."
        state = self.run_agent(query)
        
        bin_clips = state.get("bin", [])
        found_names = [c['clip_name'] for c in bin_clips]
        
        self.assertIn("forest_hike.mp4", found_names)    # 12s
        self.assertIn("interview_subject.mp4", found_names) # 45s
        self.assertNotIn("mountain_drone.mp4", found_names) # 5s
        self.assertNotIn("city_lapse.mp4", found_names)     # 8s

    def test_find_by_shot_type(self):
        print("\n--- Test: Find clips with Shot Type 'POV' ---")
        query = "Find me a POV shot."
        state = self.run_agent(query)
        
        bin_clips = state.get("bin", [])
        found_names = [c['clip_name'] for c in bin_clips]
        
        self.assertIn("forest_hike.mp4", found_names)
        self.assertNotIn("mountain_drone.mp4", found_names)

    def test_no_results_found(self):
        print("\n--- Test: No results found ---")
        query = "Find me a video of aliens landing on mars."
        state = self.run_agent(query)
        
        bin_clips = state.get("bin", [])
        self.assertEqual(len(bin_clips), 0)
        
        # Verify agent communicates failure gracefully
        messages = state["messages"]
        last_msg = messages[-1].content
        # Planner usually says something like "I couldn't find any clips"
        self.assertTrue("couldn't find" in last_msg.lower() or "no clips" in last_msg.lower() or "sorry" in last_msg.lower(), 
                        f"Agent response should indicate failure. Got: {last_msg}")

if __name__ == '__main__':
    unittest.main()
