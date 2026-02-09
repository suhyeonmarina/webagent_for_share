import json
from traj_gen.llm_utils import call_gpt4v

class ContextExtractionAgent:
    def extract(self, trajectory_data):
        """
        trajectory의 screenshot와 action history에서 structured information 추출
        """
        prompt = f"""
        Extract key information from this task execution:
        - Dates selected
        - Locations
        - Prices
        - Booking references
        
        Actions performed: {trajectory_data['actions']}
        """
        
        extracted = call_gpt4v(self.args, messages=[...])
        return json.loads(extracted)  # {'destination': 'Paris', 'check_in': '2024-05-01'}
    
    def act_with_context(self, acc_tree, image_obs, task_type, context):
        prompt = f"""
        Task type: {task_type}
        Context from previous task: {json.dumps(context)}
        
        Example: If previous task booked a flight to Paris (May 1-7),
        now book a hotel in Paris for the same dates.
        
        Generate the first action for this task.
        """