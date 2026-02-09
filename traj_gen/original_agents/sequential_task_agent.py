import ast
import json
import logging
import re
import traceback

import tiktoken
from PIL import Image

from ..utils import pil_to_b64
from traj_gen.llm_utils import call_gpt4v
from traj_gen.utils import calc_num_tokens


class SequentialTaskAgent:
    """
    Agent for generating sequential multi-task scenarios where multiple independent tasks
    are connected to achieve a larger goal (e.g., trip planning: book flight + book hotel + find transportation).

    Each task is independent and may involve different websites, but they are logically connected
    to accomplish an overarching objective.

    This agent only generates the task plan - actual execution is done by
    TaskRefinerAgent instances for each task (sequentially, as later tasks depend on earlier ones).
    """

    def __init__(self, args, browser_env, image_processor):
        self.args = args
        self.browser_env = browser_env
        self.image_processor = image_processor

        self.sm = """What does this webpage show? Imagine you are a real user on this webpage. Given the webpage screenshot and parsed HTML/accessibility tree, please propose a SEQUENTIAL MULTI-TASK SCENARIO where multiple independent tasks are connected to achieve a larger goal.

    Do the following step by step:
    1. Analyze the current webpage and identify what type of service it provides
    2. Think of a realistic scenario where a user would need to complete MULTIPLE RELATED BUT INDEPENDENT TASKS to achieve a goal
    3. Generate 2-4 connected tasks, each potentially involving different websites
    4. For each task, describe what needs to be done and what information to extract

    *SEQUENTIAL MULTI-TASK EXAMPLES*:

    Example 1 - Trip Planning:
    - Goal: "Plan a trip from Seoul to Jeju"
    - Task 1: Book a round-trip flight (Google Flights / Delta Air Lines)
    - Task 2: Reserve a hotel near the beach (Booking.com / Expedia / Airbnb)
    - Task 3: Find transportation from airport to hotel (Google Maps / Uber / Lyft)
    - Task 4: Look up tourist attractions and restaurants (TripAdvisor/Yelp)

    Example 2 - Job Application:
    - Goal: "Apply for a software engineer position"
    - Task 1: Search and find job postings (LinkedIn / Indeed / Glassdoor)
    - Task 2: Research the company (Glassdoor / Crunchbase)

    Example 3 - Moving to a New Place:
    - Goal: "Prepare for moving to a new apartment"
    - Task 1: Search for apartments (Zillow / Apartments.com / Redfin)
    - Task 2: Get moving service quotes (U-Haul / Moving.com)
    - Task 3: Find nearby supermarket on maps (Google Maps / Yelp)

    *IMPORTANT RULES*:

    * Scenario proposal rules *
    1. The overall goal should be a realistic scenario that requires multiple independent tasks
    2. Each task should be INDEPENDENT but LOGICALLY CONNECTED to the overall goal
    3. Tasks may involve DIFFERENT WEBSITES appropriate for each sub-task
    4. Each task should have its own clear objective and success criteria
    5. Tasks should NOT require login or payment (stop before those steps)
    6. Include specific, realistic details (dates, locations, names, preferences, etc.)
    7. Information from earlier tasks may be needed for later tasks (e.g., flight arrival time needed for hotel check-in)
    8. Tasks are executed SEQUENTIALLY (not in parallel), so later tasks can use results from earlier tasks

    *OUTPUT FORMAT*: Please give a short analysis, then put your answer within ``` ```, in the following JSON format:
    ```{
        "task": "<THE BIG PICTURE GOAL>",
        "scenario_description": "<BRIEF DESCRIPTION OF THE SCENARIO>",
        "shared_context": {
            "key_details": "<IMPORTANT DETAILS SHARED ACROSS TASKS - e.g., dates, locations, preferences>"
        },
        "sub_tasks": [
            {
                "website_name": "<WEBSITE NAME>",
                "url": "<URL>",
                "sub_task_description": "<DETAILED DESCRIPTION OF WHAT TO DO>",
                "dependencies": "none",
                "success_criteria": "<HOW TO KNOW THIS TASK IS COMPLETE>",
                "output_for_next_task": "<WHAT INFO TO PASS TO NEXT TASK>"
            },
            {
                "website_name": "<WEBSITE NAME>",
                "url": "<URL>",
                "sub_task_description": "<DETAILED DESCRIPTION>",
                "dependencies": "<WHAT INFO FROM PREVIOUS TASKS IS NEEDED>",
                "success_criteria": "<HOW TO KNOW THIS TASK IS COMPLETE>",
                "output_for_next_task": "<WHAT INFO TO PASS TO NEXT TASK>"
            }
        ]
    }```

    """

        self.task_plan = None

    def act(self, acc_tree, image_obs):
        """
        Generate a sequential multi-task plan based on the current webpage.

        Note: This method only generates the task plan. It does NOT execute any actions.
        Actual execution should be done by TaskRefinerAgent instances sequentially.

        Returns:
            response: Raw LLM response
            pred: Parsed task plan dict
            is_valid: Whether the task plan was successfully generated
        """
        is_valid = True

        try:
            messages = self.create_request(acc_tree, image_obs)

            ans_1st_pass, _ = call_gpt4v(
                self.args, messages
            )

            if self.args.print_num_toks:
                n_inp_tokens = calc_num_tokens(messages)
                encoding = tiktoken.encoding_for_model("gpt-5-mini")
                n_op_tokens = len(encoding.encode(ans_1st_pass))
                n_tokens = n_inp_tokens + n_op_tokens

                logging.info(f"Number of tokens: {n_tokens}")

        except Exception as e:
            ans_1st_pass = ""
            logging.info(traceback.format_exc())
            is_valid = False

        response = ans_1st_pass
        logging.info(f"response = {response}")

        pred, is_valid = self._parse_response(response, is_valid)
        self.task_plan = pred

        return response, pred, is_valid

    def _get_default_failed_response(self):
        """파싱 실패 시 반환할 기본 응답"""
        return {
            "task_type": "sequential",
            "task": "regex fail",
            "scenario_description": "regex fail",
            "shared_context": {},
            "sub_tasks": []
        }

    def _extract_json_block(self, response):
        """LLM 응답에서 JSON 블록 추출"""
        matches = re.findall(r"```(.*?)```", response, re.DOTALL)
        if matches:
            return matches[-1]

        parts = response.split("```")
        if len(parts) >= 3:
            return parts[-2]

        return None

    def _parse_json_string(self, json_str):
        """JSON 문자열을 dict로 파싱"""
        if json_str.startswith("json"):
            json_str = json_str[4:]

        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass

        try:
            return ast.literal_eval(json_str)
        except (ValueError, SyntaxError):
            pass

        return None

    def _parse_response(self, response, is_valid):
        """LLM 응답을 파싱하여 task plan dict로 변환"""

        if not is_valid:
            logging.error("Skip parsing because LLM call was invalid")
            return self._get_default_failed_response(), False

        json_block = self._extract_json_block(response)
        if json_block is None:
            logging.error("Error in parsing the prediction ``````")
            return self._get_default_failed_response(), False

        pred = self._parse_json_string(json_block)
        if pred is None:
            logging.error(f"Error in parsing the prediction dict {json_block}")
            return self._get_default_failed_response(), False

        # Add task_type and task_id by code (not generated by LLM)
        pred["task_type"] = "sequential"
        for idx, sub_task in enumerate(pred.get("sub_tasks", []), start=1):
            sub_task["task_id"] = idx

        return pred, is_valid

    def create_request(self, acc_tree, image_obs):
        prompt = [
            {
                "type": "text",
                "text": f"WEBSITE URL: {self.args.init_url}\n PARSED HTML/ACCESSIBILITY TREE:\n {acc_tree}",
            },
            {
                "type": "image_url",
                "image_url": {"url": pil_to_b64(Image.fromarray(image_obs))},
            },
        ]

        messages = [{"role": "system", "content": [{"type": "text", "text": self.sm}]}]

        messages.append({"role": "user", "content": prompt})
        return messages

    # ============ Helper methods ============

    # TODO : 나중에 구현 완료해야됨, 틀만 잡아둔 상태

    def get_task_plan(self):
        """Get the generated task plan."""
        return self.task_plan

    def get_all_sub_tasks(self):
        """Get all sub-tasks in the sequential scenario."""
        if self.task_plan is None:
            return []
        return self.task_plan.get("sub_tasks", [])

    def get_sub_task_by_id(self, task_id):
        """Get a specific sub-task by its task_id."""
        for sub_task in self.get_all_sub_tasks():
            if sub_task.get("task_id") == task_id:
                return sub_task
        return None

    def get_task(self):
        """Get the overall task of the sequential multi-task scenario."""
        if self.task_plan is None:
            return None
        return self.task_plan.get("task")

    def get_shared_context(self):
        """Get the shared context/details across all sub-tasks."""
        if self.task_plan is None:
            return {}
        return self.task_plan.get("shared_context", {})

    def get_all_urls(self):
        """Get list of all URLs for the sequential sub-tasks."""
        return [
            {
                "task_id": st.get("task_id"),
                "website_name": st.get("website_name"),
                "url": st.get("url")
            }
            for st in self.get_all_sub_tasks()
        ]

    def create_sequential_refiner_configs(self):
        """
        Create configuration for sequential TaskRefinerAgent execution.

        Returns:
            List of dicts (in order), each containing:
            - task_id: Execution order (1-based)
            - url: Website URL to open
            - website_name: Name of the website
            - sub_task_description: What to do on this site
            - dependencies: What info from previous tasks is needed
            - success_criteria: How to know task is complete
            - output_for_next_task: What info to pass to next task
        """
        if self.task_plan is None:
            return []

        configs = []
        for sub_task in self.get_all_sub_tasks():
            configs.append({
                "task_id": sub_task.get("task_id"),
                "url": sub_task.get("url"),
                "website_name": sub_task.get("website_name"),
                "sub_task_description": sub_task.get("sub_task_description"),
                "dependencies": sub_task.get("dependencies"),
                "success_criteria": sub_task.get("success_criteria"),
                "output_for_next_task": sub_task.get("output_for_next_task"),
                "task": self.task_plan.get("task"),
                "shared_context": self.task_plan.get("shared_context", {})
            })

        # Sort by task_id to ensure correct sequence
        configs.sort(key=lambda x: x.get("task_id", 0))

        return configs


if __name__ == "__main__":
    import argparse
    import sys
    import os

    # Add parent directory to path for imports
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from traj_gen.browser_env import ScriptBrowserEnv
    from traj_gen.processors import ImageObservationProcessor

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

    parser = argparse.ArgumentParser()
    parser.add_argument("--init-url", type=str, default="https://www.google.com/travel/flights", help="initial url")
    parser.add_argument("--deployment", type=str, default="gpt-5-mini", help="model deployment")
    parser.add_argument("--viewport-width", type=int, default=1280)
    parser.add_argument("--viewport-height", type=int, default=720)
    parser.add_argument("--print-num-toks", action="store_true", default=False)
    args = parser.parse_args()

    viewport_size = {"width": args.viewport_width, "height": args.viewport_height}

    print(f"\n{'='*60}")
    print(f"Testing SequentialTaskAgent")
    print(f"URL: {args.init_url}")
    print(f"Model: {args.deployment}")
    print(f"{'='*60}\n")

    # Setup browser environment
    browser_env = ScriptBrowserEnv(args, browser_type="chrome", viewport_size=viewport_size)
    browser_env.setup(args.init_url)

    # Setup image processor
    image_processor = ImageObservationProcessor(args, "image_som", viewport_size)

    # Get page state
    som_image_obs, parsed_html_str = image_processor.process_new(
        browser_env.page,
        browser_env.page.client,
        use_id_selector=True,
        intent=None,
    )

    # Create agent and run
    agent = SequentialTaskAgent(args, browser_env, image_processor)
    response, pred, is_valid = agent.act(parsed_html_str, som_image_obs)

    print(f"\n{'='*60}")
    print("RESULT")
    print(f"{'='*60}")
    print(f"is_valid: {is_valid}")
    print(f"\nTask Plan:")
    print(json.dumps(pred, indent=2, ensure_ascii=False))


    ### 결과물 저장
    from datetime import datetime

    output_dir = "sequential_test_output"
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    base_name = args.init_url.replace("https://", "").replace("/", "_")
    output_path = os.path.join(
        output_dir,
        f"output_{base_name}_{timestamp}.json"
    )

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(pred, f, indent=2, ensure_ascii=False)

    print(f"Saved to: {output_path}")


    # Cleanup
    browser_env.close()
    print(f"\n{'='*60}")
    print("Test completed!")
    print(f"{'='*60}")
