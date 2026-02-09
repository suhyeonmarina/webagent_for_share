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


class ComparisonTaskAgent:
    """
    Agent for generating comparison tasks that involve comparing similar content
    across multiple websites (e.g., price comparison, feature comparison).

    This agent only generates the task plan - actual execution is done by
    TaskRefinerAgent instances running in parallel browser sessions.
    """

    def __init__(self, args, browser_env, image_processor):
        self.args = args
        self.browser_env = browser_env
        self.image_processor = image_processor

        self.sm = """What does this webpage show? Imagine you are a real user on this webpage. Given the webpage screenshot and parsed HTML/accessibility tree, please propose a COMPARISON TASK that involves comparing similar information across multiple websites.

    Do the following step by step:
    1. Analyze the current webpage and identify what type of content/service it provides (e.g., e-commerce, travel booking, news, etc.)
    2. Generate a comparison task that a user might realistically perform across multiple similar websites
    3. Suggest 2-4 websites (including the current one) that would be relevant for this comparison task
    4. For each website, describe what specific information to search for and extract

    *COMPARISON TASK EXAMPLES*:
    - Compare prices of a specific product across different e-commerce sites
    - Compare flight prices from different airline/booking websites
    - Compare hotel prices and reviews from different travel sites
    - Compare news coverage of a topic from different news outlets
    - Compare features and pricing of similar software/services

    *IMPORTANT RULES*:

    * Task proposal rules *
    1. The comparison task should be specific and include concrete details (product name, dates, search terms, etc.)
    2. Suggested websites should be real, well-known alternatives in the same category
    3. The task should be feasible without requiring login
    4. ALL websites should use the SAME search query/criteria for fair comparison
    5. Each sub-task should clearly specify what information to extract (price, rating, availability, etc.)
    6. Use mock-up but realistic information (dates, product names, locations, etc.)

    * CRITICAL - Consistency Rules *
    7. "comparison_target" MUST list exactly 2-4 key items to compare (e.g., "price, rating, availability"). Keep it concise.
    8. "sub_task_description" MUST always include the search_query. Format: "Search for '[search_query]' on [website], then [next steps]..."
    9. "target_info" MUST extract the items listed in "comparison_target" for ALL websites. Use identical wording.

    *OUTPUT FORMAT*: Please give a short analysis, then put your answer within ``` ```, in the following JSON format:
    ```{
        "task": "<OVERALL COMPARISON TASK DESCRIPTION>",
        "comparison_target": "<LIST 2-4 KEY ITEMS TO COMPARE, e.g., 'price, rating, shipping cost'>",
        "search_query": "<THE COMMON SEARCH TERM/CRITERIA TO USE ON ALL WEBSITES>",
        "sub_tasks": [
            {
                "website_name": "<WEBSITE NAME>",
                "url": "<WEBSITE URL>",
                "sub_task_description": "Search for '[search_query]' on [website]. Then <next steps>...",
                "target_info": "<EXTRACT THE SAME ITEMS FROM comparison_target FOR ALL SITES>"
            },
            {
                "website_name": "<WEBSITE NAME>",
                "url": "<WEBSITE URL>",
                "sub_task_description": "Search for '[search_query]' on [website]. Then <next steps>...",
                "target_info": "<EXTRACT THE SAME ITEMS FROM comparison_target FOR ALL SITES>"
            }
        ]
    }```

    """

        self._task_plan = None

    def act(self, acc_tree, image_obs):
        """
        Generate a comparison task plan based on the current webpage.

        Note: This method only generates the task plan. It does NOT execute any actions.
        Actual execution should be done by creating multiple TaskRefinerAgent instances
        with separate browser sessions for each sub_task.

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
                encoding = tiktoken.encoding_for_model("gpt-4o-mini")
                n_op_tokens = len(encoding.encode(ans_1st_pass))
                n_tokens = n_inp_tokens + n_op_tokens

                logging.info(f"Number of tokens: {n_tokens}")

        # gpt calling 실패했을 때
        except Exception as e:
            ans_1st_pass = ""
            logging.info(traceback.format_exc())
            is_valid = False

        response = ans_1st_pass
        logging.info(f"response = {response}")

        ### 여기서 부터 코드 구조 정리함 ㅠㅠ

        pred, is_valid = self._parse_response(response, is_valid)
        self._task_plan = pred

        return response, pred, is_valid

    def _get_default_failed_response(self):
        """파싱 실패 시 반환할 기본 응답"""
        return {
            "task_type": "comparison",
            "task": "regex fail",
            "comparison_target": "regex fail",
            "search_query": "regex fail",
            "sub_tasks": []
        }

    def _extract_json_block(self, response):
        """LLM 응답에서 JSON 블록 추출"""
        # 방법 1: ```...``` 패턴으로 추출
        matches = re.findall(r"```(.*?)```", response, re.DOTALL)
        if matches:
            return matches[-1]

        # 방법 2: split으로 추출
        parts = response.split("```")
        if len(parts) >= 3:
            return parts[-2]

        return None

    def _parse_json_string(self, json_str):
        """JSON 문자열을 dict로 파싱"""
        if json_str.startswith("json"):
            json_str = json_str[4:]

        # 방법 1: json.loads
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass

        # 방법 2: ast.literal_eval
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

        # Step 1: JSON 블록 추출
        json_block = self._extract_json_block(response)
        if json_block is None:
            logging.error("Error in parsing the prediction ``````")
            return self._get_default_failed_response(), False

        # Step 2: JSON 파싱
        pred = self._parse_json_string(json_block)
        if pred is None:
            logging.error(f"Error in parsing the prediction dict {json_block}")
            return self._get_default_failed_response(), False

        # Add task_type and task_id by code (not generated by LLM)
        pred["task_type"] = "comparison"
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


    # TODO : 나중에 구현 완료해야됨, 틀만 잡아둔 상태
    def get_task_plan(self):
        """Get the generated task plan."""
        return self._task_plan

    def get_all_sub_tasks(self):
        """Get all sub-tasks in the comparison plan."""
        if self._task_plan is None:
            return []
        return self._task_plan.get("sub_tasks", [])

    def get_sub_task_by_id(self, task_id):
        """Get a specific sub-task by its task_id."""
        for sub_task in self.get_all_sub_tasks():
            if sub_task.get("task_id") == task_id:
                return sub_task
        return None

    def get_all_urls(self):
        """
        Get list of all URLs for parallel browser sessions.

        Returns:
            List of dicts with task_id, website_name, and url
        """
        return [
            {
                "task_id": st.get("task_id"),
                "website_name": st.get("website_name"),
                "url": st.get("url")
            }
            for st in self.get_all_sub_tasks()
        ]

    def get_search_query(self):
        """Get the common search query for all websites."""
        if self._task_plan is None:
            return None
        return self._task_plan.get("search_query")

    def get_comparison_target(self):
        """Get what is being compared (e.g., price, features)."""
        if self._task_plan is None:
            return None
        return self._task_plan.get("comparison_target")

    def create_parallel_refiner_configs(self):
        """
        Create configuration for parallel TaskRefinerAgent instances.

        Returns:
            List of dicts, each containing:
            - task_id: Unique identifier for this sub-task
            - url: Website URL to open in new browser session
            - website_name: Name of the website
            - sub_task_description: What to do on this site
            - search_query: Common search term to use
            - target_info: What information to extract
        """
        if self._task_plan is None:
            return []

        search_query = self._task_plan.get("search_query", "")

        configs = []
        for sub_task in self.get_all_sub_tasks():
            configs.append({
                "task_id": sub_task.get("task_id"),
                "url": sub_task.get("url"),
                "website_name": sub_task.get("website_name"),
                "sub_task_description": sub_task.get("sub_task_description"),
                "search_query": search_query,
                "target_info": sub_task.get("target_info"),
                "overall_task": self._task_plan.get("task"),
                "comparison_target": self._task_plan.get("comparison_target")
            })

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
    parser.add_argument("--init-url", type=str, default="https://www.booking.com/", help="initial url")
    parser.add_argument("--deployment", type=str, default="gpt-5-mini", help="model deployment")
    parser.add_argument("--viewport-width", type=int, default=1280)
    parser.add_argument("--viewport-height", type=int, default=720)
    parser.add_argument("--print-num-toks", action="store_true", default=False)
    args = parser.parse_args()

    viewport_size = {"width": args.viewport_width, "height": args.viewport_height}

    print(f"\n{'='*60}")
    print(f"Testing ComparisonTaskAgent")
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
    agent = ComparisonTaskAgent(args, browser_env, image_processor)
    response, pred, is_valid = agent.act(parsed_html_str, som_image_obs)

    print(f"\n{'='*60}")
    print("RESULT")
    print(f"{'='*60}")
    print(f"is_valid: {is_valid}")
    print(f"\nTask Plan:")
    print(json.dumps(pred, indent=2, ensure_ascii=False))


    ### 결과물 저장
    from datetime import datetime

    output_dir = "comparison_test_output"
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
