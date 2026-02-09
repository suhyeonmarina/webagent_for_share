import ast
import json
import logging
import re
import traceback

import tiktoken
from PIL import Image

from .utils import pil_to_b64
from .actions import create_id_based_action, create_none_action
from traj_gen.llm_utils import call_gpt4v
from traj_gen.utils import calc_num_tokens

# act_multi_task() -> 1개 action 실행
# is_sub_task_complete = True (stop action시에)
# advance_to_next_task() -> 다음 sub-task로 이동
# is_all_tasks_complete() -> 전체 완료 여부 체크

class TaskRefinerAgent:
    def __init__(self, args, browser_env, image_processor):
        self.args = args
        self.browser_env = browser_env
        self.image_processor = image_processor

        # Track multi-task state
        self.multi_task_plan = None
        self.current_task_index = 0
        self.task_type = "single"  # "single", "comparison", "sequential"
        self.task_results = []  # Store results from completed tasks

        self.sm = """Imagine you are real user on this webpage, and your overall task is {overall_task}. This is the list of actions you have performed which lead to the current page {prev_action_list}. You are also given the webpage screenshot and parsed HTML/accessibility tree.
    Do the following step by step:
    1. Please predict what action the user might perform next that is consistent with the overall task and previous action list in natural language.
    2. Then based on the parsed HTML/accessibility tree of the webpage and the natural language action, generate the grounded action.
    3. Update the overall task aligned with this set of actions.

    *  Task update rules *
    1. The task must contain some actions: "Buy, Book, Find, Check, Choose show me, give me, add to cart, ...', ideally invovling transactions with a specific product or service. If possible, avoid information seeking tasks like "explore, review, read" etc. 
    2. You should only propose tasks that do not require login to execute the task.
    3. You should propose tasks that are clear and specific, e.g. it should contain details like "buy/book something under $100", "find a product with 4 stars" etc.
    4. Update the details of the task, such price, date, location, etc. based on the current set of actions and the proposed action.

    *ACTION SPACE*: Your action space is: [`click [element ID]`, `type [element ID] [content]`, `select [element ID] [content of option to select]`, `scroll [up]`, `scroll [down]`, and `stop`].
    Action output should follow the syntax as given below:
    `click [element ID]`: This action clicks on an element with a specific id on the webpage.
    `type [element ID] [content]`: Use this to type the content into the field with id. By default, the "Enter" key is pressed after typing. Both the content and the id should be within square braces as per the syntax.
    `select [element ID] [content of option to select]`: Select an option from a dropdown menu. The content of the option to select should be within square braces. When you get (select and option) tags from the accessibility tree , you need to select the serial number (element_id) corresponding to the select tag , not the option, and select the most likely content corresponding to the option as input.
    `scroll [down]`: Scroll the page down. 
    `scroll [up]`: Scroll the page up.

    *IMPORTANT* To be successful, it is important to STRICTLY follow the below rules:

    *  Action generation rules *
    1. You should generate a single atomic action at each step.
    2. The action should be an atomic action from the given action space - click, type, scroll (up or down) or stop
    3. The arguments to each action should be within square braces. For example, "click [127]", "type [43] [content to type]", "scroll [up]", "scroll [down]".
    4. The natural language form of action (corresponding to the field "action_in_natural_language") should be consistent with the grounded version of the action (corresponding to the field "grounded_action"). Do NOT add any additional information in the grounded action. For example, if a particular element ID is specified in the grounded action, a description of that element must be present in the natural language action. 
    5. If the type action is selected, the natural language form of action ("action_in_natural_language") should always specify the actual text to be typed. 
    6. You should issue a “stop” action if the current webpage asks to login or for credit card information. 
    7. To input text, there is NO need to click textbox first, directly type content. After typing, the system automatically hits the `ENTER` key.
    8. STRICTLY Avoid repeating the same action (click/type) if the webpage remains unchanged. You may have selected the wrong web element.
    9. Do NOT use quotation marks in the action generation.

    The output should be in below format:
    *OUTPUT FORMAT*: Please give a short analysis of the screenshot, parsed HTML/accessibility tree, and history, then put your answer within ``` ```, for example, "In summary, the proposed task and the corresponding action is: ```{{"task": <TASK>:str, "action_in_natural_language":<ACTION_IN_NATURAL_LANGUAGE>:str, "grounded_action": <ACTION>:str}}```"
    """

    # ============ Response Parsing Helper Methods ============

    def _get_default_failed_response(self):
        """파싱 실패 시 반환할 기본 응답"""
        return {
            "task": "regex fail",
            "action_in_natural_language": "regex fail",
            "grounded_action": "regex fail",
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
        """LLM 응답을 파싱하여 prediction dict로 변환"""
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

        return pred, is_valid



    # ============ Main Action Methods ============


    def _execute_action(self, action_grounded):
        """grounded action을 실행하고 결과 반환"""
        try:
            cur_action = create_id_based_action(action_grounded)
        except Exception:
            logging.error("Action parsing error")
            logging.error(traceback.format_exc())
            return False

        logging.info(f"Action to be executed: {cur_action}")
        is_success = self.browser_env.step(cur_action)
        logging.info(f"URL: {self.browser_env.page.url}")

        return is_success


    def act(self, acc_tree, image_obs, action_history, refined_goal):
        """단일 task에 대해 한 개의 action 실행"""
        is_action_valid = True
        self.refined_goal = refined_goal

        # Step 1: LLM 호출
        try:
            messages = self.create_request(acc_tree, image_obs, action_history)
            response, _ = call_gpt4v(self.args, messages)

            if self.args.print_num_toks:
                n_inp_tokens = calc_num_tokens(messages)
                encoding = tiktoken.encoding_for_model("gpt-5-mini")
                n_op_tokens = len(encoding.encode(response))
                logging.info(f"Number of tokens: {n_inp_tokens + n_op_tokens}")

        except Exception:
            response = ""
            logging.info(traceback.format_exc())
            is_action_valid = False

        logging.info(f"response = {response}")

        # Step 2: 응답 파싱
        pred, is_action_valid = self._parse_response(response, is_action_valid)

        # Step 3: Action 실행
        if is_action_valid:
            is_success = self._execute_action(pred["grounded_action"])
            if not is_success:
                is_action_valid = False

        return response, pred, is_action_valid

    def create_request(self, acc_tree, image_obs, action_history):
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

        messages = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": self.sm.format(
                            prev_action_list=action_history,
                            overall_task=self.refined_goal,
                        ),
                    }
                ],
            }
        ]

        messages.append({"role": "user", "content": prompt})
        return messages

    def set_multi_task_plan(self, task_plan, task_type):
        """
        Set the multi-task plan from ComparisonTaskAgent or SequentialTaskAgent.

        Args:
            task_plan: The full task plan dict from Comparison/Sequential agent
            task_type: "comparison" or "sequential"
        """
        self.multi_task_plan = task_plan
        self.task_type = task_type
        self.current_task_index = 0
        self.task_results = []

    def get_current_sub_task(self):
        """Get the current sub-task based on task type."""
        if self.multi_task_plan is None:
            return None

        # Both comparison and sequential now use "sub_tasks"
        sub_tasks = self.multi_task_plan.get("sub_tasks", [])

        if self.current_task_index < len(sub_tasks):
            return sub_tasks[self.current_task_index]
        return None

    def get_overall_context(self):
        """Get overall context for the current multi-task scenario."""
        if self.multi_task_plan is None:
            return ""

        task = self.multi_task_plan.get('task', '')
        if self.task_type == "comparison":
            return f"Comparison Task: {task} - Comparing: {self.multi_task_plan.get('comparison_target', '')}"
        elif self.task_type == "sequential":
            return f"Sequential Task: {task} - Context: {self.multi_task_plan.get('shared_context', {}).get('key_details', '')}"
        return ""

    def advance_to_next_task(self, task_result=None):
        """
        Move to the next sub-task in the multi-task plan.

        Args:
            task_result: Optional result/data from the completed task to pass to next task

        Returns:
            next_sub_task: The next sub-task dict, or None if all tasks completed
        """
        if task_result:
            self.task_results.append({
                "task_index": self.current_task_index,
                "result": task_result
            })

        self.current_task_index += 1
        return self.get_current_sub_task()

    def create_multi_task_request(self, acc_tree, image_obs, action_history):
        """
        Create request for multi-task scenarios (comparison or sequential).
        Uses context from the overall plan and previous tasks.
        """
        current_sub_task = self.get_current_sub_task()
        if current_sub_task is None:
            return None

        overall_context = self.get_overall_context()

        # Build context from previous task results
        prev_results_context = ""
        if self.task_results:
            prev_results_context = "\n\nResults from previous tasks:\n"
            for result in self.task_results:
                prev_results_context += f"- Task {result['task_index'] + 1}: {result['result']}\n"

        # Get sub-task specific info (unified field names for both types)
        sub_task_desc = current_sub_task.get("sub_task_description", "")
        website_name = current_sub_task.get("website_name", "")

        if self.task_type == "comparison":
            task_context = f"Current website: {website_name}\nSub-task: {sub_task_desc}"
        else:  # sequential
            dependencies = current_sub_task.get("dependencies", "none")
            task_context = f"Current website: {website_name}\nSub-task: {sub_task_desc}\nDependencies from previous tasks: {dependencies}"

        multi_task_prompt = f"""{overall_context}
{prev_results_context}
{task_context}

Previous actions in this sub-task: {action_history}

Based on the current webpage, predict the next action to complete this sub-task."""

        prompt = [
            {
                "type": "text",
                "text": f"WEBSITE URL: {self.browser_env.page.url}\n PARSED HTML/ACCESSIBILITY TREE:\n {acc_tree}",
            },
            {
                "type": "image_url",
                "image_url": {"url": pil_to_b64(Image.fromarray(image_obs))},
            },
        ]

        messages = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": self.sm.format(
                            prev_action_list=action_history,
                            overall_task=multi_task_prompt,
                        ),
                    }
                ],
            }
        ]

        messages.append({"role": "user", "content": prompt})
        return messages

    def act_multi_task(self, acc_tree, image_obs, action_history):
        """
        Multi-task 시나리오에서 한 개의 action 실행

        Returns:
            response: Raw LLM response
            pred: Parsed prediction dict
            is_action_valid: Whether the action was successfully executed
            is_sub_task_complete: Whether the current sub-task is complete (stop action)
        """
        is_action_valid = True
        is_sub_task_complete = False

        # Step 1: LLM 호출
        try:
            messages = self.create_multi_task_request(acc_tree, image_obs, action_history)
            if messages is None:
                logging.error("No more sub-tasks to execute")
                return None, None, False, True

            response, _ = call_gpt4v(self.args, messages)

        except Exception:
            response = ""
            logging.info(traceback.format_exc())
            is_action_valid = False

        logging.info(f"Multi-task response = {response}")

        # Step 2: 응답 파싱
        pred, is_action_valid = self._parse_response(response, is_action_valid)

        # Step 3: Action 실행 또는 sub-task 완료 처리
        if is_action_valid:
            action_grounded = pred.get("grounded_action", "")

            if action_grounded.strip().lower() == "stop":
                is_sub_task_complete = True
                logging.info(f"Sub-task {self.current_task_index + 1} completed")
            else:
                is_success = self._execute_action(action_grounded)
                if not is_success:
                    is_action_valid = False

        return response, pred, is_action_valid, is_sub_task_complete

    def is_all_tasks_complete(self):
        """Check if all sub-tasks in the multi-task plan are complete."""
        if self.multi_task_plan is None:
            return True

        # Both comparison and sequential now use "sub_tasks"
        total_tasks = len(self.multi_task_plan.get("sub_tasks", []))
        return self.current_task_index >= total_tasks

    def get_next_task_url(self):
        """Get the URL for the next sub-task (for new browser session)."""
        next_task = self.get_current_sub_task()
        if next_task is None:
            return None

        # Both comparison and sequential now use "url"
        return next_task.get("url")
