import os
import re
import json
import logging
import traceback
from typing import Dict, List, Optional

from PIL import Image

from .llm_utils import call_gpt4v
from .utils import pil_to_b64
from .SharedContext import SharedContext


class ExplorationVerifierAgent:
    def __init__(self, args):
        self.args = args

    def verify(self, trajectory_data: Dict, context: SharedContext,
               screenshot_paths: list[str]) -> Dict:
        action_history = self._format_action_history(trajectory_data)

        # 모드별 검증 프롬프트 구성
        if context.mode == "comparison":
            mode_instruction = self._build_comparison_instruction(context)
        else:
            mode_instruction = self._build_sequential_instruction(context)

        prompt = self._build_prompt(
            goal=context.high_level_goal,
            task_summary=trajectory_data.get("task_summary", ""),
            action_history=action_history,
            mode=context.mode,
            mode_instruction=mode_instruction,
        )

        # 이미지 메시지 구성
        messages = self._build_messages(prompt, screenshot_paths)

        # API 호출
        try:
            response, call_success = call_gpt4v(self.args, messages)
            if not call_success:
                logging.warning("Verifier API call failed after retries")
                return self._error_result("API call failed after retries")
        except Exception as e:
            logging.error(f"Verifier error: {e}")
            logging.error(traceback.format_exc())
            return self._error_result(str(e))

        # 응답 파싱
        return self._parse_response(response)

    def _format_action_history(self, trajectory_data: Dict) -> str:
        actions = trajectory_data.get("actions", [])
        lines = []
        for a in actions:
            step = a.get("step", "?")
            nl = a.get("step_action_nl", "unknown")
            grounded = a.get("new_action_grounded", "")
            url = a.get("URL_after", "")
            lines.append(f"Step {step}: {nl} ({grounded}) → {url}")
        return "\n".join(lines) if lines else "No actions recorded."

    def _build_comparison_instruction(self, context: SharedContext) -> str:
        criteria = ", ".join(context.comparison_criteria) if context.comparison_criteria else "not specified"
        target = context.comparison_target or "not specified"

        target_info_str = "None gathered yet."
        if context.comparison_target_info:
            lines = [f"  - {site}: {info}" for site, info in context.comparison_target_info.items()]
            target_info_str = "\n".join(lines)

        prev_results_str = "None."
        if context.results:
            lines = []
            for r in context.results:
                site = r.get("website", "unknown")
                info = r.get("extracted_info", {})
                lines.append(f"  - {site}: {info}")
            prev_results_str = "\n".join(lines)

        return f"""[COMPARISON MODE VERIFICATION]
This trajectory is part of a cross-site comparison task.

- Comparison Target: {target}
- Comparison Criteria: {criteria}

Information gathered from previous sites:
{target_info_str}

Previous site results:
{prev_results_str}

You must verify:
a) Did the agent search for the correct comparison target on this site?
b) Did the agent find and extract information relevant to the comparison criteria ({criteria})?
c) Is the extracted information specific enough to be compared with other sites?
   (e.g., exact price, rating, availability — not vague descriptions)
d) If the agent could not find the target, did it clearly report that the item is unavailable?"""

    def _build_sequential_instruction(self, context: SharedContext) -> str:
        dependency = context.dependency or "not specified"

        key_details_str = "None gathered yet."
        if context.key_details:
            lines = [f"  - {k}: {v}" for k, v in context.key_details.items()]
            key_details_str = "\n".join(lines)

        prev_results_str = "None."
        if context.results:
            lines = []
            for r in context.results:
                site = r.get("website", "unknown")
                task = r.get("task_accomplished", "")
                info = r.get("extracted_info", {})
                lines.append(f"  - {site}: task='{task}', info={info}")
            prev_results_str = "\n".join(lines)

        return f"""[SEQUENTIAL MODE VERIFICATION]
This trajectory is part of a multi-site sequential task where each site's output feeds into the next.

- Dependency: {dependency}

Key details from previous sites:
{key_details_str}

Previous site results:
{prev_results_str}

You must verify:
a) Did the agent incorporate the key_details from previous sites into its actions?
   (e.g., if a previous site found an address, did this site use that address?)
b) Did the agent perform a task that logically follows from the dependency chain?
c) Did the agent produce output (information or action) that can feed into the next site's task?
d) Are the extracted key_details accurate and specific enough for downstream use?"""

    def _build_prompt(self, goal: str, task_summary: str, action_history: str,
                      mode: str, mode_instruction: str) -> str:
        return f"""You are an expert evaluator for a multi-site web navigation agent.

The agent explores websites to complete a high-level goal. Your job is to evaluate whether the agent's trajectory on this particular site was successful.

=== TASK INFORMATION ===
High-level Goal: {goal}
Mode: {mode}
Site-specific Task Summary: {task_summary}

=== ACTION HISTORY ===
{action_history}

=== The final screenshot of the webpage is attached. ===

=== VERIFICATION INSTRUCTIONS ===

[PART 1: GOAL ACHIEVEMENT]
Evaluate whether the agent achieved its site-specific task based on the action history and the final screenshot.

*IMPORTANT*
- If a product has been added to cart/bag but purchase is pending, count as SUCCESS.
- If the agent reached a login page AFTER completing the main action (e.g., after adding to cart), count as SUCCESS.

{mode_instruction}

=== OUTPUT FORMAT ===
You MUST respond in the following JSON format:
```json
{{
    "goal_verification": {{
        "status": "success" or "failure",
        "reasoning": "<explain whether the site-specific task was achieved>"
    }},
    "mode_verification": {{
        "status": "success" or "failure",
        "reasoning": "<explain whether the mode-specific requirements were met>"
    }},
}}
```
"""

    def _build_messages(self, prompt: str, screenshot_paths: list[str]) -> list:
        messages = [
            {"role": "system", "content": [
                {"type": "text", "text": "You are an expert in evaluating web navigation agent trajectories."}
            ]},
        ]

        user_content = [{"type": "text", "text": prompt}]

        for path in screenshot_paths:
            if os.path.exists(path):
                try:
                    img = Image.open(path)
                    user_content.append({
                        "type": "image_url",
                        "image_url": {"url": pil_to_b64(img)}
                    })
                except Exception as e:
                    logging.warning(f"Failed to load screenshot {path}: {e}")

        messages.append({"role": "user", "content": user_content})
        return messages

    def _parse_response(self, response: str) -> Dict:
        result = {"raw_response": response}

        try:
            matches = re.findall(r"```(?:json)?(.*?)```", response, re.DOTALL)
            if matches:
                parsed = json.loads(matches[-1])
                result["goal_verification"] = parsed.get("goal_verification", {})
                result["mode_verification"] = parsed.get("mode_verification", {})
                return result
        except (json.JSONDecodeError, KeyError) as e:
            logging.warning(f"Failed to parse verifier JSON: {e}")

        # Fallback: 텍스트에서 status 추출
        response_lower = response.lower()
        status_matches = re.findall(r'["\']?(success|failure)["\']?', response_lower)
        fallback_status = status_matches[-1] if status_matches else "failure"

        result["goal_verification"] = {"status": fallback_status, "reasoning": "parsed from raw text"}
        result["mode_verification"] = {"status": fallback_status, "reasoning": "parsed from raw text"}

        return result

    def _error_result(self, error_msg: str) -> Dict:
        return {
            "goal_verification": {"status": "failure", "reasoning": error_msg},
            "mode_verification": {"status": "failure", "reasoning": error_msg},
            "raw_response": f"Error: {error_msg}"
        }
