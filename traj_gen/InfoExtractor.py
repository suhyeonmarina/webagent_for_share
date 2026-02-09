import re
import json
import logging
from typing import Dict
from PIL import Image

from .llm_utils import call_gpt4v
from .utils import pil_to_b64
from .SharedContext import SharedContext


class InfoExtractor:
    """
    각 사이트 탐색 완료 후 다음 task에 필요한 정보 추출
    """

    def __init__(self, args):
        self.args = args

        self.extraction_prompt = """Based on the exploration trajectory below, extract the key information that was discovered.

High-level Goal: {goal}
Mode: {mode}
Website: {website}

Actions performed:
{action_history}

Final page screenshot is attached.

{mode_specific_extraction}

*OUTPUT FORMAT*:
```json
{{
    "extracted_info": {{
        // key-value pairs of discovered information
    }},
    "comparison_target": "<if this is the first site in comparison mode, what should we compare?>",
    "search_query_used": "<the search query used, if any>",
    "summary": "<one sentence summary of what was accomplished>"
}}
```
"""

    def extract(self, trajectory_data: Dict, context: SharedContext,
                final_screenshot_path: str) -> Dict:
        """탐색 결과에서 정보 추출"""

        action_history = "\n".join([
            f"- {a.get('step_action_nl', 'unknown')}"
            for a in trajectory_data.get("actions", [])
        ])

        if context.mode == "comparison":
            criteria_hint = ""
            if context.comparison_criteria:
                criteria_hint = f"\nComparison criteria to focus on: {', '.join(context.comparison_criteria)}"
            mode_instructions = f"""Extract all details relevant to comparing this option/subject against alternatives from other sites.
This could include (depending on the goal): cost, duration, schedule, convenience, quality, distance, availability, ratings, features, restrictions, or any other distinguishing factors.{criteria_hint}"""
        else:
            mode_instructions = """Extract:
- Key information discovered (addresses, dates, prices, etc.)
- Any details that will be needed for subsequent website tasks
- Constraints or requirements learned"""

        prompt_text = self.extraction_prompt.format(
            goal=context.high_level_goal,
            mode=context.mode,
            website=trajectory_data.get("init_url", "unknown"),
            action_history=action_history,
            mode_specific_extraction=mode_instructions
        )

        # Load final screenshot
        try:
            final_img = Image.open(final_screenshot_path)
            img_b64 = pil_to_b64(final_img)
        except:
            img_b64 = None

        messages = [
            {"role": "system", "content": [{"type": "text", "text": "You are an information extraction assistant."}]},
            {"role": "user", "content": [
                {"type": "text", "text": prompt_text},
            ]}
        ]

        if img_b64:
            messages[1]["content"].append({
                "type": "image_url",
                "image_url": {"url": img_b64}
            })

        try:
            response, _ = call_gpt4v(self.args, messages)

            # Parse JSON response
            matches = re.findall(r"```(?:json)?(.*?)```", response, re.DOTALL)
            if matches:
                return json.loads(matches[-1])
        except Exception as e:
            logging.error(f"Info extraction error: {e}")

        return {
            "extracted_info": {},
            "summary": "extraction_failed"
        }