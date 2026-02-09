import logging
import traceback
from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any
from PIL import Image

from .actions import create_id_based_action
from .SharedContext import SharedContext

from .llm_utils import call_gpt4v
from .utils import pil_to_b64
from .parsing_utils import parse_llm_response


class BaseExplorationAgent(ABC):
    def __init__(self, args, browser_env, image_processor):
        self.args = args
        self.browser_env = browser_env
        self.image_processor = image_processor
        self.system_prompt_template = self._get_system_prompt_template()

    @abstractmethod
    def _get_system_prompt_template(self) -> str:
        pass

    @abstractmethod
    def _get_mode_instructions(self, context: SharedContext) -> str:
        pass

    @abstractmethod
    def _get_output_format(self, context: SharedContext) -> str:
        pass


    def _build_messages(self, system_prompt: str, acc_tree: str, image_obs) -> List[Dict]:
        prompt = [
            {
                "type": "text",
                "text": f"PARSED HTML/ACCESSIBILITY TREE:\n{acc_tree}",
            },
            {
                "type": "image_url",
                "image_url": {"url": pil_to_b64(Image.fromarray(image_obs))},
            },
        ]

        messages = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": prompt}
        ]

        return messages

    def _call_llm(self, messages: List[Dict]) -> Tuple[str, bool]:
        try:
            response, _ = call_gpt4v(self.args, messages)
            return response, True
        except Exception as e:
            logging.error(traceback.format_exc())
            return "", False

    def _process_response(self, response: str, is_valid: bool, context: SharedContext = None) -> Tuple[Dict, bool]:
        pred, is_action_valid = parse_llm_response(response, is_valid)

        if is_action_valid and context is not None:
            context.update_from_prediction(pred)

        return pred, is_action_valid

    def _execute_if_valid(self, pred: Dict, is_action_valid: bool) -> bool:
        if is_action_valid and pred.get("grounded_action", "").strip().lower() != "stop":
            is_success = self._execute_action(pred["grounded_action"])
            if not is_success:
                return False
        return is_action_valid

    def _execute_action(self, action_grounded: str) -> bool:
        try:
            cur_action = create_id_based_action(action_grounded)
            logging.info(f"Action to be executed: {cur_action}")
            return self.browser_env.step(cur_action)
        except Exception as e:
            logging.error(f"Action execution error: {e}")
            return False