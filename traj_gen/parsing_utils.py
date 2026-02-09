import re
import json
import ast
import logging
import traceback
from typing import Dict, Optional, Tuple


def extract_json_from_response(response: str) -> Optional[Dict]:
    """
    LLM 응답에서 JSON dict를 추출한다.

    시도 순서:
    1. ```...``` 블록 안의 JSON
    2. 백틱 없이 raw JSON ({...})
    3. json.loads 실패 시 ast.literal_eval fallback

    Returns:
        파싱된 dict 또는 None (실패 시)
    """
    try:
        # 1) ```...``` 블록에서 JSON 추출 시도
        matches = re.findall(r"```(.*?)```", response, re.DOTALL)
        if matches:
            json_str = matches[-1]
            if json_str.startswith("json"):
                json_str = json_str[4:]
        else:
            # 2) 백틱 없이 raw JSON이 온 경우 fallback
            json_match = re.search(r"\{.*\}", response, re.DOTALL)
            json_str = json_match.group(0) if json_match else None

        if json_str:
            json_str = json_str.strip()
            # double curly brace 처리 (LLM이 {{...}} 형태로 반환하는 경우)
            while json_str.startswith("{{") and json_str.endswith("}}"):
                json_str = json_str[1:-1].strip()

            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                return ast.literal_eval(json_str)

    except Exception as e:
        logging.error(f"Parse error: {e}")
        logging.error(traceback.format_exc())

    return None


def parse_llm_response(response: str, is_valid: bool) -> Tuple[Dict, bool]:
    """
    LLM 응답을 파싱하여 (pred_dict, success) 튜플을 반환한다.

    Args:
        response: LLM 원본 응답 문자열
        is_valid: API 호출 성공 여부

    Returns:
        (parsed_dict, True) 또는 (default_failed_response, False)
    """
    if not is_valid:
        return default_failed_response(), False

    result = extract_json_from_response(response)
    if result is not None:
        return result, True

    return default_failed_response(), False


def default_failed_response() -> Dict:
    return {
        "task": "parse_failed",
        "action_in_natural_language": "parse_failed",
        "grounded_action": "stop",
    }