import re
import json
import logging
from typing import List
from PIL import Image

from .base_exploration_agent import BaseExplorationAgent
from .SharedContext import SharedContext


class ExplorationRefinerAgent(BaseExplorationAgent):

    def _get_system_prompt_template(self) -> str:
        return """You are exploring a website to accomplish a goal. Continue from previous actions.

{context}

Current Website: {current_url}
Current Sub-task: {current_task}
Previous Actions: {action_history}

Based on the current webpage state, do the following:
1. Predict the next action to complete the sub-task.
2. Ground the action to a specific element.
3. Update the sub-task description if needed based on what you've learned.

{mode_specific_instructions}

*ACTION SPACE*: [`click [element ID]`, `type [element ID] [content]`, `select [element ID] [content of option to select]`, `scroll [up]`, `scroll [down]`, `stop`]

*IMPORTANT RULES*:
1. Generate a single atomic action.
2. Issue "stop" when the sub-task is complete or if login/payment is required.
3. Do NOT repeat the same action if page is unchanged.
4. No quotation marks in action generation.

{output_format}
"""

    def _get_mode_instructions(self, context: SharedContext) -> str:
        if context.mode == "comparison":
            return """*COMPARISON MODE*:
- You are comparing the same target across different websites.
- As you navigate, look for information matching the comparison criteria.
- When you find relevant information (e.g., a price, a rating, availability), include it in "comparison_target_info".
- When issuing "stop", make sure "comparison_target_info" contains the value for the comparison criteria on this site."""
        else:  # sequential
            return """*SEQUENTIAL MODE*:
- You are in a causal chain where each site's output feeds into the next site's task.
- If key_details from previous steps are available in the context, USE them to guide your actions.
- As you discover concrete information relevant to the dependency (e.g., an address, a date, a price, a VIN), include it in key_details.
- When issuing "stop", make sure key_details contains all information the next site will need."""

    def _get_output_format(self, context: SharedContext) -> str:
        if context.mode == "comparison":
            return """*OUTPUT FORMAT*:
```{
  "task": "<UPDATED_SUB_TASK>",
  "comparison_criteria": ["<SINGLE_CRITERION>"],
  "comparison_target_info": {"<SITE_NAME>": "<OBSERVED_VALUE>"},
  "action_in_natural_language": "<ACTION_DESCRIPTION>",
  "grounded_action": "<ACTION>"
}```
Note: "comparison_target_info" should be a dict with site name as key and concrete value (e.g., price, rating) as value. Update it whenever you discover relevant information."""
        else:  # sequential
            return """*OUTPUT FORMAT*:
```{
  "task": "<UPDATED_SUB_TASK>",
  "dependency": "<KEY_INFO_THE_NEXT_SITE_WILL_NEED_FROM_THIS_TASK>",
  "key_details": {"<KEY>": "<VALUE_RELATED_TO_DEPENDENCY>"},
  "action_in_natural_language": "<ACTION_DESCRIPTION>",
  "grounded_action": "<ACTION>"
}```
Note: "key_details" should contain concrete values related to the dependency that the next site will need. Update key_details whenever you discover new relevant information on the page."""

    def act(self, acc_tree: str, image_obs, action_history: List[str],
            current_task: str, context: SharedContext):
        # 시스템 프롬프트 생성
        system_prompt = self.system_prompt_template.format(
            context=context.to_prompt_string(),
            current_url=self.browser_env.page.url,
            current_task=current_task,
            action_history=action_history,
            mode_specific_instructions=self._get_mode_instructions(context),
            output_format=self._get_output_format(context)
        )

        # LLM 호출
        messages = self._build_messages(system_prompt, acc_tree, image_obs)
        response, is_valid = self._call_llm(messages)
        logging.info(f"Exploration Refiner Agent response = {response}")

        # 응답 처리 및 context 업데이트
        pred, is_action_valid = self._process_response(response, is_valid, context)

        # 액션 실행
        is_action_valid = self._execute_if_valid(pred, is_action_valid)

        return response, pred, is_action_valid

    def make_final_comparison(self, context: SharedContext, image_obs=None):
        """
        comparison 모드에서 stop 시, comparison_criteria와 comparison_target_info를 기반으로
        여러 옵션 중 최종 답변(final_answer)을 선택한다.
        """
        if context.mode != "comparison":
            return None

        if not context.comparison_target_info:
            logging.warning("No comparison_target_info available for final comparison")
            return None

        options_text = "\n".join(
            f"  - {site}: {info}"
            for site, info in context.comparison_target_info.items()
        )

        prompt_text = f"""You have finished exploring multiple websites to compare options.

High-level Goal: {context.high_level_goal}
Comparison Target: {context.comparison_target or "N/A"}
Comparison Criteria: {', '.join(context.comparison_criteria) if context.comparison_criteria else "N/A"}

Information gathered from each site:
{options_text}

Based on the comparison criteria above, choose the BEST option among the gathered results and explain WHY.

*OUTPUT FORMAT*:
```{{
  "final_answer": "<THE BEST OPTION (site name + key value)>",
  "reasoning": "<WHY this option is the best based on the comparison criteria>",
  "ranking": [
    {{"site": "<SITE_NAME>", "value": "<VALUE>", "rank": 1}},
    {{"site": "<SITE_NAME>", "value": "<VALUE>", "rank": 2}}
  ]
}}```
"""
        user_content = [{"type": "text", "text": prompt_text}]
        if image_obs is not None:
            from .utils import pil_to_b64
            user_content.append({
                "type": "image_url",
                "image_url": {"url": pil_to_b64(Image.fromarray(image_obs))},
            })

        messages = [
            {"role": "system", "content": [{"type": "text", "text": "You are a comparison analyst. Pick the best option based on the given criteria."}]},
            {"role": "user", "content": user_content},
        ]

        response, is_valid = self._call_llm(messages)
        logging.info(f"Final comparison response = {response}")

        result, is_valid = self._process_response(response, is_valid)
        if is_valid:
            return result
        else:
            logging.error("Failed to parse final comparison response")
            return {"final_answer": "parse_failed", "reasoning": response, "ranking": []}


if __name__ == "__main__":
    import argparse
    import os
    from PIL import Image
    from datetime import datetime

    from traj_gen.browser_env import ScriptBrowserEnv
    from traj_gen.processors import ImageObservationProcessor
    from traj_gen.SharedContext import SharedContext

    # --- argparse ---
    parser = argparse.ArgumentParser(description="Test ExplorationRefinerAgent")
    parser.add_argument("--prev-result", type=str, default="explore_result/www.uniqlo.com__20260204_210142",
                        help="Path to previous explore_result directory (e.g., explore_result/www.yelp.com__20260202_161127)")
    parser.add_argument("--deployment", type=str, default="gpt-5-mini",
                        choices=["gpt-4o", "gpt-4o-mini", "gpt-5-mini"],
                        help="LLM model deployment")
    parser.add_argument("--viewport-width", type=int, default=1280)
    parser.add_argument("--viewport-height", type=int, default=720)
    parser.add_argument("--max-steps", type=int, default=5)

    args = parser.parse_args()

    # --- 이전 결과 로드 ---
    prev_result_path = os.path.join(args.prev_result, "result.json")
    with open(prev_result_path, "r", encoding="utf-8") as f:
        prev_result = json.load(f)

    init_url = prev_result.get("init_url", prev_result.get("url"))
    mode = prev_result["mode"]
    prev_pred = prev_result["pred"]
    prev_actions = prev_result.get("actions", [])

    # --- Setup: 출력 폴더 생성 ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    site_name = re.sub(r'[^\w\-.]', '_', init_url.replace("https://", "").replace("http://", ""))
    run_dir = os.path.join(os.getcwd(), "refine_result", f"{site_name}_refiner_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join(run_dir, "run.log"), mode="w"),
        ],
    )

    viewport_size = {"width": args.viewport_width, "height": args.viewport_height}

    # 1) SharedContext 복원
    context = SharedContext(
        high_level_goal=prev_pred["high_level_goal"],
        mode=mode,
        comparison_target=prev_pred.get("comparison_target") if mode == "comparison" else None,
        comparison_criteria=prev_pred.get("comparison_criteria", []) if mode == "comparison" else [],
        dependency=prev_pred.get("dependency") if mode == "sequential" else None,
        key_details=prev_pred.get("key_details", {}) if mode == "sequential" else {},
    )
    logging.info(f"Restored SharedContext:\n{context.to_prompt_string()}")

    # 2) Browser 초기화 (이전 결과의 URL로 시작)
    logging.info(f"Setting up browser for {init_url}")
    browser_env = ScriptBrowserEnv(args, browser_type="chrome", viewport_size=viewport_size)
    browser_env.setup(init_url)

    # 3) Image processor 초기화
    image_processor = ImageObservationProcessor(args, "image_som", viewport_size)

    # 4) Proposal agent의 첫 action 재실행
    #    NOTE: SOM element ID는 브라우저 세션마다 달라지므로 직접 replay 불가.
    #    대신 process_new()로 현재 페이지의 SOM을 생성한 뒤 step()을 시도한다.
    first_action_str = prev_pred.get("grounded_action", "stop")
    logging.info(f"Proposal agent action to replay: {first_action_str}")
    replay_success = False

    if first_action_str.strip().lower() != "stop":
        from traj_gen.actions import create_id_based_action
        try:
            # 먼저 현재 페이지의 SOM을 생성 (element ID 매핑용)
            browser_env.page.wait_for_load_state("networkidle", timeout=5000)
        except:
            pass

        try:
            cur_action = create_id_based_action(first_action_str)
            replay_success = browser_env.step(cur_action)
            if replay_success:
                logging.info("Proposal action replayed successfully")
            else:
                logging.warning(f"Proposal action replay returned False (element ID mismatch across sessions)")
        except Exception as e:
            logging.warning(f"Failed to replay proposal action: {e}")

    if not replay_success and first_action_str.strip().lower() != "stop":
        logging.info("Replay failed — refiner will start from current page state (proposal action skipped)")

    # 5) Refiner loop
    agent = ExplorationRefinerAgent(args, browser_env, image_processor)
    current_task = prev_pred.get("sub_task", prev_pred.get("task", ""))
    if replay_success:
        action_history = [
            f"{prev_pred.get('action_in_natural_language', '')} -> {first_action_str}"
        ]
    else:
        action_history = []  # replay 실패 시 빈 히스토리로 시작

    # proposal 스텝을 포함한 전체 trajectory (레거시 포맷)
    all_actions = list(prev_actions) if replay_success else []  # replay 실패 시 proposal record 제외

    for step_i in range(args.max_steps):
        logging.info(f"\n{'='*40} Step {step_i + 1} {'='*40}")

        # Observation
        try:
            browser_env.page.wait_for_load_state("networkidle", timeout=5000)
        except:
            pass

        som_image_obs, parsed_html_str = image_processor.process_new(
            browser_env.page, browser_env.page.client,
            use_id_selector=True, intent=None,
        )

        # 스크린샷 저장
        browser_env.page.screenshot(path=os.path.join(run_dir, f"screenshot_step{step_i + 1}_before.png"))
        Image.fromarray(som_image_obs).save(os.path.join(run_dir, f"screenshot_step{step_i + 1}_som.png"))

        # Agent 실행
        response, pred, is_action_valid = agent.act(
            parsed_html_str, som_image_obs, action_history, current_task, context
        )

        # bounding_box 추출
        bounding_box = None
        try:
            grounded = pred.get("grounded_action", "")
            match = re.search(r"\[(\d+)\]", grounded)
            if match:
                element_id = match.group(1)
                info = image_processor.som_id_info.get(element_id)
                if info:
                    bounding_box = {
                        "x": info[0], "y": info[1],
                        "width": info[2], "height": info[3],
                    }
        except Exception:
            pass

        # URL_after 캡처
        url_after = browser_env.page.url

        # 레거시 포맷 action record
        action_record = {
            "acc_tree_before": parsed_html_str,
            "step_action_nl": pred.get("action_in_natural_language", ""),
            "new_action_grounded": pred.get("grounded_action", ""),
            "bounding_box_coord": bounding_box,
            "step_refined_goal": pred.get("task", current_task),
            "step_reasoning_response": response,
            "URL_after": url_after,
        }
        all_actions.append(action_record)

        # 결과 출력
        print(f"\n--- Step {step_i + 1} ---")
        print(f"Task        : {action_record['step_refined_goal']}")
        print(f"Action (NL) : {action_record['step_action_nl']}")
        print(f"Action (GT) : {action_record['new_action_grounded']}")
        print(f"BBox        : {bounding_box}")
        print(f"URL After   : {url_after}")
        if pred.get("key_details"):
            print(f"Key Details : {pred['key_details']}")
        print(f"Valid       : {is_action_valid}")

        # 액션 실행 후 스크린샷
        try:
            browser_env.page.screenshot(path=os.path.join(run_dir, f"screenshot_step{step_i + 1}_after.png"))
        except:
            pass

        # stop 또는 실패 시 종료
        if not is_action_valid or pred.get("grounded_action", "").strip().lower() == "stop":
            logging.info(f"Stopping at step {step_i + 1}")
            break

        # action history 업데이트
        action_history.append(
            f"{pred.get('action_in_natural_language', '')} -> {pred.get('grounded_action', '')}"
        )
        # task 업데이트
        if pred.get("task"):
            current_task = pred["task"]

    # 6) Comparison 모드: final_answer 생성
    final_comparison = None
    if mode == "comparison" and context.comparison_target_info:
        logging.info("Comparison mode — running final comparison to pick the best option...")
        final_comparison = agent.make_final_comparison(context)
        if final_comparison:
            print(f"\n{'='*60}")
            print("FINAL COMPARISON RESULT")
            print(f"{'='*60}")
            print(f"Final Answer : {final_comparison.get('final_answer', 'N/A')}")
            print(f"Reasoning    : {final_comparison.get('reasoning', 'N/A')}")
            if final_comparison.get("ranking"):
                print("Ranking:")
                for r in final_comparison["ranking"]:
                    print(f"  #{r.get('rank', '?')} {r.get('site', '?')}: {r.get('value', '?')}")
            print(f"{'='*60}")

    # 7) 최종 결과 출력
    print("\n" + "=" * 60)
    print("ExplorationRefinerAgent Result")
    print("=" * 60)
    print(f"URL           : {init_url}")
    print(f"Mode          : {mode}")
    print(f"Goal          : {context.high_level_goal}")
    print(f"Total actions : {len(all_actions)} (proposal: {len(prev_actions)}, refiner: {len(all_actions) - len(prev_actions)})")
    print(f"Key Details   : {context.key_details}")
    if final_comparison:
        print(f"Final Answer  : {final_comparison.get('final_answer', 'N/A')}")
    print("=" * 60)

    # 8) 결과 JSON 저장 (레거시 포맷)
    result = {
        "init_url": init_url,
        "viewport-width": args.viewport_width,
        "viewport-height": args.viewport_height,
        "mode": mode,
        "prev_result_path": args.prev_result,
        "context": context.to_prompt_string(),
        "key_details": context.key_details,
        "comparison_target_info": context.comparison_target_info if mode == "comparison" else {},
        "final_comparison": final_comparison,
        "actions": all_actions,
    }
    with open(os.path.join(run_dir, "result.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"\nAll outputs saved to: {run_dir}")

    # 9) 정리
    browser_env.close()
    logging.info("Done.")
