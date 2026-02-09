import re
import json
import logging
from PIL import Image

from .base_exploration_agent import BaseExplorationAgent
from .SharedContext import SharedContext


class ExplorationProposalAgent(BaseExplorationAgent):

    def _get_system_prompt_template(self) -> str:
        return """You are exploring a website. You may have context from previous exploration steps (if any).

{context}

Current Website: {current_url}

Based on the webpage screenshot and parsed HTML/accessibility tree, do the following:
1. If no high-level goal is provided, first propose a realistic and specific high-level goal that a real user would want to accomplish across multiple websites. The goal should be concrete and specific, NOT generic like "search for a product".
{mode_specific_steps}
2. Determine what specific sub-task you should accomplish on THIS website to contribute to the high-level goal.
3. Generate the first action towards completing that sub-task.
4. Ground the action to a specific element on the page.

{mode_specific_instructions}

*ACTION SPACE*: Your action space is: [`click [element ID]`, `type [element ID] [content]`, `select [element ID] [content of option to select]`, `scroll [up]`, `scroll [down]`, and `stop`].
Action output should follow the syntax as given below:
`click [element ID]`: This action clicks on an element with a specific id on the webpage.
`type [element ID] [content]`: Use this to type the content into the field with id. By default, the "Enter" key is pressed after typing. Both the content and the id should be within square braces as per the syntax.
`select [element ID] [content of option to select]`: Select an option from a dropdown menu. The content of the option to select should be within square braces. When you get (select and option) tags from the accessibility tree , you need to select the serial number (element_id) corresponding to the select tag , not the option, and select the most likely content corresponding to the option as input.
`scroll [down]`: Scroll the page down.
`scroll [up]`: Scroll the page up.

*IMPORTANT RULES*:
1. Generate a single atomic action at each step.
2. Element ID MUST be a numeric identifier from the accessibility tree.
3. Natural language action must be consistent with grounded action.
4. Issue "stop" if page asks for login or credit card.
5. Do NOT repeat the same action if page is unchanged.
6. No quotation marks in action generation.

{output_format}

"""

    def _get_mode_steps(self, context: SharedContext) -> str:
        if context.mode == "comparison":
            return """1-a. If no high-level goal is provided, propose a goal that involves comparing the SAME item/service ACROSS DIFFERENT WEBSITES. The goal must require visiting at least 2 different websites to gather comparable data.
   - 1st example: "Compare prices of Sony WH-1000XM5 headphones on Amazon vs Best Buy vs Walmart", “Compare the best recommended hotels in Tokyo across Booking.com, Agoda, and Expedia.”
   - 2nd example: "Compare nail salons in San Francisco on Yelp" (this is comparing items WITHIN one site, not across sites)
1-b. If no comparison target is provided, propose a specific comparison target consistent with the cross-site goal (e.g., "Sony WH-1000XM5", "Round-trip flight Seoul to Jeju on March 15").
1-c. If no comparison criteria are provided, propose exactly ONE specific criterion to compare (e.g., "price" or "customer_rating" or "delivery_time"). Pick the single most important criterion for the given goal.
1-d. Propose 1-4 websites (including the current one) that sell/list the same item so results can be compared across them. ALWAYS prefer well-known US/English-language websites (e.g., Amazon.com, BestBuy.com, Walmart.com, Target.com, Booking.com, Expedia.com, Yelp.com). Do NOT use region-specific or non-English sites (e.g., coupang.com, naver.com, uniqlo.com/kr/)."""
        else:  # sequential
            return """1-a. If no high-level goal is provided, propose a multi-step workflow where each website handles a DIFFERENT step and the output of this site feeds into the next site's task. Do NOT propose a goal that compares or looks up the same item across sites.
1-b. If no dependency is provided, propose a specific dependency — the concrete piece of information (e.g., "apartment address", "flight arrival time", "reservation confirmation number") that this site's task will produce and the next site will consume."""

    def _get_mode_instructions(self, context: SharedContext) -> str:
        if context.mode == "comparison":
            return """*COMPARISON MODE INSTRUCTIONS*:
- COMPARISON MODE means comparing the SAME item/service ACROSS DIFFERENT WEBSITES (e.g., checking the price of the same headphones on Amazon, Best Buy, and Walmart).
- This includes comparing multiple items within a single website.
- Your goal is to find information about the comparison target on THIS website, so it can later be compared with the same target found on OTHER websites.
- Extract the specific value for the comparison criteria on this site (e.g., the price, the rating, the availability).
- Use the same or equivalent search query as previous sites for consistency.
- Propose which other websites should be visited to complete the cross-site comparison."""
        else:  # sequential
            return """*SEQUENTIAL MODE INSTRUCTIONS*:
Sequential mode means each site is a DIFFERENT STEP in a causal chain — the output of one site becomes the input for the next.
This is NOT about looking up the same item on multiple sites (that is comparison mode).

Examples of sequential workflows:
  - Autotrader (find a used car listing, get VIN and model) → Carfax (look up accident and maintenance history using that VIN)
  - Expedia (find a flight, get arrival time) → Airbnb (find accommodation near airport for that arrival date) → Google Maps (plan route from airport to accomodation)

Rules:
- The sub-task on THIS site must produce a concrete result (e.g., an address, a date, a price, a name) that the NEXT site's task logically depends on.
- Do NOT propose a task that compares or searches for the same item across sites.
- If previous key_details are provided, USE them as input for this site's task.
- Extract key_details that will be needed as input for the next site."""

    def _get_output_format(self, context: SharedContext) -> str:
        if context.mode == "comparison":
            return """*OUTPUT FORMAT*:
Analyze the screenshot and accessibility tree, then provide your answer in this format:
```{
  "high_level_goal": "<CROSS-SITE COMPARISON GOAL, e.g., Compare prices of X on SiteA vs SiteB vs SiteC>",
  "comparison_target": "<SPECIFIC ITEM/SERVICE TO COMPARE, e.g., Sony WH-1000XM5>",
  "comparison_criteria": ["<SINGLE_CRITERION>"],
  "comparison_websites": ["<URL_1>", "<URL_2>", ...],
  "comparison_target_info": {"<WEBSITE_NAME>": "<OBSERVED_VALUE_FOR_THE_SINGLE_CRITERION_ON_THIS_SITE>"},
  "sub_task": "<SUB_TASK_FOR_THIS_SITE>",
  "action_in_natural_language": "<ACTION_DESCRIPTION>",
  "grounded_action": "<ACTION>"
}```
Note: "comparison_websites" should list 1-4 website URLs (including the current site) where the same item can be compared."""
        else:  # sequential
            return """*OUTPUT FORMAT*:
Analyze the screenshot and accessibility tree, then provide your answer in this format:
```{
  "high_level_goal": "<YOUR_PROPOSED_OR_GIVEN_GOAL>",
  "sub_task": "<SUB_TASK_FOR_THIS_SITE>",
  "sequential_websites": ["<URL_1>", "<URL_2>", ...],
  "dependency": "<KEY_INFO_THE_NEXT_SITE_WILL_NEED_FROM_THIS_TASK>",
  "key_details": {"<KEY>": "<VALUE_RELATED_TO_DEPENDENCY_EXTRACTED_FROM_THIS_SITE>"},
  "action_in_natural_language": "<ACTION_DESCRIPTION>",
  "grounded_action": "<ACTION>"
}```
Note: "dependency" should describe what specific information or result from this site's task will be needed by the next site (e.g., "reservation date", "address", "price"). If nothing is needed, use "none".
"key_details" should contain the concrete values related to the dependency (e.g., if dependency is "apartment address", key_details might be {"apartment_address": "123 Main St, Austin, TX"})."""

    def act(self, acc_tree: str, image_obs, context: SharedContext):
        # 시스템 프롬프트 생성
        system_prompt = self.system_prompt_template.format(
            context=context.to_prompt_string(),
            current_url=self.browser_env.page.url,
            mode_specific_steps=self._get_mode_steps(context),
            mode_specific_instructions=self._get_mode_instructions(context),
            output_format=self._get_output_format(context)
        )

        # LLM 호출
        messages = self._build_messages(system_prompt, acc_tree, image_obs)
        response, is_valid = self._call_llm(messages)
        logging.info(f"Exploration Proposal Agent response = {response}")

        # 응답 처리 및 context 업데이트
        pred, is_action_valid = self._process_response(response, is_valid, context)

        # 액션 실행
        is_action_valid = self._execute_if_valid(pred, is_action_valid)

        return response, pred, is_action_valid


if __name__ == "__main__":
    import argparse
    import os
    from PIL import Image

    from traj_gen.browser_env import ScriptBrowserEnv
    from traj_gen.processors import ImageObservationProcessor
    from traj_gen.SharedContext import SharedContext

    # --- argparse ---
    parser = argparse.ArgumentParser(description="Test ExplorationProposalAgent")
    parser.add_argument("--init-url", type=str, default="https://www.amazon.com/",
                        help="URL to explore")
    parser.add_argument("--deployment", type=str, default="gpt-5-mini",
                        choices=["gpt-4o", "gpt-4o-mini", "gpt-5-mini"],
                        help="LLM model deployment")
    parser.add_argument("--viewport-width", type=int, default=1280)
    parser.add_argument("--viewport-height", type=int, default=720)
    parser.add_argument("--max-steps", type=int, default=5)

    # SharedContext 설정 (--goal 미지정 시 agent가 페이지를 보고 자동 제안)
    parser.add_argument("--mode", type=str, default="comparison",
                        choices=["sequential", "comparison"],
                        help="Exploration mode")
    parser.add_argument("--goal", type=str, default="",
                        help="High-level goal (미지정 시 agent가 페이지 보고 자동 제안)")
    parser.add_argument("--comparison-target", type=str, default=None,
                        help="Comparison target (comparison mode)")
    parser.add_argument("--comparison-criteria", type=str, nargs="*",
                        default=None,
                        help="Comparison criteria (미지정 시 agent가 자동 제안)")
    parser.add_argument("--dependency", type=str, default=None,
                        help="Dependency for sequential mode (미지정 시 agent가 자동 제안)")

    args = parser.parse_args()

    # --- Setup: 출력 폴더 생성 (explore_result/{website}_{timestamp}) ---
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    site_name = re.sub(r'[^\w\-.]', '_', args.init_url.replace("https://", "").replace("http://", ""))
    run_dir = os.path.join(os.getcwd(), "proposal_result", f"{site_name}_{timestamp}")
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

    # 1) Browser 초기화
    logging.info(f"Setting up browser for {args.init_url}")
    browser_env = ScriptBrowserEnv(args, browser_type="chrome", viewport_size=viewport_size)
    browser_env.setup(args.init_url)

    # 2) Image processor 초기화
    image_processor = ImageObservationProcessor(args, "image_som", viewport_size)

    # 3) SharedContext 생성
    context = SharedContext(
        high_level_goal=args.goal,
        mode=args.mode,
        comparison_target=args.comparison_target,
        comparison_criteria=args.comparison_criteria or [],
        dependency=args.dependency,
    )
    logging.info(f"Shared Context:\n{context.to_prompt_string()}")

    # 4) Observation 가져오기
    try:
        browser_env.page.wait_for_load_state("networkidle", timeout=5000)
    except:
        pass

    som_image_obs, parsed_html_str = image_processor.process_new(
        browser_env.page, browser_env.page.client,
        use_id_selector=True, intent=None,
    )

    # 스크린샷 저장
    browser_env.page.screenshot(path=os.path.join(run_dir, "screenshot_before.png"))
    Image.fromarray(som_image_obs).save(os.path.join(run_dir, "screenshot_som.png"))

    # acc_tree 저장
    with open(os.path.join(run_dir, "acc_tree.txt"), "w") as f:
        f.write(parsed_html_str)

    # 5) Agent 실행
    agent = ExplorationProposalAgent(args, browser_env, image_processor)
    response, pred, is_action_valid = agent.act(parsed_html_str, som_image_obs, context)

    # 5-a) bounding_box 추출
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

    # 5-b) URL_after 캡처
    url_after = browser_env.page.url

    # 액션 실행 후 스크린샷
    try:
        browser_env.page.screenshot(path=os.path.join(run_dir, "screenshot_after.png"))
    except:
        pass

    # 6) 레거시 포맷 action record 생성
    action_record = {
        "acc_tree_before": parsed_html_str,
        "step_action_nl": pred.get("action_in_natural_language", ""),
        "new_action_grounded": pred.get("grounded_action", ""),
        "bounding_box_coord": bounding_box,
        "step_refined_goal": pred.get("sub_task", pred.get("task", "")),
        "step_reasoning_response": response,
        "URL_after": url_after,
    }

    # 7) 결과 출력
    print("\n" + "=" * 60)
    print("Exploration Proposal Agent Result")
    print("=" * 60)
    print(f"URL         : {args.init_url}")
    print(f"Mode        : {args.mode}")
    print(f"Goal        : {context.high_level_goal or '(failed to propose)'}")
    print(f"Valid action: {is_action_valid}")
    print("-" * 60)
    print(f"Task        : {action_record['step_refined_goal']}")
    print(f"Action (NL) : {action_record['step_action_nl']}")
    print(f"Action (GT) : {action_record['new_action_grounded']}")
    print(f"BBox        : {bounding_box}")
    print(f"URL After   : {url_after}")
    print("-" * 60)
    print(f"Raw response:\n{response}")
    print("=" * 60)

    # 8) 결과 JSON 저장 (레거시 포맷)
    result = {
        "init_url": args.init_url,
        "viewport-width": args.viewport_width,
        "viewport-height": args.viewport_height,
        "mode": args.mode,
        "context": context.to_prompt_string(),
        "is_action_valid": is_action_valid,
        "pred": pred,
        "actions": [action_record],
    }
    with open(os.path.join(run_dir, "result.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"\nAll outputs saved to: {run_dir}")

    # 8) 정리
    browser_env.close()
    logging.info("Done.")
