import argparse
import json
import logging
import os
import random
import re
import traceback

from PIL import Image

from ..browser_env import ScriptBrowserEnv
from .captcha_detection_agent import CaptchaDetectionAgent
from ..processors import ImageObservationProcessor
from .task_proposal_agent import TaskProposalAgent
from .task_refiner_agent import TaskRefinerAgent
from .task_summarization_flow import TaskSummarizationAgent
from .trajectory_verifier import TrajectoryVerifierAgent

logger = logging.getLogger(__name__)


class Explorer:
    def __init__(self, args):
        self.args = args
        self.viewport_size = {
            "width": args.viewport_width,
            "height": args.viewport_height,
        }
        self.image_observation_type = "image_som"

        self.browser_env = ScriptBrowserEnv(
            args, browser_type="chrome", viewport_size=self.viewport_size
        )

        self.init_setup_error = False
        try:
            self.browser_env.setup(args.init_url)
        except Exception:
            self.init_setup_error = True
            logger.info("Error in setting up the environment. Exiting...")
            logger.info(traceback.format_exc())
            return

        # ImageObservationProcessor: 이미지 위에 각 요소에 숫자 ID를 부여
        self.image_processor = ImageObservationProcessor(
            args, self.image_observation_type, self.viewport_size
        )

        self.task_proposal_agent = TaskProposalAgent(
            args, self.browser_env, self.image_processor
        )
        self.task_refiner_agent = TaskRefinerAgent(
            args, self.browser_env, self.image_processor
        )
        self.summarization_agent = TaskSummarizationAgent(
            args, self.browser_env, self.image_processor
        )
        self.verifier_agent = TrajectoryVerifierAgent(args)
        self.captcha_detection_agent = CaptchaDetectionAgent(args)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def get_state(self):
        som_image_obs, parsed_html_str = self.image_processor.process_new(
            self.browser_env.page,
            self.browser_env.page.client,
            use_id_selector=True,
            intent=None,
        )
        html = self.browser_env.page.content()

        return {
            "page": self.browser_env.page,
            "client": self.browser_env.page.client,
            "content_str": parsed_html_str,  # 간이 접근성 트리 ([ID], [Tag], [aria-name])
            "image_obs": som_image_obs,       # SoM 스크린샷 (ID 라벨 포함)
            "html": html,
        }

    def _save_step_artifacts(self, ex_log_dir, step, state):
        """HTML, 스크린샷, SoM 이미지를 디스크에 저장한다."""
        html_path = os.path.join(ex_log_dir, f"html_{step}.html")
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(state["html"])

        if not self.args.no_dump_screenshots:
            self.browser_env.page.screenshot(
                path=os.path.join(ex_log_dir, f"screenshot_{step}.png")
            )
            img = Image.fromarray(state["image_obs"])
            img.save(os.path.join(ex_log_dir, f"screenshot_som_{step}.png"))

    def _extract_bounding_box(self, grounded_action):
        """grounded_action 에서 [ID] 를 파싱해 bbox 좌표를 반환한다."""
        try:
            match = re.search(r"\[(\d+)\]", grounded_action)
            element_id = match.group(1)
            info = self.image_processor.som_id_info[element_id]
            return {"x": info[0], "y": info[1], "width": info[2], "height": info[3]}
        except Exception:  # scroll 등 element ID 가 없는 액션
            return None

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self, ex_log_dir="."):
        if self.init_setup_error:
            return [], "Error in setting up the environment", False

        task_trajectory_data = {
            "init_url": self.args.init_url,
            "viewport-width": self.args.viewport_width,
            "viewport-height": self.args.viewport_height,
            "actions": [],
        }

        completed = False
        task_refinement_history = []
        action_history = []
        refined_goal = None
        step = 0
        execution_id = 0  # 환경 리셋 / 재시도 횟수

        try:
            while step < self.args.max_steps and execution_id <= 2:
                if completed:
                    break

                action = {}
                logger.info(f"Step {step}:")

                # ── 1. 페이지 상태 획득 ──────────────────────────────
                if self.browser_env.page is not None:
                    try:
                        self.browser_env.page.wait_for_load_state(
                            "networkidle", timeout=3000
                        )
                    except Exception:
                        pass

                    try:
                        browser_env_state = self.get_state()
                    except Exception:
                        logger.info("Error in getting state, resetting the environment...")
                        logger.info(traceback.format_exc())
                        self.browser_env.setup(self.args.init_url)
                        task_trajectory_data["actions"] = []
                        task_refinement_history = []
                        action_history = []
                        step = 0
                        execution_id += 1
                        continue

                    if self.args.print_parsed_tree:
                        logger.info(f"acc_tree = {browser_env_state['content_str']}")

                    action["acc_tree_before"] = browser_env_state["content_str"]
                    self._save_step_artifacts(ex_log_dir, step, browser_env_state)
                else:
                    browser_env_state = None

                # ── 2. Captcha 감지 (첫 스텝만) ──────────────────────
                if step == 0:
                    captcha_response = self.captcha_detection_agent.act(
                        os.path.join(ex_log_dir, f"screenshot_{step}.png")
                    )
                    logger.info(f"captcha_response = {captcha_response}")

                    is_captcha = captcha_response.split("Answer:")[-1].strip().lower()
                    if is_captcha == "yes":
                        logger.info("Captcha detected. Terminating the traj.")
                        return [], "Captcha detected", False

                # ── 3. 에이전트 행동 결정 ─────────────────────────────
                if step == 0:
                    response, pred, is_action_valid = self.task_proposal_agent.act(
                        browser_env_state["content_str"],
                        browser_env_state["image_obs"],
                    )
                else:
                    response, pred, is_action_valid = self.task_refiner_agent.act(
                        browser_env_state["content_str"],
                        browser_env_state["image_obs"],
                        action_history,
                        refined_goal,
                    )

                logger.info(f"pred = {pred}")

                new_action_nl = pred["action_in_natural_language"]
                new_action_grounded = pred["grounded_action"]
                refined_goal = pred["task"]

                bounding_box_coord = self._extract_bounding_box(new_action_grounded)

                logger.info(f"Agent response: {response}")
                logger.info(f"Action (NL): {new_action_nl}")
                logger.info(f"Action (grounded): {new_action_grounded}")
                logger.info(f"refined_goal: {refined_goal}")

                action["step_action_nl"] = new_action_nl
                action["new_action_grounded"] = new_action_grounded
                action["bounding_box_coord"] = bounding_box_coord
                action["step_refined_goal"] = refined_goal
                action["step_reasoning_response"] = response

                task_refinement_history.append(refined_goal)
                action_history.append(new_action_nl)

                # ── 4. 액션 실행 ──────────────────────────────────────
                if new_action_grounded == "stop":
                    completed = True
                    break

                logger.info(f"URL: {self.browser_env.page.url}")

                if is_action_valid:
                    action["URL_after"] = self.browser_env.page.url
                    task_trajectory_data["actions"].append(action)

                logger.info("##############################\n")
                step += 1

        except Exception:
            logger.error(f"Error in step {step}")
            logger.error(traceback.format_exc())
            step += 1

        # ── 5. 요약 및 검증 ───────────────────────────────────────────
        screenshot_history = [
            os.path.join(ex_log_dir, f"screenshot_som_{i}.png")
            for i in range(step + 1)
        ]
        summarization_response, summarization_pred = self.summarization_agent.act(
            action_history, screenshot_history
        )

        user_intent = summarization_pred
        history = [a["step_action_nl"] for a in task_trajectory_data["actions"]]
        img_path = os.path.join(ex_log_dir, "screenshot_final.png")

        logger.info(f"user_intent = {user_intent}")
        logger.info(f"history = {history}")

        self.browser_env.page.screenshot(path=img_path)
        self.browser_env.close()

        try:
            response.raise_for_status()
            last_page_md = response.content.decode("utf-8")
        except Exception:
            last_page_md = None

        if self.args.use_all_screenshots_verifier:
            all_screenshots = [
                os.path.join(ex_log_dir, f"screenshot_{i}.png")
                for i in range(step + 1)
            ] + [img_path]
            verifier_agent_response = self.verifier_agent.act(
                user_intent, history, all_screenshots, last_page_md
            )
        else:
            verifier_agent_response = self.verifier_agent.act(
                user_intent, history, img_path, last_page_md
            )

        logger.info(f"verifier_agent_response = {verifier_agent_response}")

        task_trajectory_data["task_summary"] = user_intent
        task_trajectory_data["verifier_agent_response"] = verifier_agent_response

        return task_trajectory_data, verifier_agent_response, True


def to_raw_string(s):
    return s.replace("\\", "\\\\")


def setup_logging(ex_log_dir):
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    log_file = os.path.join(ex_log_dir, "step_simulator_flow.log")
    logging.basicConfig(
        level=logging.INFO,
        filename=log_file,
        filemode="w",
        format="%(asctime)s - %(message)s",
    )


def main(args):
    random.seed(args.seed)

    if args.model_dir is None:
        args.model_dir = f"model_{random.randint(0, 1_000_000)}"

    os.makedirs(args.model_dir, exist_ok=True)

    flow = Explorer(args)
    setup_logging(args.model_dir)

    task_trajectory_data, verifier_agent_response, is_traj_success = flow.run(
        args.model_dir
    )

    if not is_traj_success:
        return

    output_path = os.path.join(args.model_dir, "task_trajectory_data.json")
    with open(output_path, "w") as f:
        json.dump(task_trajectory_data, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--max-steps", type=int, default=5,
        help="Maximum number of steps to simulate",
    )
    parser.add_argument(
        "--print-parsed-tree", action="store_true",
        help="Print the parsed tree in stdout",
    )
    parser.add_argument(
        "--no-dump-screenshots", action="store_true",
        help="Do NOT dump screenshots of each step",
    )
    parser.add_argument(
        "--model-dir", type=str, default=None,
        help="Directory to save the models",
    )
    parser.add_argument("--seed", type=int, default=736537, help="Random seed")
    parser.add_argument(
        "--init-url", type=str, default="https://www.amazon.com/",
        help="Initial URL for the browser env",
    )
    parser.add_argument(
        "--omit-acc-tree", action="store_true",
        help="Omit the accessibility tree",
    )
    parser.add_argument("--viewport-width", type=int, default=1280, help="Viewport width")
    parser.add_argument("--viewport-height", type=int, default=720, help="Viewport height")
    parser.add_argument(
        "--print-num-toks", action="store_true", default=False,
        help="Print the token count for each module",
    )
    parser.add_argument(
        "--deployment", type=str, default="gpt-5-mini",
        choices=["gpt-4o", "gpt-4o-mini", "gpt-5-mini"],
        help="API model deployment",
    )
    parser.add_argument(
        "--use-all-screenshots-verifier", action="store_true", default=True,
        help="Use all screenshots for verifier",
    )

    args = parser.parse_args()
    print(args)
    main(args)
