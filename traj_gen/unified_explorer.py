import os
import re
import copy
import json
import logging
import traceback
from typing import List, Dict
from PIL import Image

from .browser_env import ScriptBrowserEnv
from .processors import ImageObservationProcessor
from .original_agents.captcha_detection_agent import CaptchaDetectionAgent
from .SharedContext import SharedContext
from .InfoExtractor import InfoExtractor
from .exploration_proposal_agent import ExplorationProposalAgent
from .exploration_refiner_agent import ExplorationRefinerAgent
from .exploration_verifier_agent import ExplorationVerifierAgent


class UnifiedExplorer:

    def __init__(self, args):
        self.args = args
        self.viewport_size = {
            "width": args.viewport_width,
            "height": args.viewport_height,
        }

    def run(self, high_level_goal: str, websites: List[str], mode: str = "sequential",
            output_dir: str = "unified_explorer", comparison_criteria: List[str] = None,
            dependency: str = None) -> Dict:
                
        # Initialize shared context
        context = SharedContext(
            high_level_goal=high_level_goal or "",
            mode=mode,
            comparison_criteria=comparison_criteria or [],
            dependency=dependency,
        )

        all_trajectories = []

        for i, website_url in enumerate(websites):
            context.current_site_index = i
            logging.info(f"\n{'='*50}")
            logging.info(f"Exploring site {i+1}/{len(websites)}: {website_url}")
            logging.info(f"{'='*50}")

            # Create site-specific output directory
            site_output_dir = os.path.join(output_dir, f"site_{i}_{self._sanitize_url(website_url)}")
            os.makedirs(site_output_dir, exist_ok=True)

            # Explore this site
            trajectory_data, success = self._explore_single_site(
                website_url, context, site_output_dir
            )

            if not success:
                logging.warning(f"Failed to explore {website_url}")
                continue

            # Verify trajectory
            verifier = ExplorationVerifierAgent(self.args)
            screenshot_paths = [os.path.join(site_output_dir, "screenshot_final.png")]
            verification = verifier.verify(trajectory_data, context, screenshot_paths)

            trajectory_data["verification"] = verification
            logging.info(f"Verification result for {website_url}: {verification.get('overall_status', 'unknown')}")

            # Extract information for next site
            info_extractor = InfoExtractor(self.args)
            final_screenshot = os.path.join(site_output_dir, "screenshot_final.png")
            extracted = info_extractor.extract(trajectory_data, context, final_screenshot)

            # Update context based on mode
            if mode == "comparison":
                # 첫 사이트에서 comparison target 설정
                if i == 0 and extracted.get("comparison_target"):
                    context.comparison_target = extracted["comparison_target"]

                context.results.append({
                    "website": website_url,
                    "extracted_info": extracted.get("extracted_info", {}),
                    "search_query": extracted.get("search_query_used", ""),
                    "summary": extracted.get("summary", "")
                })

            else:  # sequential
                # 추출된 정보를 key_details에 병합
                context.key_details.update(extracted.get("extracted_info", {}))
                context.results.append({
                    "website": website_url,
                    "task_accomplished": trajectory_data.get("task_summary", ""),
                    "extracted_info": extracted.get("extracted_info", {}),
                    "summary": extracted.get("summary", "")
                })

            all_trajectories.append({
                "website": website_url,
                "trajectory": trajectory_data,
                "extracted": extracted
            })

            logging.info(f"Extracted from {website_url}: {extracted}")

        # Generate final summary
        final_result = self._generate_final_summary(context, all_trajectories)

        # Save final result
        with open(os.path.join(output_dir, "unified_result.json"), "w") as f:
            json.dump(final_result, f, indent=2, ensure_ascii=False)

        return final_result

    def _explore_single_site(self, url: str, context: SharedContext,
                             output_dir: str) -> tuple[Dict, bool]:
        """단일 사이트 탐색 (기존 Explorer.run()과 유사)"""

        # Ensure URL has protocol
        if not url.startswith("http"):
            url = "https://" + url

        # Setup browser
        args_copy = copy.deepcopy(self.args)
        args_copy.init_url = url

        browser_env = ScriptBrowserEnv(
            args_copy, browser_type="chrome", viewport_size=self.viewport_size
        )

        try:
            browser_env.setup(url)
        except Exception as e:
            logging.error(f"Failed to setup browser for {url}: {e}")
            return {}, False

        image_processor = ImageObservationProcessor(
            args_copy, "image_som", self.viewport_size
        )

        # Initialize agents
        proposal_agent = ExplorationProposalAgent(args_copy, browser_env, image_processor)
        refiner_agent = ExplorationRefinerAgent(args_copy, browser_env, image_processor)
        captcha_agent = CaptchaDetectionAgent(args_copy)

        # Trajectory data
        trajectory_data = {
            "init_url": url,
            "viewport-width": self.args.viewport_width,
            "viewport-height": self.args.viewport_height,
            "context_at_start": context.to_prompt_string(),
            "actions": []
        }

        action_history = []
        current_task = ""
        step = 0
        completed = False

        try:
            while step < self.args.max_steps and not completed:
                logging.info(f"\nStep {step}:")

                # Wait for page load
                try:
                    browser_env.page.wait_for_load_state("networkidle", timeout=3000)
                except:
                    pass

                # Get current state
                try:
                    som_image_obs, parsed_html_str = image_processor.process_new(
                        browser_env.page, browser_env.page.client,
                        use_id_selector=True, intent=None
                    )
                except Exception as e:
                    logging.error(f"Failed to get state: {e}")
                    break

                # Save screenshots
                browser_env.page.screenshot(
                    path=os.path.join(output_dir, f"screenshot_{step}.png")
                )
                img = Image.fromarray(som_image_obs)
                img.save(os.path.join(output_dir, f"screenshot_som_{step}.png"))

                # Check for CAPTCHA on first step
                if step == 0:
                    captcha_response = captcha_agent.act(
                        os.path.join(output_dir, f"screenshot_{step}.png")
                    )
                    if "yes" in captcha_response.lower():
                        logging.warning("CAPTCHA detected, terminating")
                        browser_env.close()
                        return trajectory_data, False

                # Generate action
                if step == 0:
                    response, pred, is_valid = proposal_agent.act(
                        parsed_html_str, som_image_obs, context
                    )
                else:
                    response, pred, is_valid = refiner_agent.act(
                        parsed_html_str, som_image_obs, action_history, current_task, context
                    )

                if not is_valid:
                    logging.warning("Invalid action, stopping")
                    break

                # Extract action info
                new_action_nl = pred.get("action_in_natural_language", "")
                new_action_grounded = pred.get("grounded_action", "")
                current_task = pred.get("task", current_task)

                logging.info(f"Task: {current_task}")
                logging.info(f"Action (NL): {new_action_nl}")
                logging.info(f"Action (grounded): {new_action_grounded}")

                # Get bounding box
                try:
                    match = re.search(r"\[(\d+)\]", new_action_grounded)
                    if match:
                        element_id = match.group(1)
                        som_id_info = image_processor.som_id_info
                        bounding_box = {
                            "x": som_id_info[element_id][0],
                            "y": som_id_info[element_id][1],
                            "width": som_id_info[element_id][2],
                            "height": som_id_info[element_id][3],
                        }
                    else:
                        bounding_box = None
                except:
                    bounding_box = None

                # Record action
                action_record = {
                    "step": step,
                    "step_action_nl": new_action_nl,
                    "new_action_grounded": new_action_grounded,
                    "bounding_box_coord": bounding_box,
                    "step_refined_goal": current_task,
                    "step_reasoning_response": response,
                    "acc_tree_before": parsed_html_str,
                    "URL_after": browser_env.page.url
                }
                trajectory_data["actions"].append(action_record)
                action_history.append(new_action_nl)

                # Check for stop
                if new_action_grounded.strip().lower() == "stop":
                    completed = True
                    break

                step += 1

        except Exception as e:
            logging.error(f"Error during exploration: {e}")
            logging.error(traceback.format_exc())

        # Save final screenshot
        try:
            browser_env.page.screenshot(
                path=os.path.join(output_dir, "screenshot_final.png")
            )
        except:
            pass

        trajectory_data["task_summary"] = current_task
        trajectory_data["completed"] = completed
        trajectory_data["total_steps"] = step

        browser_env.close()

        return trajectory_data, True

    def _sanitize_url(self, url: str) -> str:
        """URL을 파일명으로 사용 가능하게 변환"""
        return re.sub(r'[^\w\-.]', '_', url.replace("https://", "").replace("http://", ""))

    def _generate_final_summary(self, context: SharedContext,
                                trajectories: List[Dict]) -> Dict:
        """최종 결과 요약 생성"""

        if context.mode == "comparison":
            return {
                "task_type": "comparison",
                "high_level_goal": context.high_level_goal,
                "comparison_target": context.comparison_target,
                "comparison_criteria": context.comparison_criteria,
                "site_results": context.results,
                "trajectories": trajectories,
                "summary": self._generate_comparison_summary(context)
            }
        else:
            return {
                "task_type": "sequential",
                "high_level_goal": context.high_level_goal,
                "accumulated_info": context.key_details,
                "site_results": context.results,
                "trajectories": trajectories,
                "summary": self._generate_sequential_summary(context)
            }

    def _generate_comparison_summary(self, context: SharedContext) -> str:
        """비교 결과 요약"""
        if not context.results:
            return "No comparison results available"

        summaries = []
        for r in context.results:
            info = r.get("extracted_info", {})
            summaries.append(f"{r['website']}: {info}")

        return f"Compared '{context.comparison_target}' across {len(context.results)} sites:\n" + "\n".join(summaries)

    def _generate_sequential_summary(self, context: SharedContext) -> str:
        """순차 작업 결과 요약"""
        if not context.results:
            return "No sequential results available"

        steps = []
        for i, r in enumerate(context.results):
            steps.append(f"{i+1}. {r['website']}: {r.get('summary', 'completed')}")

        return f"Completed {len(context.results)} sequential tasks:\n" + "\n".join(steps)


def setup_logging(output_dir: str):
    """로깅 설정"""
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    log_file = os.path.join(output_dir, "unified_explorer.log")
    logging.basicConfig(
        level=logging.INFO,
        filename=log_file,
        filemode="w",
        format="%(asctime)s - %(message)s",
    )


def main():
    """CLI 엔트리포인트"""
    import argparse

    parser = argparse.ArgumentParser(description="Unified Explorer - Exploration-First Approach")
    parser.add_argument("--goal", type=str, help="High-level goal")
    parser.add_argument("--websites", type=str, nargs="+", required=True, help="Websites to explore")
    parser.add_argument("--mode", type=str, choices=["sequential", "comparison"], default="sequential")
    parser.add_argument("--output-dir", type=str, default="./unified_output")
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--viewport-width", type=int, default=1280)
    parser.add_argument("--viewport-height", type=int, default=720)
    parser.add_argument("--deployment", type=str, default="gpt-4o")
    parser.add_argument("--comparison-criteria", type=str, nargs="*", default=None,
                        help="Comparison criteria (미지정 시 agent가 자동 제안)")
    parser.add_argument("--dependency", type=str, default=None,
                        help="Dependency for sequential mode (미지정 시 agent가 자동 제안)")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    setup_logging(args.output_dir)

    explorer = UnifiedExplorer(args)
    result = explorer.run(
        high_level_goal=args.goal,
        websites=args.websites,
        mode=args.mode,
        output_dir=args.output_dir,
        comparison_criteria=args.comparison_criteria,
        dependency=args.dependency,
    )

    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
