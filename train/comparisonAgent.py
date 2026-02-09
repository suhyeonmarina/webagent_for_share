from traj_gen.llm_utils import call_gpt4v
import json

class ComparisonAgent:
    def compare(self, results: List[Dict]):
        """
        LLM을 사용하여 여러 웹사이트의 결과를 비교
        - 가격 비교
        - 리뷰 점수 비교
        - 배송 옵션 비교 등
        """
        screenshots = [r['final_screenshot'] for r in results]
        extracted_info = [r['extracted_info'] for r in results]
        
        prompt = f"""
        Compare these results from different websites:
        {json.dumps(extracted_info, indent=2)}
        
        Determine which option is best based on:
        - Price
        - Rating
        - Availability
        """
        
        comparison = call_gpt4v(self.args, messages=[...])
        return comparison
    
    def act_with_template(self, acc_tree, image_obs, task_template):
        # 기존 act() 메서드를 수정하여 task_template를 프롬프트에 포함
        prompt = f"""
        Given this task template: "{task_template}"
        Adapt it to the current website and generate the first action.
        """