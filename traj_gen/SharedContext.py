from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field

@dataclass
class SharedContext:
    """사이트 간 공유되는 컨텍스트"""
    high_level_goal: str  # "Austin에서 2BR 아파트 찾고 이사 준비"
    mode: str  # "sequential" or "comparison"

    # Comparison 모드용
    comparison_target: Optional[str] = None  # "MacBook Pro 16인치 M3 Max"
    comparison_criteria: List[str] = field(default_factory=list)  # ["price"]
    comparison_websites: List[str] = field(default_factory=list)  # ["https://amazon.com", "https://bestbuy.com"]
    comparison_target_info: Dict[str, Any] = field(default_factory=dict)  # {"Amazon": "$299", "BestBuy": "$279"}

    # Sequential 모드용
    sequential_websites: List[str] = field(default_factory=list)  # ["https://zillow.com", "https://maps.google.com", "https://uhaul.com"]
    dependency: Optional[str] = None  # 다음 사이트가 depend하는 정보 (e.g., "apartment address found on Zillow")
    key_details: Dict[str, Any] = field(default_factory=dict)  # 이전 사이트에서 추출한 정보

    # 공통
    sub_task: Optional[str] = None  # 현재 사이트에서 수행할 세부 태스크
    #NOTE : summary agent 필요한지 보고 삭제하든가 말든가 해야됨
    results: List[Dict] = field(default_factory=list)  # 각 사이트 탐색 결과 => 나중에 sumamry agent가 사용함, 필요할 진 모르겠음
    current_site_index: int = 0

    def to_prompt_string(self) -> str:
        """LLM 프롬프트에 포함할 컨텍스트 문자열 생성"""
        lines = []
        if self.high_level_goal:
            lines.append(f"High-level Goal: {self.high_level_goal}")
        elif self.mode == "sequential":
            lines.append(
                "High-level Goal: (Not specified — propose a multi-step workflow "
                "where each site's result is a prerequisite for the next site's task. "
                "Example: 'Plan a move to Austin: find an apartment on Zillow → check utility setup on Austin Energy → book movers on U-Haul'. "
                "The goal must involve a CAUSAL CHAIN, NOT comparing the same item across sites.)"
            )
        elif self.mode == "comparison":
            lines.append(
                "High-level Goal: (Not specified — propose a CROSS-SITE comparison task. "
                "The goal must involve comparing the SAME item/service across DIFFERENT websites. "
                "Example: 'Compare prices of Sony WH-1000XM5 on Amazon vs Best Buy vs Walmart'. "
                "Do NOT propose a goal that compares multiple items within a single website.)"
            )
        else:
            lines.append("High-level Goal: (Not specified — propose a realistic task based on the current page)")

        if self.sub_task:
            lines.append(f"Sub-task: {self.sub_task}")

        if self.mode == "comparison":
            if self.comparison_target:
                lines.append(f"Comparison Target: {self.comparison_target}")
            else:
                lines.append("Comparison Target: (Not specified — propose a specific target consistent with the goal)")
            if self.comparison_criteria:
                lines.append(f"Compare by: {', '.join(self.comparison_criteria)}")
            else:
                lines.append("Compare by: (Not specified — propose relevant comparison criteria based on the goal and website)")
            if self.comparison_websites:
                lines.append(f"Websites to compare: {', '.join(self.comparison_websites)}")
            if self.comparison_target_info:
                lines.append("\nComparison info gathered so far:")
                for site, info in self.comparison_target_info.items():
                    lines.append(f"  - {site}: {info}")
            if self.results:
                lines.append("\nPrevious site results:")
                for r in self.results:
                    lines.append(f"  - {r.get('website', 'Unknown')}: {r.get('extracted_info', {})}")

        elif self.mode == "sequential":
            if self.sequential_websites:
                lines.append(f"Websites in sequence: {', '.join(self.sequential_websites)}")
            if self.dependency:
                lines.append(f"Dependency: {self.dependency}")
            else:
                lines.append("Dependency: (Not specified — propose a dependency consistent with the high-level goal)")
            if self.key_details:
                lines.append(f"\nInformation gathered so far:")
                for k, v in self.key_details.items():
                    lines.append(f"  - {k}: {v}")

        return "\n".join(lines)

    def update_from_prediction(self, pred: Dict[str, Any]) -> None:
        import logging

        # 공통
        if not self.high_level_goal and pred.get("high_level_goal"):
            self.high_level_goal = pred["high_level_goal"]
            logging.info(f"Agent proposed goal: {self.high_level_goal}")

        if pred.get("sub_task"):
            self.sub_task = pred["sub_task"]
            logging.info(f"Agent proposed sub_task: {self.sub_task}")

        # Comparison 모드
        if self.mode == "comparison":
            if not self.comparison_target and pred.get("comparison_target"):
                self.comparison_target = pred["comparison_target"]
                logging.info(f"Agent proposed target: {self.comparison_target}")

            if not self.comparison_criteria and pred.get("comparison_criteria"):
                self.comparison_criteria = pred["comparison_criteria"]
                logging.info(f"Agent proposed criteria: {self.comparison_criteria}")

            if not self.comparison_websites and pred.get("comparison_websites"):
                self.comparison_websites = pred["comparison_websites"]
                logging.info(f"Agent proposed websites: {self.comparison_websites}")

            if pred.get("comparison_target_info") and isinstance(pred["comparison_target_info"], dict):
                self.comparison_target_info.update(pred["comparison_target_info"])
                logging.info(f"Updated comparison_target_info: {self.comparison_target_info}")

        # Sequential 모드
        if self.mode == "sequential":
            if not self.sequential_websites and pred.get("sequential_websites"):
                self.sequential_websites = pred["sequential_websites"]
                logging.info(f"Agent proposed sequential_websites: {self.sequential_websites}")

            if not self.dependency and pred.get("dependency"):
                self.dependency = pred["dependency"]
                logging.info(f"Agent proposed dependency: {self.dependency}")

            if pred.get("key_details"):
                self.key_details.update(pred["key_details"])
                logging.info(f"Updated key_details: {self.key_details}")
