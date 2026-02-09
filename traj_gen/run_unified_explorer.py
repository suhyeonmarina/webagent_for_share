"""
Unified Explorer 실행 예시

사용법:
    # Sequential 모드 (아파트 찾기 → 출퇴근 시간 확인 → 이사 견적)
    python -m traj_gen.run_unified_explorer --mode sequential

    # Comparison 모드 (여러 사이트에서 같은 제품 가격 비교)
    python -m traj_gen.run_unified_explorer --mode comparison
"""

import os
import argparse
from datetime import datetime

from .unified_explorer import UnifiedExplorer, setup_logging


def run_sequential_example(args):
    """Sequential 모드 예시: 아파트 찾기 여정"""

    explorer = UnifiedExplorer(args)

    result = explorer.run(
        high_level_goal="Find a 2-bedroom pet-friendly apartment in Austin, TX under $1800/month and prepare for the move",
        websites=[
            "https://www.zillow.com/homes/for_rent/",
            "https://www.google.com/maps",
            "https://www.uhaul.com/",
        ],
        mode="sequential",
        output_dir=args.output_dir,
    )

    return result


def run_comparison_example(args):
    """Comparison 모드 예시: 노트북 가격 비교"""

    explorer = UnifiedExplorer(args)

    result = explorer.run(
        high_level_goal="Find the best deal on MacBook Pro 16-inch laptop",
        websites=[
            "https://www.amazon.com/",
            "https://www.bestbuy.com/",
            "https://www.walmart.com/",
        ],
        mode="comparison",
        output_dir=args.output_dir,
        comparison_criteria=["price", "shipping", "availability", "seller_rating"]
    )

    return result


def run_custom(args):
    """커스텀 모드: CLI 인자로 받은 설정 사용"""

    explorer = UnifiedExplorer(args)

    result = explorer.run(
        high_level_goal=args.goal,
        websites=args.websites,
        mode=args.mode,
        output_dir=args.output_dir,
        comparison_criteria=args.comparison_criteria if args.mode == "comparison" else None
    )

    return result


def main():
    parser = argparse.ArgumentParser(description="Unified Explorer Examples")

    # 실행 모드
    parser.add_argument(
        "--example",
        type=str,
        choices=["sequential", "comparison", "custom"],
        default="custom",
        help="Run predefined example or custom"
    )

    # Custom 모드용 인자
    parser.add_argument("--goal", type=str, help="High-level goal (for custom mode)")
    parser.add_argument("--websites", type=str, nargs="+", help="Websites to explore (for custom mode)")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["sequential", "comparison"],
        default="sequential",
        help="Exploration mode"
    )
    parser.add_argument(
        "--comparison-criteria",
        type=str,
        nargs="*",
        default=["price", "rating"],
        help="Criteria for comparison mode"
    )

    # 공통 설정
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory")
    parser.add_argument("--max-steps", type=int, default=10, help="Max steps per site")
    parser.add_argument("--viewport-width", type=int, default=1280)
    parser.add_argument("--viewport-height", type=int, default=720)
    parser.add_argument("--deployment", type=str, default="gpt-4o", help="LLM model to use")

    args = parser.parse_args()

    # Set default output dir with timestamp
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = f"./unified_output_{args.example}_{timestamp}"

    os.makedirs(args.output_dir, exist_ok=True)
    setup_logging(args.output_dir)

    print(f"Output directory: {args.output_dir}")
    print(f"Mode: {args.example}")

    if args.example == "sequential":
        result = run_sequential_example(args)
    elif args.example == "comparison":
        result = run_comparison_example(args)
    else:
        if not args.goal or not args.websites:
            print("Error: --goal and --websites are required for custom mode")
            print("Example:")
            print('  python -m traj_gen.run_unified_explorer --example custom \\')
            print('    --goal "Find best laptop under $1000" \\')
            print('    --websites amazon.com bestbuy.com \\')
            print('    --mode comparison')
            return

        result = run_custom(args)

    print("\n" + "=" * 60)
    print("EXPLORATION COMPLETE")
    print("=" * 60)

    if result.get("task_type") == "comparison":
        print(f"\nComparison Target: {result.get('comparison_target', 'N/A')}")
        print("\nResults by site:")
        for site_result in result.get("site_results", []):
            print(f"\n  {site_result['website']}:")
            for k, v in site_result.get("extracted_info", {}).items():
                print(f"    - {k}: {v}")
    else:
        print(f"\nAccumulated Information:")
        for k, v in result.get("accumulated_info", {}).items():
            print(f"  - {k}: {v}")

        print("\nTask progression:")
        for i, site_result in enumerate(result.get("site_results", [])):
            print(f"  {i+1}. {site_result['website']}: {site_result.get('summary', 'completed')}")

    print(f"\nFull results saved to: {args.output_dir}/unified_result.json")


if __name__ == "__main__":
    main()
