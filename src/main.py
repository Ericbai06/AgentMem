import argparse
import sys
import os

# 将当前目录加入路径，防止模块导入报错
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modules.ingestor import MemoryIngestor
from modules.agent import LocomoAgent

def main():
    parser = argparse.ArgumentParser(description="LOCOMO Agent with Context Engineering")
    parser.add_argument("--step", choices=["add", "search"], required=True, help="add: Ingest memories; search: Run QA evaluation")
    
    args = parser.parse_args()

    if args.step == "add":
        ingestor = MemoryIngestor()
        ingestor.run()
    elif args.step == "search":
        agent = LocomoAgent()
        agent.run_eval()

if __name__ == "__main__":
    main()