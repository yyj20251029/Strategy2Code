from openai import OpenAI
import json
from tqdm import tqdm
import argparse
import os
import sys
from utils import print_response, print_log_cost, load_accumulated_cost, save_accumulated_cost

parser = argparse.ArgumentParser()

parser.add_argument('--paper_name',type=str)
parser.add_argument('--gpt_version',type=str)
parser.add_argument('--paper_format',type=str, default="JSON", choices=["JSON", "LaTeX"])
parser.add_argument('--pdf_json_path', type=str) # json format
parser.add_argument('--pdf_latex_path', type=str) # latex format
parser.add_argument('--output_dir',type=str, default="")

args    = parser.parse_args()

client = OpenAI(api_key = os.environ["OPENAI_API_KEY"])

paper_name = args.paper_name
gpt_version = args.gpt_version
paper_format = args.paper_format
pdf_json_path = args.pdf_json_path
pdf_latex_path = args.pdf_latex_path
output_dir = args.output_dir


if paper_format == "JSON":
    with open(f'{pdf_json_path}') as f:
        paper_content = json.load(f)
elif paper_format == "LaTeX":
    with open(f'{pdf_latex_path}') as f:
        paper_content = f.read()
else:
    print(f"[ERROR] Invalid paper format. Please select either 'JSON' or 'LaTeX.")
    sys.exit(0)

plan_msg = [
        {'role': "system", "content": f"""You are an expert researcher and strategic planner with a deep understanding of experimental design and reproducibility in scientific research. 
You will receive a research paper in {paper_format} format. 
Your task is to create a detailed and efficient plan to reproduce the experiments and methodologies described in the paper.
This plan should align precisely with the paper's methodology, experimental setup, and evaluation metrics. 

Instructions:

1. Align with the Paper: Your plan must strictly follow the methods, datasets, model configurations, hyperparameters, and experimental setups described in the paper.
2. Be Clear and Structured: Present the plan in a well-organized and easy-to-follow format, breaking it down into actionable steps.
3. Prioritize Efficiency: Optimize the plan for clarity and practical implementation while ensuring fidelity to the original experiments."""},
        {"role": "user",
         "content" : f"""## Paper
{paper_content}

## Task
1. We want to reproduce the method described in the attached paper. 
2. The authors did not release any official code, so we have to plan our own implementation.
3. Before writing any Python code, please outline a comprehensive plan that covers:
   - Key details from the paper's **Methodology**.
   - Important aspects of **Experiments**, including dataset requirements, experimental settings, hyperparameters, or evaluation metrics.
4. The plan should be as **detailed and informative** as possible to help us write the final code later.

## Requirements
- You don't need to provide the actual code yet; focus on a **thorough, clear strategy**.
- If something is unclear from the paper, mention it explicitly.

## Instruction
The response should give us a strong roadmap, making it easier to write the code later."""}]

file_list_msg = [
    {
        "role": "user",
        "content": """Your goal is to create a concise, usable, and complete quantitative trading system design for reproducing the paper's strategy. Use appropriate open-source libraries and keep the overall architecture simple.

Based on the plan for reproducing the paper’s main method, please design a concise, usable, and complete software system that:

- downloads historical price data from Yahoo Finance,
- implements the trading strategy described in the paper, and
- runs a backtest over a fixed period from 2000-01-01 to 2024-01-01, reporting key performance metrics such as annualized return, Sharpe ratio, and maximum drawdown.

Important constraints:
- Regardless of the sample period used in the original paper, the actual data range for implementation and backtesting MUST be:
  start_date = 2000-01-01, end_date = 2024-01-01.
- All code, configuration, and backtesting logic MUST be built around this fixed data range.

-----

## Format Example
[CONTENT]
{
    "Implementation approach": "We will use Yahoo Finance to download historical daily price data for the selected instruments between 2000-01-01 and 2024-01-01, construct trading signals according to the paper's methodology, and run a unified backtesting engine that computes standard performance metrics.",
    "File list": [
        "main.py",
        "data/loader_yahoo.py",
        "strategy/strategy.py",
        "backtest/backtester.py",
        "backtest/metrics.py",
        "config.yaml"
    ],
    "Data structures and interfaces": "classDiagram\\n    class Main {\\n        +__init__(config_path: str)\\n        +run() -> None\\n    }\\n    class DataLoader {\\n        +__init__(config: dict)\\n        +load_price_data() -> DataFrame\\n    }\\n    class Strategy {\\n        +__init__(config: dict)\\n        +generate_signals(price_df: DataFrame) -> DataFrame\\n    }\\n    class Backtester {\\n        +__init__(config: dict)\\n        +run_backtest(price_df: DataFrame, signals_df: DataFrame) -> DataFrame\\n    }\\n    class Metrics {\\n        +__init__(config: dict)\\n        +compute(perf_df: DataFrame) -> dict\\n    }\\n    Main --> DataLoader\\n    Main --> Strategy\\n    Main --> Backtester\\n    Main --> Metrics\\n    Backtester --> Strategy\\n    Backtester --> DataLoader\\n",
    "Program call flow": "sequenceDiagram\\n    participant M as Main\\n    participant DL as DataLoader\\n    participant ST as Strategy\\n    participant BT as Backtester\\n    participant MT as Metrics\\n    M->>DL: load_price_data() [2000-01-01, 2024-01-01]\\n    DL-->>M: price_df\\n    M->>ST: generate_signals(price_df)\\n    ST-->>M: signals_df\\n    M->>BT: run_backtest(price_df, signals_df)\\n    BT-->>M: perf_df\\n    M->>MT: compute(perf_df)\\n    MT-->>M: metrics\\n",
    "Anything UNCLEAR": "Need clarification on trading costs, leverage and short-selling constraints, and whether the paper specifies a particular universe of assets that is not fully available on Yahoo Finance."
}
[/CONTENT]

## Nodes: "<node>: <type>  # <instruction>"
- Implementation approach: <class 'str'>  # Summarize the chosen solution strategy, emphasizing Yahoo Finance, the fixed 2000-2024 data range, and a unified backtesting framework.
- File list: typing.List[str]  # Only need relative paths. ALWAYS include main.py and config.yaml, and you MUST include at least: main.py, data/loader_yahoo.py, strategy/strategy.py, backtest/backtester.py, backtest/metrics.py, config.yaml.
- Data structures and interfaces: typing.Optional[str]  # Use mermaid classDiagram code syntax, including classes, methods (__init__ etc.) and functions with type annotations, CLEARLY MARK the relationships between classes, and comply with PEP8 standards. The data structures SHOULD BE VERY DETAILED and the API should be comprehensive with a complete design.
- Program call flow: typing.Optional[str] # Use sequenceDiagram code syntax, COMPLETE and VERY DETAILED, using CLASSES AND API DEFINED ABOVE accurately, covering the initialization and interactions of each object. SYNTAX MUST BE CORRECT.
- Anything UNCLEAR: <class 'str'>  # Mention ambiguities and ask for clarifications (e.g., universe definition, transaction costs, leverage constraints, data availability).

## Constraint
Format: output wrapped inside [CONTENT][/CONTENT] like the format example, nothing else.

## Action
Follow the instructions for the nodes, generate the output, and ensure it follows the format example."""
    }
]


task_list_msg = [
    {
        'role': 'user',
        'content': """Your goal is to break down tasks according to the PRD/technical design, generate a task list, and analyze task dependencies for a quantitative trading backtesting system.

You have outlined a clear PRD/technical design for reproducing the paper’s trading strategy and experiments.

Now, please break down tasks according to the PRD/technical design, generate a task list, and analyze task dependencies. The Logic Analysis should not only consider the dependencies between files, but also provide detailed descriptions to assist in writing the code needed to reproduce the strategy and backtest it.

The system is expected to include at least the following files (or equivalent ones from the previous design step):
- main.py
- data/loader_yahoo.py
- strategy/strategy.py
- backtest/backtester.py
- backtest/metrics.py
- config.yaml

Important constraints:
- The data source is Yahoo Finance.
- The actual backtest date range MUST be fixed to 2000-01-01 to 2024-01-01, regardless of the original sample period used in the paper.
- The Logic Analysis should explicitly describe how each file uses this fixed date range and interacts with the others.

-----

## Format Example
[CONTENT]
{
    "Required packages": [
        "numpy==1.26.0",
        "pandas==2.2.0",
        "yfinance==0.2.50",
        "matplotlib==3.8.0"
    ],
    "Required Other language third-party packages": [
        "No third-party dependencies required"
    ],
    "Logic Analysis": [
        [
            "data/loader_yahoo.py",
            "Implements a DataLoader class that reads config.yaml, uses Yahoo Finance to download daily OHLCV data for the specified tickers between 2000-01-01 and 2024-01-01, handles missing data, aligns time series, and returns a clean price DataFrame."
        ],
        [
            "strategy/strategy.py",
            "Implements a Strategy class that reads parameters from config.yaml and, based on the paper, generates trading signals or portfolio weights from the price DataFrame (e.g., time-series momentum, DRL-based signals, etc.)."
        ],
        [
            "backtest/backtester.py",
            "Implements a Backtester class that takes prices and signals, simulates portfolio holdings over time, applies transaction costs, updates cash and positions, and produces a performance DataFrame over 2000-01-01 to 2024-01-01."
        ],
        [
            "backtest/metrics.py",
            "Implements a Metrics class or functions to compute key performance indicators such as annualized return, volatility, Sharpe ratio, maximum drawdown, and win rate, based on the performance DataFrame."
        ],
        [
            "main.py",
            "Entry point that loads config.yaml, initializes DataLoader, Strategy, Backtester, and Metrics, runs the full pipeline, and prints or saves results."
        ],
        [
            "config.yaml",
            "Configuration file that specifies data settings (tickers, date range fixed to 2000-01-01 to 2024-01-01), strategy hyperparameters, and backtest parameters (initial capital, transaction cost, etc.)."
        ]
    ],
    "Task list": [
        "config.yaml",
        "data/loader_yahoo.py",
        "strategy/strategy.py",
        "backtest/backtester.py",
        "backtest/metrics.py",
        "main.py"
    ],
    "Full API spec": "",
    "Shared Knowledge": "All modules share the same date range (2000-01-01 to 2024-01-01) and tickers specified in config.yaml. DataLoader, Strategy, Backtester, and Metrics must all assume consistent index alignment and frequency.",
    "Anything UNCLEAR": "Clarification needed on the exact universe of assets, transaction cost model, leverage and short-selling constraints, and whether the paper requires additional risk constraints beyond standard drawdown limits."
}
[/CONTENT]

## Nodes: "<node>: <type>  # <instruction>"
- Required packages: typing.Optional[typing.List[str]]  # Provide required third-party packages in requirements.txt format (e.g., "numpy==1.26.0").
- Required Other language third-party packages: typing.List[str]  # List packages required for non-Python languages. If none, specify "No third-party dependencies required".
- Logic Analysis: typing.List[typing.List[str]]  # Provide a list of files with the classes/methods/functions to be implemented, including dependency analysis and imports. Include as much detailed description as possible, especially for data/loader_yahoo.py, strategy/strategy.py, backtest/backtester.py, backtest/metrics.py, and main.py.
- Task list: typing.List[str]  # Break down the tasks into a list of filenames, prioritized based on dependency order. The task list must include the previously generated file list.
- Full API spec: <class 'str'>  # Describe all APIs using OpenAPI 3.0 if needed. If front-end and back-end communication is not required, you may leave this blank.
- Shared Knowledge: <class 'str'>  # Detail any shared knowledge, like common configuration variables (e.g., date range, tickers) and conventions (e.g., daily frequency, portfolio weights).
- Anything UNCLEAR: <class 'str'>  # Mention any unresolved questions or clarifications needed from the paper or project scope.

## Constraint
Format: output wrapped inside [CONTENT][/CONTENT] like the format example, nothing else.

## Action
Follow the node instructions above, generate your output accordingly, and ensure it follows the given format example."""
    }
]


# config
config_msg = [
    {
        'role': 'user',
        'content': """You write elegant, modular, and maintainable code. Adhere to Google-style guidelines.

Based on the paper, as well as the previously defined plan and design, you now need to generate ONLY the configuration file `config.yaml`.

The purpose of `config.yaml` is to configure:
- data settings (Yahoo Finance, tickers, fixed date range),
- strategy hyperparameters (as described in the paper), and
- backtest parameters (initial capital, transaction costs, etc.).

Important constraints:
- The actual backtest date range MUST be fixed to:
  start_date = 2000-01-01, end_date = 2024-01-01.
- Regardless of the original sample period used in the paper, the implementation and backtesting will always use this fixed date range.
- You MUST NOT fabricate strategy details; use only what is provided or clearly implied by the paper. If something is missing, choose a reasonable default and clearly mark it as such in a YAML comment.

ATTENTION: Use '##' to SPLIT SECTIONS, not '#'. Your output format must follow the example below exactly.

-----

# Format Example
## Code: config.yaml
```yaml
## config.yaml
data:
  source: "yahoo"
  tickers: ["SPY"]  # Replace with the asset universe used or implied in the paper, if available.
  start_date: "2000-01-01"  # Fixed implementation range.
  end_date: "2024-01-01"    # Fixed implementation range.
  interval: "1d"            # Daily data from Yahoo Finance.

strategy:
  name: "paper_strategy"    # A short name describing the strategy from the paper.
  type: "tsmom"             # Or "drlpm" or any label that matches the paper's methodology.
  params:
    lookback_window: 12     # Example: months or periods; if not specified in the paper, mark as a reasonable default.
    # Add additional hyperparameters as described in the paper.
    # If a parameter is not specified by the paper, add a comment that it is a default engineering choice.

backtest:
  initial_capital: 100000.0
  transaction_cost_bp: 5      # Transaction cost in basis points per trade; adjust per paper if specified.
  rebalance_frequency: "1d"   # Rebalance frequency; can be "1d", "1w", etc., depending on the paper.
  allow_short: true           # Set according to whether the paper allows short-selling.
  max_leverage: 1.0           # If the paper specifies leverage, reflect it; otherwise use a conservative default.
  risk_free_rate: 0.0         # Risk-free rate assumption used in metrics; adjust if the paper specifies one.
  metrics:
    - "annual_return"
    - "volatility"
    - "sharpe"
    - "max_drawdown"
    - "win_rate"
...
```

-----

## Code: config.yaml
"""
    }]

def api_call(msg, gpt_version):
    if "o3-mini" in gpt_version:
        completion = client.chat.completions.create(
            model=gpt_version, 
            reasoning_effort="high",
            messages=msg
        )
    else:
        completion = client.chat.completions.create(
            model=gpt_version, 
            messages=msg
        )

    return completion 

responses = []
trajectories = []
total_accumulated_cost = 0

for idx, instruction_msg in enumerate([plan_msg, file_list_msg, task_list_msg, config_msg]):
    current_stage = ""
    if idx == 0 :
        current_stage = f"[Planning] Overall plan"
    elif idx == 1:
        current_stage = f"[Planning] Architecture design"
    elif idx == 2:
        current_stage = f"[Planning] Logic design"
    elif idx == 3:
        current_stage = f"[Planning] Configuration file generation"
    print(current_stage)

    trajectories.extend(instruction_msg)

    completion = api_call(trajectories, gpt_version)
    
    # response
    completion_json = json.loads(completion.model_dump_json())

    # print and logging
    print_response(completion_json)
    temp_total_accumulated_cost = print_log_cost(completion_json, gpt_version, current_stage, output_dir, total_accumulated_cost)
    total_accumulated_cost = temp_total_accumulated_cost

    responses.append(completion_json)

    # trajectories
    message = completion.choices[0].message
    trajectories.append({'role': message.role, 'content': message.content})


# save
save_accumulated_cost(f"{output_dir}/accumulated_cost.json", total_accumulated_cost)

os.makedirs(output_dir, exist_ok=True)

with open(f'{output_dir}/planning_response.json', 'w') as f:
    json.dump(responses, f)

with open(f'{output_dir}/planning_trajectories.json', 'w') as f:
    json.dump(trajectories, f)
