# Harbor Example

This example demonstrates how to use Harbor to run AI agents on software engineering tasks and collect rollouts for reinforcement learning.

## Overview

Harbor is a framework for running AI agents in isolated environments to solve software engineering tasks. This example shows how to:
- Run agents on tasks and collect reward signals
- Extract token IDs and trajectories for training
- Host models locally using SGLang
- Integrate Harbor with Slime's training pipeline

## Prerequisites

### 1. Environment Variables

Create a `.env` file in the root of the repository with the following two keys:

```bash
DAYTONA_API_KEY=your-daytona-api-key
OPENAI_API_KEY=your-openai-api-key
```

- **DAYTONA_API_KEY**: Required for running tasks in Daytona containerized environments
- **OPENAI_API_KEY**: Required for using OpenAI models with Harbor agents

### 2. Local Model Hosting (Optional)

For faster iteration and lower costs, you can host models locally using SGLang:

```bash
python -m sglang.launch_server --model Qwen/Qwen3-Coder-Next --port 30000 --tp 4
```

This will start an OpenAI-compatible API server at `http://localhost:30000/v1`.

**Parameters:**
- `--model`: HuggingFace model name
- `--port`: Port to serve the model on (default: 30000)
- `--tp`: Tensor parallelism degree (number of GPUs to use)

## Running the Example

### Basic Usage

Run the rollout test script:

```bash
python examples/harbor/rollout_test.py
```

This will:
1. Load the example task from `seta-task-0`
2. Run an agent to solve the task
3. Collect rewards and trajectories
4. Test the SGLang server connection (if running)
5. Generate rollouts using the locally hosted model

### Output

The script generates two output files:

- `rollouts_output.json`: Contains all rollout data including rewards, token IDs, and metadata
- `generate_output.json`: Contains the result from the `generate()` function

## Task Structure

Each task directory (e.g., `seta-task-0`) contains:

```
seta-task-0/
├── task.toml              # Task configuration and metadata
├── instruction.md         # Task description and requirements
├── environment/           # Docker environment setup
│   ├── Dockerfile
│   ├── draft_spec.md
│   └── weights.json
├── solution/              # Reference solution
│   └── solve.sh
└── tests/                 # Verification tests
    ├── test.sh
    └── test_outputs.py
```

## Configuration

### Agent Configuration

Agents are configured in the `rollout_test.py` script:

```python
AgentConfig(
    name=AgentName.TERMINUS_2,
    model_name="openai/gpt-5.2",  # or "hosted_vllm/qwen3-coder-next"
    kwargs={
        "api_base": "http://localhost:30000/v1",  # for local models
        "temperature": 0.7,
        "max_tokens": 2048,
    }
)
```

### Environment Type

You can switch between environment types:

- `EnvironmentType.DAYTONA`: Remote containerized environments (default)
- `EnvironmentType.DOCKER`: Local Docker containers (for development/testing)

Change this in the `JobConfig`:

```python
environment=EnvironmentConfig(
    type=EnvironmentType.DOCKER,  # or DAYTONA
)
```

## Integration with Slime

The `generate()` function in `rollout_test.py` is designed to integrate with Slime's rollout generation pipeline. It returns a `Sample` object with:

- `reward`: Task completion reward (0-1)
- `token_ids`: Token IDs for the generated trajectory
- `mask_ids`: Mask IDs for training
- `messages`: Agent trajectory in OpenAI message format
- `metadata`: Additional information (task ID, duration, etc.)

This allows Harbor rollouts to be used directly for policy optimization in Slime.

## Troubleshooting

### SGLang Server Connection Issues

If you get connection errors when using local models:

1. Ensure the SGLang server is running: `curl http://localhost:30000/v1/models`
2. Check the port number matches your configuration
3. Verify the model loaded successfully (check server logs)

### Harbor Installation

If Harbor is not installed:

```bash
pip install harbor-agent
```

### Environment Variables Not Loaded

Make sure the `.env` file is in the repository root (not in `examples/harbor/`). The script loads it from:

```python
load_dotenv(Path(__file__).parent.parent.parent / ".env")
```

## Additional Resources

- Harbor Documentation: [harbor-agent.ai](https://harbor-agent.ai)
- SGLang Documentation: [sgl-project.github.io](https://sgl-project.github.io/)
- Slime Training Pipeline: See `slime/` directory
