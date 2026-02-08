"""
Minimal runnable script for generating rollouts from Harbor trials.
This script runs an agent on tasks and collects rewards and token data.
"""
import asyncio
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from pydantic import BaseModel

# Load environment variables from .env file
load_dotenv(Path(__file__).parent.parent.parent / ".env")

from harbor.job import Job
from harbor.models.agent.name import AgentName
from harbor.models.environment_type import EnvironmentType
from harbor.models.job.config import JobConfig, OrchestratorConfig
from harbor.models.trial.config import AgentConfig, EnvironmentConfig, TaskConfig
from slime.utils.http_utils import post
from slime.utils.types import Sample


# Map task index to task path
TASK_PATHS = {
    0: Path("examples/harbor/seta-task-0"),
    # Add more task paths as needed
    # 1: Path("slime/examples/harbor/seta-task-1"),
    # 2: Path("slime/examples/harbor/seta-task-2"),
}


class Rollout(BaseModel):
    """Represents a single rollout with reward and token information."""
    reward: float
    token_ids: list[int] | None = None
    mask_ids: list[int] | None = None
    trial_name: str | None = None
    metadata: dict[str, Any] | None = None
    messages: list[dict[str, Any]] | None = None  # Agent trajectory messages


async def run_rollouts(
    task_configs: list[TaskConfig],
    agent_name: str = "terminus-2",
    model_name: str = "openai/gpt-5.2",
    api_base: str | None = None,
    sampling_params: dict | None = None,
    model_info: dict | None = None,
) -> list[Rollout]:
    """
    Run Harbor trials and collect rollouts.

    Args:
        task_configs: List of task configurations to run
        agent_name: Name of the agent to use (default: terminus-2)
        model_name: Model name (e.g., "openai/gpt-5.2" or "hosted_vllm/qwen2.5-3b")
        api_base: API base URL for hosted models (e.g., sglang server URL)
        sampling_params: Sampling parameters for generation
        model_info: Model information (max_input_tokens, max_output_tokens, costs)

    Returns:
        List of Rollout objects containing rewards and token data
    """
    # Build kwargs for agent config
    agent_kwargs = {}
    if api_base:
        agent_kwargs["api_base"] = api_base
    if sampling_params:
        agent_kwargs.update(sampling_params)
    if model_info:
        agent_kwargs["model_info"] = model_info

    job = Job(
        config=JobConfig(
            jobs_dir=Path("jobs"),
            environment=EnvironmentConfig(
                type=EnvironmentType.DAYTONA,  # Change to DOCKER for local testing
            ),
            agents=[
                AgentConfig(
                    name=AgentName.TERMINUS_2,
                    model_name=model_name,
                    kwargs=agent_kwargs,
                )
            ],
            orchestrator=OrchestratorConfig(
                n_concurrent_trials=4,  # Reduced for local testing
            ),
            tasks=task_configs,
        ),
    )

    print(f"Running {len(task_configs)} tasks with agent {agent_name}...")
    result = await job.run()

    # Load individual trial results from disk
    from harbor.models.trial.result import TrialResult
    rollouts = []
    job_dir = Path("jobs") / result.started_at.strftime("%Y-%m-%d__%H-%M-%S")
    for trial_dir in job_dir.iterdir():
        if not trial_dir.is_dir():
            continue
        trial_result_path = trial_dir / "result.json"
        if not trial_result_path.exists():
            continue
        trial_result = TrialResult.model_validate_json(trial_result_path.read_text())

        # Create rollout from trial result
        reward = (
            trial_result.verifier_result.rewards.get("reward", 0)
            if trial_result.verifier_result and trial_result.verifier_result.rewards
            else 0
        )

        # Extract token and mask IDs if available
        token_ids = None
        mask_ids = None
        if (
            trial_result.agent_result
            and trial_result.agent_result.metadata
        ):
            token_ids = trial_result.agent_result.metadata.get("token_ids")
            mask_ids = trial_result.agent_result.metadata.get("mask_ids")

        # Load trajectory messages and convert to OpenAI format
        messages = None
        trajectory_path = trial_dir / "agent" / "trajectory.json"
        if trajectory_path.exists():
            import json
            trajectory = json.loads(trajectory_path.read_text())
            # Convert ATIF format to OpenAI message format
            messages = []
            for step in trajectory.get("steps", []):
                source = step.get("source")

                if source == "user":
                    # User message
                    messages.append({
                        "role": "user",
                        "content": step.get("message", "")
                    })
                elif source == "agent":
                    # Agent message (assistant role)
                    msg = {
                        "role": "assistant",
                        "content": step.get("message", "")
                    }

                    # Add tool calls if present
                    tool_calls = step.get("tool_calls")
                    if tool_calls:
                        # Convert to OpenAI tool_calls format
                        openai_tool_calls = []
                        for tc in tool_calls:
                            openai_tool_calls.append({
                                "id": tc.get("tool_call_id"),
                                "type": "function",
                                "function": {
                                    "name": tc.get("function_name"),
                                    "arguments": json.dumps(tc.get("arguments", {}))
                                }
                            })
                        msg["tool_calls"] = openai_tool_calls

                    messages.append(msg)

                    # Add tool response messages if there are observations
                    observation = step.get("observation")
                    if observation and observation.get("results"):
                        for idx, result in enumerate(observation["results"]):
                            # Find corresponding tool call
                            tool_call_id = None
                            if tool_calls and idx < len(tool_calls):
                                tool_call_id = tool_calls[idx].get("tool_call_id")

                            messages.append({
                                "role": "tool",
                                "tool_call_id": tool_call_id,
                                "content": result.get("content", "")
                            })

        rollout = Rollout(
            reward=reward,
            token_ids=token_ids,
            mask_ids=mask_ids,
            trial_name=trial_result.trial_name,
            messages=messages,
            metadata={
                "task_id": str(trial_result.task_id),
                "agent": trial_result.agent_info.name,
                "duration_sec": (
                    (trial_result.finished_at - trial_result.started_at).total_seconds()
                    if trial_result.finished_at
                    else 0
                ),
                "n_messages": len(messages) if messages else 0,
            }
        )

        rollouts.append(rollout)
        print(f"  Trial {trial_result.trial_name}: reward={reward}")

    return rollouts


async def generate(args, sample: Sample, sampling_params) -> Sample:
    """
    Wrapper around run_rollouts for single sample generation with sglang.

    Args:
        args: Arguments containing sglang configuration (sglang_router_ip, sglang_router_port, model_name)
        sample: Sample object containing prompt and index
        sampling_params: Sampling parameters for generation

    Returns:
        Sample with generated response and metadata
    """
    # Get task path from sample's index using global TASK_PATHS
    task_path = TASK_PATHS.get(sample.index)
    if task_path is None:
        raise ValueError(f"No task path found for index {sample.index}")

    # Create task config
    task_config = TaskConfig(
        id=f"task-{sample.index}",
        path=task_path,
    )

    # Construct API base URL for sglang (include /v1 for OpenAI compatibility)
    api_base = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/v1"

    # Replace slashes in model name to avoid Harbor validation error
    # Harbor expects hosted_vllm models to have exactly one '/'
    model_name_safe = args.model_name.replace("/", "--")

    # Get model_info from args if available
    model_info = getattr(args, 'model_info', None)

    # Use run_rollouts to execute the task
    rollouts = await run_rollouts(
        task_configs=[task_config],
        agent_name="terminus-2",
        model_name=f"hosted_vllm/{model_name_safe}",
        api_base=api_base,
        sampling_params=sampling_params,
        model_info=model_info,
    )

    # Extract rollout data and update sample
    if rollouts:
        rollout = rollouts[0]
        sample.reward = rollout.reward
        sample.token_ids = rollout.token_ids
        sample.mask_ids = rollout.mask_ids
        sample.messages = rollout.messages
        sample.metadata = rollout.metadata
        sample.status = Sample.Status.COMPLETED
    else:
        sample.status = Sample.Status.ABORTED

    return sample


async def main():
    """Example usage of the rollout collection."""
    import json
    from argparse import Namespace

    # Example: Create a simple task configuration
    # In practice, you would load these from a dataset or create them programmatically
    example_tasks = [
        TaskConfig(
            id="example-task-1",
            path=TASK_PATHS[0],  # Adjust to your actual task path
        ),
    ]

    print("Starting rollout collection with run_rollouts...")
    rollouts = await run_rollouts(example_tasks)

    print(f"\nCollected {len(rollouts)} rollouts:")
    for i, rollout in enumerate(rollouts, 1):
        print(f"\n{i}. {rollout.trial_name}")
        print(f"   Reward: {rollout.reward}")
        print(f"   Has token_ids: {rollout.token_ids is not None}")
        print(f"   Has mask_ids: {rollout.mask_ids is not None}")
        print(f"   Metadata: {rollout.metadata}")

    # Pretty print the full structure
    print("\n" + "="*80)
    print("Full rollouts structure (as JSON):")
    print("="*80)
    rollouts_json = [rollout.model_dump() for rollout in rollouts]
    print(json.dumps(rollouts_json, indent=2))

    # Optionally save to file
    output_file = Path("rollouts_output.json")
    output_file.write_text(json.dumps(rollouts_json, indent=2))
    print(f"\n✓ Saved rollouts to {output_file}")

    # Test sglang server with OpenAI compatible request
    print("\n" + "="*80)
    print("Testing sglang server with OpenAI compatible request...")
    print("="*80)

    sglang_ip = "localhost"
    sglang_port = 30000
    model_name = "Qwen/Qwen3-Coder-Next"
    
    import openai

    client = openai.Client(base_url=f"http://{sglang_ip}:{sglang_port}/v1", api_key="None")

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "user", "content": "List 3 countries and their capitals."},
        ],
        temperature=0,
        max_tokens=64,
    )

    print(f"✓ Server responded successfully!")
    print(f"   Response preview: {response.choices[0].message.content[:100]}...")

    # Test generate function with sglang
    print("\n" + "="*80)
    print("Testing generate function with sglang (Qwen/Qwen3-Coder-Next)...")
    print("="*80)

    # Create args for sglang configuration
    args = Namespace(
        sglang_router_ip=sglang_ip,
        sglang_router_port=sglang_port,
        model_name=model_name,
        model_info={
            "max_input_tokens": 32768,
            "max_output_tokens": 8192,
            "input_cost_per_token": 0.0,
            "output_cost_per_token": 0.0,
        },
    )

    # Create a sample with index matching TASK_PATHS
    sample = Sample(
        index=0,
        prompt="Solve the task using the provided environment.",
    )

    # Define sampling parameters
    sampling_params = {
        "temperature": 0.7,
        "top_p": 0.9,
        "max_tokens": 2048,
    }

    print(f"\nGenerating with sglang at {args.sglang_router_ip}:{args.sglang_router_port}")
    print(f"Model: {args.model_name}")
    print(f"Task index: {sample.index} (path: {TASK_PATHS[sample.index]})")

    # Call generate
    result_sample = await generate(args, sample, sampling_params)

    print(f"\n✓ Generation completed!")
    print(f"   Status: {result_sample.status}")
    print(f"   Reward: {result_sample.reward if hasattr(result_sample, 'reward') else 'N/A'}")
    print(f"   Messages: {len(result_sample.messages) if result_sample.messages else 0}")
    print(f"   Metadata: {result_sample.metadata}")

    # Save generate result
    generate_output_file = Path("generate_output.json")
    generate_result = {
        "index": result_sample.index,
        "status": str(result_sample.status),
        "reward": result_sample.reward if hasattr(result_sample, 'reward') else None,
        "messages": result_sample.messages,
        "metadata": result_sample.metadata,
    }
    generate_output_file.write_text(json.dumps(generate_result, indent=2))
    print(f"\n✓ Saved generate result to {generate_output_file}")


if __name__ == "__main__":
    asyncio.run(main())