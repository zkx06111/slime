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


class Rollout(BaseModel):
    """Represents a single rollout with reward and token information."""
    reward: float
    token_ids: list[int] | None = None
    mask_ids: list[int] | None = None
    trial_name: str | None = None
    metadata: dict[str, Any] | None = None
    messages: list[dict[str, Any]] | None = None  # Agent trajectory messages


async def run_rollouts(task_configs: list[TaskConfig], agent_name: str = "terminus-2") -> list[Rollout]:
    """
    Run Harbor trials and collect rollouts.

    Args:
        task_configs: List of task configurations to run
        agent_name: Name of the agent to use (default: terminus-2)

    Returns:
        List of Rollout objects containing rewards and token data
    """
    job = Job(
        config=JobConfig(
            jobs_dir=Path("jobs"),
            environment=EnvironmentConfig(
                type=EnvironmentType.DAYTONA,  # Change to DOCKER for local testing
            ),
            agents=[
                AgentConfig(
                    name=AgentName.TERMINUS_2,
                    model_name="openai/gpt-5.2",  # Default model
                    kwargs={},
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


async def main():
    """Example usage of the rollout collection."""
    import json

    # Example: Create a simple task configuration
    # In practice, you would load these from a dataset or create them programmatically
    example_tasks = [
        TaskConfig(
            id="example-task-1",
            path=Path("slime/examples/harbor/seta-task-0"),  # Adjust to your actual task path
        ),
    ]

    print("Starting rollout collection...")
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


if __name__ == "__main__":
    asyncio.run(main())