import pytest
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra

from balrog.agents import AgentFactory
from balrog.evaluator import EvaluatorManager
from balrog.utils import setup_environment

agents = [
    "dummy",
]

environments = [
    "nle",
    "minihack",
    "babyai",
    "crafter",
    "textworld",
    "babaisai",
]

clients = [
    ("gemini", "gemini-1.5-flash"),
    ("claude", "claude-3-5-sonnet-20240620"),
    ("openai", "gpt-4o-mini"),
]


@pytest.mark.parametrize("agent", agents)
@pytest.mark.parametrize("environment", environments)
@pytest.mark.parametrize("client,model_id", clients)
@pytest.mark.parametrize("max_image_history", [1])
def test_evaluation(agent, environment, client, model_id, max_image_history):
    with initialize(config_path="../config", version_base=None):
        cfg = compose(
            config_name="config",
            overrides=[
                f"agent.type={agent}",
                f"envs.names={environment}",
                f"client.client_name={client}",
                f"client.model_id={model_id}",
                f"agent.max_image_history={max_image_history}",
                # to reduce computational footprint of the tests
                f"eval.num_episodes.{environment}={1}",
                f"eval.num_workers={1}",
                f"eval.max_steps_per_episode={1}",
            ],
            return_hydra_config=True,
        )
        gh = GlobalHydra.instance()
        assert gh.is_initialized()

        # Check that the config is correct
        assert cfg.agent.type == agent
        assert cfg.envs.names == environment
        assert cfg.client.client_name == client

        # Run evaluation
        env_name = cfg.envs.names.split("-")[0]
        # we could pass task name as an argument, for now just use the first task
        cfg.tasks[f"{env_name}_tasks"] = cfg.tasks[f"{env_name}_tasks"][:1]
        evaluator = EvaluatorManager(cfg, original_cwd=cfg.hydra.runtime.cwd)
        agent_factory = AgentFactory(cfg)
        evaluator.run(agent_factory)
