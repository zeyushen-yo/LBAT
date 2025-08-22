from balrog.environments.babaisai.base import BabaIsAIWrapper

__all__ = [BabaIsAIWrapper]


ACTIONS = {
    "idle": "wait for one step",
    "up": "take one step up",
    "right": "take one step to the right",
    "down": "take one step down",
    "left": "take one step to the left",
}


def get_instruction_prompt(env, task=None):
    action_strings = ",\n".join(f"{action}: {description}" for action, description in ACTIONS.items())

    instruction_prompt = f"""
Baba Is You is a puzzle game where you can manipulate the rules of each level. The following are the possible actions you can take in the game, followed by a short description of each action:

{action_strings}.

Tips:
- Examine the level carefully, noting all objects and text blocks present.
- Identify the current rules, which are formed by text blocks in the format "[Subject] IS [Property]" (e.g. "BABA IS YOU").
- Consider how you can change or create new rules by moving text blocks around.
- Remember that you can only move objects or text that are not defined as "STOP" or similar immovable properties.
- Your goal is usually to reach an object defined as "WIN", but this can be changed.
- Think creatively about how changing rules can alter the properties and behaviors of objects in unexpected ways.
- If stuck, try breaking apart existing rules or forming completely new ones.
- Sometimes the solution involves making yourself a different object or changing what counts as the win condition.

PLAY!
""".strip()

    return instruction_prompt
