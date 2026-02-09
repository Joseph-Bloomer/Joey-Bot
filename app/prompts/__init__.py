"""Prompt templates loader."""

import os
import yaml
from typing import Dict, Any


def load_prompts(templates_file: str = None) -> Dict[str, Any]:
    """
    Load prompt templates from YAML file.

    Args:
        templates_file: Path to YAML file. Defaults to chat_templates.yaml
                       in the same directory as this module.

    Returns:
        Dictionary of prompt templates
    """
    if templates_file is None:
        templates_file = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'chat_templates.yaml'
        )

    with open(templates_file, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def get_mode_prefix(prompts: Dict[str, Any], mode: str) -> str:
    """
    Get the mode prefix for a given mode.

    Args:
        prompts: Loaded prompts dictionary
        mode: Mode name (normal, concise, logic)

    Returns:
        Mode prefix string, empty if mode is 'normal'
    """
    if mode == 'normal':
        return ''
    return prompts.get('modes', {}).get(mode, '')


def format_fact_extraction_prompt(prompts: Dict[str, Any], messages_text: str) -> str:
    """Format the fact extraction prompt with messages."""
    return prompts['fact_extraction'].format(messages_text=messages_text)


def format_rolling_summary_prompt(
    prompts: Dict[str, Any],
    existing_summary: str,
    new_messages: str
) -> str:
    """Format the rolling summary prompt."""
    return prompts['rolling_summary'].format(
        existing_summary=existing_summary,
        new_messages=new_messages
    )


def format_title_summary_prompt(prompts: Dict[str, Any], history: str) -> str:
    """Format the title/summary generation prompt."""
    return prompts['title_summary'].format(history=history)


__all__ = [
    'load_prompts',
    'get_mode_prefix',
    'format_fact_extraction_prompt',
    'format_rolling_summary_prompt',
    'format_title_summary_prompt'
]
