#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# @Time    : 2024/3/13 10:56
# @Author  : heji
# @Email   : lc299034@antgroup.com
# @FileName: peer_planner.py
"""Peer planner module."""
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
from typing import Any

from agentuniverse.agent.action.tool.tool_manager import ToolManager
from agentuniverse.agent.agent_manager import AgentManager
from agentuniverse.agent.agent_model import AgentModel
from agentuniverse.agent.input_object import InputObject
from agentuniverse.agent.output_object import OutputObject
from agentuniverse.agent.plan.planner.planner import Planner
from agentuniverse.base.util.logging.logging_util import LOGGER

# Define specific sub-agents for your needs
default_sub_agents = {
    'financial_expert': 'financial_analysis_agent',
    'intelligence_expert': 'intelligence_analysis_agent',
    'stock_factor_expert': 'stock_factor_analysis_agent',
    'summary_expert': 'demo_reviewing_agent',
}

default_retry_count = 2
default_eval_threshold = 60
default_timeout = 30  # Timeout for each expert in seconds


class StockGPeerPlanner(Planner):
    """Peer planner class."""

    def invoke(self, agent_model: AgentModel, planner_input: dict, input_object: InputObject) -> dict:
        """Invoke the planner.

        Args:
            agent_model (AgentModel): Agent model object.
            planner_input (dict): Planner input object.
            input_object (InputObject): The input parameters passed by the user.
        Returns:
            dict: The planner result.
        """
        planner_config = agent_model.plan.get('planner')
        sub_agents = self.generate_sub_agents(planner_config)
        return self.agents_run(agent_model, sub_agents, planner_config, planner_input, input_object)

    def generate_sub_agents(self, planner_config: dict) -> dict:
        """Generate sub agents.

        Args:
            planner_config (dict): Planner config object.
        Returns:
            dict: Planner agents.
        """
        agents = dict()
        agent_manager = AgentManager()
        for config_key, default_agent in default_sub_agents.items():
            config_data = planner_config.get(config_key)
            if config_data is None:
                agents[config_key] = agent_manager.get_instance_obj(default_agent)
            elif config_data:
                agents[config_key] = agent_manager.get_instance_obj(config_data)
        return agents

    def run_expert(self, agent_name: str, agent, input_object: InputObject) -> OutputObject | Any:
        """Run a single expert agent.

        Args:
            agent_name (str): The name of the agent.
            agent: The agent instance.
            input_object (InputObject): The input parameters.
        Returns:
            dict: The result from the expert agent.
        """
        try:
            return agent.run(**input_object.to_dict())
        except Exception as e:
            LOGGER.error(f"Error running {agent_name}: {e}")
            return OutputObject({})  # Return empty result on failure

    def agents_run(self, agent_model: AgentModel, agents: dict, planner_config: dict, agent_input: dict,
                   input_object: InputObject) -> dict:
        """Planner agents run.

        Args:
            agent_model (AgentModel): Agent model object.
            agents (dict): Planner agents.
            planner_config (dict): Planner config object.
            agent_input (dict): Planner input object.
            input_object (InputObject): Agent input object.
        Returns:
            dict: The planner result.
        """
        result: dict = dict()
        retry_count = planner_config.get('retry_count', default_retry_count)
        eval_threshold = planner_config.get('eval_threshold', default_eval_threshold)
        timeout = planner_config.get('timeout', default_timeout)

        # Initialize results for each expert
        results = {
            'financial_expert': OutputObject({}),
            'intelligence_expert': OutputObject({}),
            'stock_factor_expert': OutputObject({}),
            'summary_expert': OutputObject({})
        }

        # Create a thread pool executor for concurrent execution
        with ThreadPoolExecutor(max_workers=4) as executor:
            future_to_expert = {}
            for expert_name, expert_agent in agents.items():
                # Submit tasks to the executor
                for attempt in range(retry_count):
                    future = executor.submit(self.run_expert, expert_name, expert_agent, input_object)
                    future_to_expert[future] = expert_name
                    LOGGER.info(f"Submitted task for {expert_name}, attempt {attempt + 1}")

            # Collect results as they complete
            for future in as_completed(future_to_expert):
                expert_name = future_to_expert[future]
                try:
                    result_obj = future.result(timeout=timeout)  # Set timeout for each expert
                    results[expert_name] = result_obj
                    LOGGER.info(f"{expert_name} completed with result: {result_obj.to_dict()}")
                except TimeoutError:
                    LOGGER.warning(f"{expert_name} did not respond within timeout.")
                    results[expert_name] = OutputObject({})  # Mark as failed due to timeout
                except Exception as e:
                    LOGGER.error(f"Error retrieving result for {expert_name}: {e}")

        # Process summary expert results
        summary_result = results.get('summary_expert')
        if summary_result.get_data('score') and summary_result.get_data('score') >= eval_threshold:
            result['result'] = results
        else:
            result['result'] = {
                key: val for key, val in results.items() if key != 'summary_expert'
            }

        return result
