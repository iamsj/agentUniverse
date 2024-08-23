#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# @Time    : 2024/3/13 10:56
# @Author  : heji
# @Email   : lc299034@antgroup.com
# @FileName: peer_planner.py
"""Peer planner module."""
import json
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
from typing import Any

from langchain_core.utils.json import parse_json_markdown

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
    'summary_expert': 'ReviewingAgent',  # 确保总结专家使用的是 ReviewingAgent
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
            OutputObject: The result from the expert agent.
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
        #final_result = {'suggestions': [], 'final_decision': '不投资'}
        final_result = {'suggestions': [], 'final_decision': '不投资'}  # 确保 'suggestions' 已初始化为列表
        retry_count = planner_config.get('retry_count', default_retry_count)
        eval_threshold = planner_config.get('eval_threshold', default_eval_threshold)
        timeout = planner_config.get('timeout', default_timeout)

        # Initialize results for each expert
        results = {
            'financial_expert': OutputObject({}),
            'intelligence_expert': OutputObject({}),
            'stock_factor_expert': OutputObject({}),
        }

        # Create a thread pool executor for concurrent execution
        with ThreadPoolExecutor(max_workers=3) as executor:
            future_to_expert = {}
            for expert_name, expert_agent in agents.items():
                if expert_name != 'summary_expert':
                    # Submit task to the executor (initial attempt)
                    future = executor.submit(self.run_expert, expert_name, expert_agent, input_object)
                    future_to_expert[future] = (expert_name, 0)  # Track attempts

            # Collect results as they complete
            while future_to_expert:
                for future in as_completed(future_to_expert):
                    expert_name, attempt = future_to_expert[future]
                    try:
                        result_obj = future.result(timeout=timeout)
                        results[expert_name] = result_obj
                        LOGGER.info(f"{expert_name} completed with result: {result_obj.to_dict()}")
                        future_to_expert.pop(future)  # Task completed successfully
                    except TimeoutError:
                        LOGGER.warning(f"{expert_name} did not respond within timeout.")
                        if attempt < retry_count:
                            # Resubmit task for retry
                            new_future = executor.submit(self.run_expert, expert_name, agents[expert_name],
                                                         input_object)
                            future_to_expert[new_future] = (expert_name, attempt + 1)
                            LOGGER.info(f"Retrying {expert_name}, attempt {attempt + 2}")
                        else:
                            results[expert_name] = OutputObject({})  # Mark as failed due to timeout
                            future_to_expert.pop(future)
                    except Exception as e:
                        LOGGER.error(f"Error retrieving result for {expert_name}: {e}")
                        if attempt < retry_count:
                            # Resubmit task for retry
                            new_future = executor.submit(self.run_expert, expert_name, agents[expert_name],
                                                         input_object)
                            future_to_expert[new_future] = (expert_name, attempt + 1)
                            LOGGER.info(f"Retrying {expert_name}, attempt {attempt + 2}")
                        else:
                            results[expert_name] = OutputObject({})  # Mark as failed due to error
                            future_to_expert.pop(future)

        # # Verify if summary_expert exists and is correctly instantiated before running it
        # if 'summary_expert' not in agents or agents['summary_expert'] is None:
        #     LOGGER.error("Cannot run summary_expert: summary_expert is not instantiated.")
        #     return {"error": "summary_expert instantiation failed"}
        #
        # summary_input = InputObject({
        #     'input': agent_input.get('input'),
        #     'expressing_result': OutputObject(results)
        # })
        #
        # summary_expert = agents['summary_expert']
        # summary_result = self.run_expert('summary_expert', summary_expert, summary_input)
        #
        # if summary_result.get_data('score') and summary_result.get_data('score') >= eval_threshold:
        #     result['result'] = summary_result.to_dict()
        # else:
        #     result['result'] = {
        #         key: val.to_dict() for key, val in results.items() if key != 'summary_expert'
        #     }

        # Initialize score variables
        score = 0
        # Define score mappings
        score_mapping = {
            '投资': 1,
            '持有': 0,
            '不投资': -1
        }

        for expert_name, result_obj in results.items():
            suggestion = result_obj.get_data('output')  # 获取专家建议
            final_result['suggestions'].append({expert_name: suggestion})
            # Ensure output_json is not None before parsing
            output_json = None
            if suggestion is not None:
                try:
                    output_json = parse_json_markdown(suggestion)
                except Exception as e:
                    LOGGER.error(f"Failed to parse JSON markdown for {expert_name}: {e}")

            # Use a try-except block to catch any exceptions when accessing output_json
            try:
                if output_json is not None:
                    result = output_json.get('result')
                    # Update total score based on suggestion
                    score += score_mapping.get(result, -1)
            except Exception as e:
                LOGGER.error(f"An error occurred while processing output_json for {expert_name}: {e}")

        if score == 2:
            final_result['final_decision'] = '投资'
        else:
            final_result['final_decision'] = '不投资'

        final_result_output = {'output': final_result}

        return final_result_output


def determine_final_decision(results):
    """
    Determine the final investment decision based on expert recommendations.

    :param results: List of dictionaries containing expert recommendations
    :return: Final investment decision as a string
    """
    # Initialize counts
    investment_count = 0
    hold_count = 0

    # Count the number of each type of recommendation
    for result in results:
        if result['result'] == '投资':
            investment_count += 1
        elif result['result'] == '持有':
            hold_count += 1

    # Determine the final decision
    if investment_count == 3:
        return '投资'
    elif investment_count == 2 and hold_count == 1:
        return '投资'
    else:
        return '不投资'
