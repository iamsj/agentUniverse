# !/usr/bin/env python3
# -*- coding:utf-8 -*-
# @Time    : 2024/8/15
# @Author  : YourName
# @Email   : your.email@example.com
# @FileName: stock_factor_analysis_planner.py
"""Stock Factor Analysis Planner module."""
import asyncio

from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory

from agentuniverse.agent.agent_model import AgentModel
from agentuniverse.agent.input_object import InputObject
from agentuniverse.agent.memory.chat_memory import ChatMemory
from agentuniverse.agent.plan.planner.planner import Planner
from agentuniverse.base.util.memory_util import generate_memories
from agentuniverse.base.util.prompt_util import process_llm_token
from agentuniverse.llm.llm import LLM
from agentuniverse.prompt.prompt import Prompt
from agentuniverse.prompt.prompt_manager import PromptManager
from agentuniverse.prompt.prompt_model import AgentPromptModel

class StockFactorAnalysisPlanner(Planner):
    """Stock Factor Analysis planner class."""

    def invoke(self, agent_model: AgentModel, planner_input: dict, input_object: InputObject) -> dict:
        """Invoke the planner for stock factor analysis and multi-agent collaboration.

        Args:
            agent_model (AgentModel): Agent model object.
            planner_input (dict): Planner input object.
            input_object (InputObject): The input parameters passed by the user.
        Returns:
            dict: The planner result.
        """
        # 处理多个专家分析任务，结合不同分析步骤和工具
        memory: ChatMemory = self.handle_memory(agent_model, planner_input)
        llm: LLM = self.handle_llm(agent_model)
        prompt: Prompt = self.handle_prompt(agent_model, planner_input)

        # 使用 LLM 处理 prompt 并生成结构化的分析结果
        process_llm_token(llm, prompt.as_langchain(), agent_model.profile, planner_input)

        chat_history = memory.as_langchain().chat_memory if memory else InMemoryChatMessageHistory()

        # 结合历史消息和当前的分析链条执行
        chain_with_history = RunnableWithMessageHistory(
            prompt.as_langchain() | llm.as_langchain(),
            lambda session_id: chat_history,
            history_messages_key="chat_history",
            input_messages_key=self.input_key,
        ) | StrOutputParser()

        # 执行链式分析任务并返回结果
        res = self.invoke_chain(agent_model, chain_with_history, planner_input, chat_history, input_object)
        return {**planner_input, self.output_key: res, 'chat_history': generate_memories(chat_history)}

    def handle_prompt(self, agent_model: AgentModel, planner_input: dict) -> Prompt:
        """Generate prompt template for the stock factor analysis and multi-agent collaboration.

        Args:
            agent_model (AgentModel): The agent model.
            planner_input (dict): The planner input.
        Returns:
            Prompt: The prompt instance.
        """
        expert_framework = planner_input.pop('expert_framework', '') or ''

        profile: dict = agent_model.profile

        # 将不同专家的prompt结合并生成最终的prompt指令
        profile_instruction = profile.get('instruction')
        profile_instruction = expert_framework + profile_instruction if profile_instruction else profile_instruction

        profile_prompt_model: AgentPromptModel = AgentPromptModel(
            introduction=profile.get('introduction'),
            target=profile.get('target'),
            instruction=profile_instruction
        )

        # 获取指定版本的 prompt 模型
        prompt_version: str = profile.get('prompt_version')
        version_prompt: Prompt = PromptManager().get_instance_obj(prompt_version)

        if version_prompt is None and not profile_prompt_model:
            raise Exception("Either the `prompt_version` or `introduction & target & instruction`"
                            " in agent profile configuration should be provided.")
        if version_prompt:
            version_prompt_model: AgentPromptModel = AgentPromptModel(
                introduction=getattr(version_prompt, 'introduction', ''),
                target=getattr(version_prompt, 'target', ''),
                instruction=expert_framework + getattr(version_prompt, 'instruction', '')
            )
            profile_prompt_model = profile_prompt_model + version_prompt_model

        # 构建并返回最终的 Prompt 对象
        return Prompt().build_prompt(profile_prompt_model, self.prompt_assemble_order)

