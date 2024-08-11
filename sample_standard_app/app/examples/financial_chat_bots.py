# !/usr/bin/env python3
# -*- coding:utf-8 -*-

# @Time    : 2024/6/7 11:22
# @Author  : wangchongshi
# @Email   : wangchongshi.wcs@antgroup.com
# @FileName: discussion_chat_bots.py
from agentuniverse.base.agentuniverse import AgentUniverse
from agentuniverse.agent.agent import Agent
from agentuniverse.agent.agent_manager import AgentManager

AgentUniverse().start(config_path='../../config/config.toml')


def chat(question: str):
    instance: Agent = AgentManager().get_instance_obj('financial_analysis_agent')
    return instance.run(input=question)


if __name__ == '__main__':
    print(chat("五粮液股票的未来发展前景如何？").to_json_str())
