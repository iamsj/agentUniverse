# !/usr/bin/env python3
# -*- coding:utf-8 -*-

# @Time    : 2024/3/12 17:47
# @Author  : jerry.zzw 
# @Email   : jerry.zzw@antgroup.com
# @FileName: component_enum.py
from enum import Enum


class ComponentEnum(Enum):
    """The enumeration of the supported components."""
    AGENT = "AGENT"
    KNOWLEDGE = "KNOWLEDGE"
    LLM = "LLM"
    PLANNER = "PLANNER"
    TOOL = "TOOL"
    DEFAULT = "DEFAULT"
    SERVICE = "SERVICE"
    MEMORY = "MEMORY"
    PROMPT = "PROMPT"
    SQLDB_WRAPPER = "SQLDB_WRAPPER"
    PRODUCT = "PRODUCT"

    @staticmethod
    def to_value_list():
        """Return the value list of the enumeration."""
        return [item.value for item in ComponentEnum]

    @staticmethod
    def from_value(value):
        """Return the enum member corresponding to the given value."""
        for item in ComponentEnum:
            if item.value == value:
                return item
        raise ValueError(f"No enum member with value {value}")
