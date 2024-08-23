import pandas as pd
import json
from langchain_core.utils.json import parse_json_markdown

# 创建空的列表来存储解析后的数据
data_list = []

# 读取 JSONL 文件
with open('D:\\pyworkspace\\agentUniverse\\sample_standard_app\\app\\examples\\data\\dataset_turn_1_2024-08-21-10-01'
          '-32.jsonl', 'r', encoding='utf-8') as file:
    for line in file:
        # 解析每行的 JSON 数据
        record = json.loads(line.strip())

        # 提取 query
        query = record['query']
        final_decision = 'Invest'  # 默认设为"Invest"

        # 创建临时字典来存储每个 expert 的信息
        temp_dict = {'query': query}

        # 提取每个 expert 的信息、建议和结果
        suggestions = record['answer']['suggestions']
        for suggestion in suggestions:
            for expert, analysis in suggestion.items():
                if analysis is not None:
                    analysis_data = parse_json_markdown(analysis)
                else:
                    analysis_data = {'info': 'N/A', 'suggestion': ['N/A'], 'result': 'N/A'}

                # 将每个 expert 的信息平铺到一行中
                temp_dict[f'{expert}_info'] = analysis_data['info']
                temp_dict[f'{expert}_suggestion'] = ', '.join(analysis_data['suggestion'])
                temp_dict[f'{expert}_result'] = analysis_data['result']

                # 如果有任何一个 expert 的结果是 "Not Invest"，将 final_decision 设置为 "Not Invest"
                if analysis_data['result'] != '投资':
                    final_decision = '不投资'

        # 添加 final_decision 到 temp_dict
        temp_dict['final_decision'] = final_decision

        # 将 temp_dict 添加到 data_list 中
        data_list.append(temp_dict)

# 将数据转换为 DataFrame
df = pd.DataFrame(data_list)

# 输出到 Excel 文件
df.to_excel('D:\\pyworkspace\\agentUniverse\\sample_standard_app\\app\\examples\\data\\parsed_data1.xlsx', index=False)
