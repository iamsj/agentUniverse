# !/usr/bin/env python3
# -*- coding:utf-8 -*-
import numpy as np

# @Time    : 2024/3/28 19:28
# @Author  : wangchongshi
# @Email   : wangchongshi.wcs@antgroup.com
# @FileName: sentiment_excel_knowledge.py

from agentuniverse.agent.action.knowledge.embedding.dashscope_embedding import DashscopeEmbedding
from agentuniverse.agent.action.knowledge.knowledge import Knowledge
from agentuniverse.agent.action.knowledge.reader.file.excel_reader import ExcelReader
from agentuniverse.agent.action.knowledge.store.chroma_store import ChromaStore
from agentuniverse.agent.action.knowledge.store.document import Document
from langchain.text_splitter import TokenTextSplitter
from pathlib import Path

SPLITTER = TokenTextSplitter(chunk_size=600, chunk_overlap=100)


class SentimentKnowledge(Knowledge):
    """The sentiment Excel knowledge."""

    def __init__(self, **kwargs):
        """The __init__ method.

        Some parameters, such as name and description,
        are injected into this class by the configuration.

        Args:
            name (str): Name of the knowledge.
            description (str): Description of the knowledge.
            store (Store): Store of the knowledge, store class is used to store knowledge
            and provide retrieval capabilities, such as ChromaDB store or Redis Store,
            this knowledge uses ChromaDB as the knowledge storage.
            reader (Reader): Reader is used to load data,
            this knowledge uses ExcelReader to load Excel files.
        """
        super().__init__(**kwargs)
        self.store = ChromaStore(
            collection_name="sentiment_store",
            persist_path="../../DB/sentiment.db",
            embedding_model=DashscopeEmbedding(
                embedding_model_name='text-embedding-v2'
            ),
            dimensions=1536
        )
        self.reader = ExcelReader()
        # Initialize the knowledge
        self.insert_knowledge()

    def insert_knowledge(self, **kwargs) -> None:
        """
            Load sentiment data from all Excel files in a given folder and save them into a vector database.
        """

        folder_path = "../resources/sentiment"
        # 获取文件夹路径
        folder_path = Path(folder_path)

        # 获取文件夹下所有Excel文件的路径
        excel_files = [f for f in folder_path.glob('*.xlsx') if f.is_file()]

        # 初始化一个空列表来收集所有文档
        all_sentiment_docs = []

        # 循环遍历文件路径列表
        for file_path in excel_files:
            # 加载单个文件的数据
            df = self.reader.load_data(file_path)

            # 分批次处理 DataFrame
            for index, chunk in df.groupby(np.arange(len(df)) // 1000):  # 每 1000 行作为一个批次
                # 处理每一批次的数据
                sentiment_docs = self.process_chunk(chunk)
                all_sentiment_docs.extend(sentiment_docs)

        # 使用文档分割器处理所有文档
        lc_doc_list = SPLITTER.split_documents(Document.as_langchain_list(all_sentiment_docs))

        # 将处理后的文档插入到存储中
        self.store.insert_documents(Document.from_langchain_list(lc_doc_list))

    def process_chunk(self, chunk):
        # 在这里处理每个批次的数据
        # 对于每一行，创建一个文档对象，其中包含每一列的内容
        docs = []
        for _, row in chunk.iterrows():
            # 创建一个字典来存储每一列的内容
            content_dict = {col: str(row[col]) for col in chunk.columns}
            # 将字典转换为字符串并用作 page_content 的值
            page_content = "\n".join([f"{k}: {v}" for k, v in content_dict.items()])
            doc = Document(page_content=page_content,
                           metadata={'source': row['file_name'], 'sheet_name': row['sheet_name']})
            docs.append(doc)
        return docs
