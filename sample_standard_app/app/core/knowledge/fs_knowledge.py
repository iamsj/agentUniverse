# !/usr/bin/env python3
# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd

# @Time    : 2024/3/28 19:28
# @Author  : wangchongshi
# @Email   : wangchongshi.wcs@antgroup.com
# @FileName: sentiment_excel_knowledge.py

from agentuniverse.agent.action.knowledge.embedding.dashscope_embedding import DashscopeEmbedding
from agentuniverse.agent.action.knowledge.embedding.openai_embedding import OpenAIEmbedding
from agentuniverse.agent.action.knowledge.knowledge import Knowledge
from agentuniverse.agent.action.knowledge.knowledge_manager import KnowledgeManager
from agentuniverse.agent.action.knowledge.reader.file.excel_reader import ExcelReader
from agentuniverse.agent.action.knowledge.reader.file.txt_reader import TxtReader
from agentuniverse.agent.action.knowledge.store.chroma_store import ChromaStore
from agentuniverse.agent.action.knowledge.store.document import Document
from langchain.text_splitter import TokenTextSplitter
from pathlib import Path

from agentuniverse.base.agentuniverse import AgentUniverse

SPLITTER = TokenTextSplitter(chunk_size=600, chunk_overlap=100)


class FsKnowledge(Knowledge):
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
            collection_name="fs_store",
            persist_path="../../DB/fs.db",
            embedding_model=DashscopeEmbedding(
                embedding_model_name='text-embedding-v2'
            ),
            dimensions=1536)
        self.reader = ExcelReader()
        # Initialize the knowledge
        self.insert_knowledge()

    def insert_knowledge(self, **kwargs) -> None:
        """Load sentiment data from Excel files and store embeddings."""
        folder_path = Path("../../resources/fs")
        excel_files = [f for f in folder_path.glob('*.xlsx') if f.is_file()]

        for file_path in excel_files:
            # Store the description as metadata
            # description_df = pd.read_excel(file_path, nrows=3)  # Load the first 3 rows as description
            # description_content = description_df.to_string(index=False)
            # description_doc = Document(test=description_content,
            #                            metadata={"file_name": file_path.name, "type": "description"})
            # self.store.insert_documents([description_doc])
            # 假设 file_path 是指向 Excel 文件的路径对象
            description_file_path = file_path.parent / f"{file_path.stem}[DES][xlsx].txt"
            # 加载描述文件内容
            txtReader = TxtReader()
            dlist = txtReader.load_data(Path(description_file_path), ext_info={"file_name": file_path.name, "type": "data"})

            # 将所有 Document 对象的 text 字段拼接为一个字符串
            description_content = "\n".join([doc.text for doc in dlist])

            # 生成描述文本的嵌入向量
            embedding = self.store.embedding_model.get_embeddings([description_content])

            # 创建 Document 对象并包含生成的 embedding
            description_doc = Document(text=description_content,
                                       metadata={"file_name": file_path.name, "type": "description"},
                                       embedding=embedding[0])  # embedding 返回的是列表，取第一个元素

            # 插入描述文档到向量数据库
            self.store.insert_documents([description_doc])
            # Load the Excel file
            # df = pd.read_excel(file_path, skiprows=3)  # Skip the first 3 rows of description
            # # Process and insert data in chunks
            # for index, chunk in df.groupby(np.arange(len(df)) // 1000):
            #     sentiment_docs = self.process_chunk(chunk, file_path.name)
            #     for doc in sentiment_docs:
            #         # 生成嵌入
            #         embeddings = self.store.embedding_model.get_embeddings([doc.text])
            #         doc.embedding = embeddings[0]  # 获取嵌入结果的第一个元素
            #         self.store.insert_documents([doc])

    def process_chunk(self, chunk, file_name):
        """Process each chunk of data to create Document objects."""
        docs = []
        for _, row in chunk.iterrows():
            content_dict = {col: str(row[col]) for col in chunk.columns}
            page_content = "\n".join([f"{k}: {v}" for k, v in content_dict.items()])
            doc = Document(text=page_content,
                           metadata={'source': file_name})
            docs.append(doc)
        return docs


if __name__ == '__main__':
    AgentUniverse().start(config_path='../../../config/config.toml')
    fs_knowledge = KnowledgeManager().get_instance_obj("fs_knowledge")

    fs_knowledge.insert_knowledge()
