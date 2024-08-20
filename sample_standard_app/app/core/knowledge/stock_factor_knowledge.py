# !/usr/bin/env python3
# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd

# @Time    : 2024/3/28 19:28
# @Author  : wangchongshi
# @Email   : wangchongshi.wcs@antgroup.com
# @FileName: sentiment_excel_knowledge.py

from agentuniverse.agent.action.knowledge.embedding.dashscope_embedding import DashscopeEmbedding
from agentuniverse.agent.action.knowledge.embedding.ollama_embedding import OllamaEmbedding
from agentuniverse.agent.action.knowledge.embedding.openai_embedding import OpenAIEmbedding
from agentuniverse.agent.action.knowledge.knowledge import Knowledge
from agentuniverse.agent.action.knowledge.reader.file.excel_reader import ExcelReader
from agentuniverse.agent.action.knowledge.reader.file.txt_reader import TxtReader
from agentuniverse.agent.action.knowledge.store.chroma_store import ChromaStore
from agentuniverse.agent.action.knowledge.store.document import Document
from langchain.text_splitter import TokenTextSplitter
from pathlib import Path

SPLITTER = TokenTextSplitter(chunk_size=600, chunk_overlap=100)


class StockFactorKnowledge(Knowledge):
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
            collection_name="stock_factor_store",
            persist_path="../../DB/stock_factor.db",
            embedding_model=OllamaEmbedding(
                embedding_model_name='mxbai-embed-large'
            ),
            dimensions=1536)
        self.reader = ExcelReader()
        # Initialize the knowledge
        self.insert_knowledge()

    def insert_knowledge(self, **kwargs) -> None:
        """Load sentiment data from Excel files and store embeddings."""
        folder_path = Path("../../resources/stock_factor")
        excel_files = [f for f in folder_path.glob('*.xlsx') if f.is_file()]

        for file_path in excel_files:
            self.save_desc(file_path)
            # 读取前3行作为列名
            col_names = pd.read_excel(file_path, nrows=3, header=None)
            # 拼接列名
            new_col_names = col_names.apply(lambda x: '_'.join(x.dropna().astype(str)), axis=0)

            # Load the Excel file
            df = pd.read_excel(file_path, skiprows=3)  # Skip the first 3 rows of description
            df.columns = new_col_names
            # Process and insert data in chunks
            for index, chunk in df.groupby(np.arange(len(df)) // 1000):
                sentiment_docs = self.process_chunk(chunk, file_path.name)
                for doc in sentiment_docs:
                    # 检查文本长度并进行处理
                    text = doc.text
                    if len(text) > 2048:
                        text = text[:2048]  # 截断文本到2048字符以内

                    # 生成嵌入
                    embeddings = self.store.embedding_model.get_embeddings([text])
                    doc.embedding = embeddings[0]  # 获取嵌入结果的第一个元素

                    # 插入文档到数据库
                    self.store.insert_documents([doc])

    def process_chunk(self, chunk,file_name):
        """Process each chunk of data to create Document objects."""
        docs = []
        for _, row in chunk.iterrows():
            content_dict = {col: str(row[col]) for col in chunk.columns}
            page_content = "\n".join([f"{k}: {v}" for k, v in content_dict.items()])
            doc = Document(text=page_content,
                           metadata={'source': file_name})
            docs.append(doc)
        return docs

    def save_desc(self, file_path: Path):
        # 定义嵌入API允许的最大输入长度
        MAX_INPUT_LENGTH = 2048

        # 构建描述文件的路径
        description_file_path = file_path.parent / f"{file_path.stem}[DES][xlsx].txt"

        # 加载描述文件内容
        txtReader = TxtReader()
        dlist = txtReader.load_data(Path(description_file_path), ext_info={"file_name": file_path.name, "type": "data"})

        # 将所有 Document 对象的 text 字段拼接为一个字符串
        description_content = "\n".join([doc.text for doc in dlist])

        # 将文本分割以确保每个部分都符合输入长度要求
        lc_doc_list = SPLITTER.split_documents(Document.as_langchain_list(dlist))

        # 检查并处理每个文档的长度
        valid_lc_doc_list = []
        for doc in lc_doc_list:
            if 1 <= len(doc.page_content) <= MAX_INPUT_LENGTH:
                valid_lc_doc_list.append(doc)
            else:
                # 如果文本长度超过最大限制，进行截断
                truncated_text = doc.text[:MAX_INPUT_LENGTH]
                valid_lc_doc_list.append(Document(text=truncated_text, metadata=doc.metadata))

        # 提取文本字段以传递给嵌入API
        valid_texts = [doc.page_content for doc in valid_lc_doc_list]

        # 生成每个有效文档的嵌入
        embedding = self.store.embedding_model.get_embeddings(valid_texts)

        # 创建包含生成嵌入的 Document 对象
        description_doc = Document(text=description_content,
                                   metadata={"file_name": file_path.name, "type": "description"},
                                   embedding=embedding[0])  # embedding 返回的是列表，取第一个元素

        # 将描述文档插入到向量数据库
        self.store.insert_documents([description_doc])

        # 将每个有效的文档插入到向量数据库
        self.store.insert_documents(Document.from_langchain_list(valid_lc_doc_list))
