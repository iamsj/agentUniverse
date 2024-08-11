# !/usr/bin/env python3
# -*- coding:utf-8 -*-

# @Time    : 2024/3/18 14:21
# @Author  : wangchongshi
# @Email   : wangchongshi.wcs@antgroup.com
# @FileName: excel_reader.py
from pathlib import Path
from typing import List, Optional, Dict

import pandas as pd

from agentuniverse.agent.action.knowledge.reader.reader import Reader
from agentuniverse.agent.action.knowledge.store.document import Document


class ExcelReader(Reader):
    """Excel reader."""

    def load_data(self, file: Path, ext_info: Optional[Dict] = None) -> List[Document]:
        """Parse the Excel file.

        Note:
            `pandas` is required to read Excel files: `pip install pandas`
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError(
                "pandas is required to read Excel files: `pip install pandas`"
            )

        # Load the Excel file into a dictionary of DataFrames
        sheets = pd.read_excel(file, sheet_name=None)

        documents = []
        for sheet_name, df in sheets.items():
            # Convert DataFrame to string
            text = df.to_string()

            metadata = {"sheet_name": sheet_name, "file_name": file.name}
            if ext_info is not None:
                metadata.update(ext_info)

            documents.append(Document(text=text, metadata=metadata))

        return documents
