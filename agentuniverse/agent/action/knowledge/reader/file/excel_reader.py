from pathlib import Path
from typing import Optional, Dict, List

import pandas as pd
import numpy as np

from agentuniverse.agent.action.knowledge.reader.reader import Reader
from agentuniverse.agent.action.knowledge.store.document import Document


class ExcelReader(Reader):
    def load_data(self, file: Path, ext_info: Optional[Dict] = None, batch_size=1000) -> List[Document]:
        """Parse the Excel file in batches.

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

        # Concatenate all sheets into one DataFrame
        df = pd.concat(sheets.values(), keys=sheets.keys(), names=['sheet_name']).reset_index()

        # Add file name to metadata
        df['file_name'] = file.name

        if ext_info is not None:
            for key, value in ext_info.items():
                df[key] = value

        return df
