import os
import re
from langchain.docstore.document import Document
from langchain_community.document_loaders import DirectoryLoader, TextLoader

loader = DirectoryLoader(path='characters_info', glob="*.txt", loader_cls=TextLoader)

data=loader.load()
data=data.sort(key=lambda x: x.metadata['source'].split('/')[-1])
len(data)