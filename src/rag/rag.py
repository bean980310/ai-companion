from langchain_community.document_loaders import DirectoryLoader, TextLoader

loader = DirectoryLoader(path='characters_info', glob="**/*.txt", loader_cls=TextLoader)

data=loader.load()
len(data)