from langchain_text_splitters import RecursiveCharacterTextSplitter

def get_chunker():
    """
    Returns the explicitly configured text splitter for ECHELON.
    Using chunk_size=500 and chunk_overlap=50 to fit comfortably within MiniLM context.
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
        is_separator_regex=False,
    )
