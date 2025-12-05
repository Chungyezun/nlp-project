import json, re

_SENT_SPLIT = re.compile(r'(?<=[.!?])\s+')

class MuSiQueLoader:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = [json.loads(line) for line in open(file_path, "r", encoding="utf-8") if line.strip()]

    def __len__(self):
        return len(self.data)

    def get_example(self, idx):
        ex = self.data[idx]
        q = ex.get("question")
        titles = []
        docs_sentences = []
        for i, p in enumerate(ex.get("paragraphs", [])):
            title = p.get("title", f"para_{i}")
            text  = p.get("paragraph_text", "") or ""
            sents = [s for s in _SENT_SPLIT.split(text.strip()) if s] or [text]
            titles.append(title)
            docs_sentences.append(sents)
        return q, titles, docs_sentences
