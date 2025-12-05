def build_doc_text(title, sents):
    parts = []
    if title:
        parts.append(title.strip())
    if sents:
        parts.append(" ".join(s.strip() for s in sents if s.strip()))
    return ". ".join(parts).strip()


def merge_two_docs(text_i, text_j, sep="[SEP]"):
    return f"{text_i} {sep} {text_j}"
