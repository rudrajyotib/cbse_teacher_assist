import re

# --- tokenizer helpers ---
class TokenEncoder:
    def __init__(self, prefer_tiktoken=True, model_name="cl100k_base"):
        self.use_tiktoken = False
        self.enc = None
        if prefer_tiktoken:
            try:
                import tiktoken
                try:
                    # best: encoding aligned with OpenAI embeddings
                    self.enc = tiktoken.encoding_for_model("text-embedding-3-small")
                except Exception:
                    self.enc = tiktoken.get_encoding(model_name)
                self.use_tiktoken = True
            except Exception:
                pass
        if not self.use_tiktoken:
            # fallback to HF GPT-2 tokenizer
            from transformers import GPT2TokenizerFast
            self.enc = GPT2TokenizerFast.from_pretrained("gpt2")

    def encode(self, text: str):
        return self.enc.encode(text) if self.use_tiktoken else self.enc.encode(text)

    def decode(self, tokens):
        return self.enc.decode(tokens) if self.use_tiktoken else self.enc.decode(tokens)


# optional: plug in spaCy for stronger sentence splitting (set use_spacy=True)
_nlp = None
def _maybe_load_spacy():
    global _nlp
    if _nlp is None:
        try:
            import spacy
            _nlp = spacy.load("en_core_web_sm")
        except Exception:
            _nlp = False
    return _nlp

def _split_into_sentences(text: str, use_spacy=False):
    if use_spacy and _maybe_load_spacy():
        return [s.text.strip() for s in _nlp(text).sents if s.text.strip()]
    # regex fallback
    return [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]

def _subsplit_long_sentence(sent, enc: TokenEncoder, max_tokens: int):
    """
    Break a too-long sentence into smaller pieces, preferring punctuation/commas/semicolons,
    then whitespace, then fixed-size token chunks as a last resort.
    """
    toks = enc.encode(sent)
    if len(toks) <= max_tokens:
        return [sent]

    # Try punctuation-based splits first
    parts = re.split(r'([;,:])', sent)  # keep delimiters
    if len(parts) > 1:
        # Re-stitch keeping delimiters with preceding text
        stitched = []
        cur = ""
        for i, p in enumerate(parts):
            if i % 2 == 0:
                cur += p
            else:
                cur += p  # append delimiter
                stitched.append(cur.strip())
                cur = ""
        if cur.strip():
            stitched.append(cur.strip())

        out = []
        cur_tokens = []
        cur_texts = []
        for piece in stitched:
            ptoks = enc.encode(piece)
            if len(cur_tokens) + len(ptoks) > max_tokens:
                if cur_tokens:
                    out.append(enc.decode(cur_tokens))
                cur_tokens, cur_texts = ptoks[:], [piece]
            else:
                cur_tokens.extend(ptoks)
                cur_texts.append(piece)
        if cur_tokens:
            out.append(enc.decode(cur_tokens))
        return out

    # Otherwise, split on whitespace chunks
    words = sent.split()
    out, cur = [], []
    for w in words:
        test = " ".join(cur + [w])
        if len(enc.encode(test)) > max_tokens:
            if cur:
                out.append(" ".join(cur))
                cur = [w]
            else:
                # single word too long (rare with English) -> hard token slice
                break
        else:
            cur.append(w)
    if cur: out.append(" ".join(cur))

    # Final guard: if still too long, hard slice by tokens
    final = []
    for piece in out:
        ptoks = enc.encode(piece)
        if len(ptoks) <= max_tokens:
            final.append(piece)
        else:
            for i in range(0, len(ptoks), max_tokens):
                final.append(enc.decode(ptoks[i:i+max_tokens]))
    return final


# --- Chunking ---
def chunk_paragraphs(
    text: str,
    max_tokens: int = 300,
    overlap: int = 50,
    use_spacy: bool = False,
    encoder: TokenEncoder = None,
):
    """
    Token-aware chunker:
    - Splits on paragraphs -> sentences
    - Builds chunks up to max_tokens with token overlap
    - Overlap is measured in tokens, not characters
    """
    assert overlap < max_tokens, "overlap must be smaller than max_tokens"
    enc = encoder or TokenEncoder(prefer_tiktoken=True)

    # split paragraphs; keep only meaningful ones
    paragraphs = [p.strip() for p in re.split(r'\n{2,}', text) if len(p.strip()) > 20]
    chunks = []

    for para in paragraphs:
        sents = _split_into_sentences(para, use_spacy=use_spacy)

        # expand any ultra-long sentence
        expanded_sents = []
        for s in sents:
            if len(enc.encode(s)) > max_tokens:
                expanded_sents.extend(_subsplit_long_sentence(s, enc, max_tokens))
            else:
                expanded_sents.append(s)

        chunk_tokens = []
        for s in expanded_sents:
            stoks = enc.encode(s)
            if len(chunk_tokens) + len(stoks) > max_tokens:
                if chunk_tokens:
                    # finalize current
                    chunks.append(enc.decode(chunk_tokens))
                    # start next with token overlap
                    if overlap > 0:
                        chunk_tokens = chunk_tokens[-overlap:]
                    else:
                        chunk_tokens = []
                # Now add sentence (may still be > max_tokens if weird; handled above)
                if len(stoks) > max_tokens:
                    # defensive: hard slice
                    for i in range(0, len(stoks), max_tokens):
                        piece = stoks[i:i+max_tokens]
                        if chunk_tokens:
                            # finalize previous overlap-only chunk first
                            chunks.append(enc.decode(chunk_tokens))
                            chunk_tokens = []
                        chunks.append(enc.decode(piece))
                    # seed next chunk with overlap from the last piece
                    if overlap > 0:
                        chunk_tokens = piece[-overlap:]
                    continue
                chunk_tokens.extend(stoks)
            else:
                chunk_tokens.extend(stoks)

        if chunk_tokens:
            chunks.append(enc.decode(chunk_tokens))

    # Optional: merge tiny trailing chunks into previous if they’re too small
    min_tokens = max(1, overlap)  # avoid very tiny final chunks
    merged = []
    enc_local = enc
    for ch in chunks:
        if not merged:
            merged.append(ch)
            continue
        prev = merged[-1]
        if len(enc_local.encode(ch)) < min_tokens and len(enc_local.encode(prev)) + len(enc_local.encode(ch)) <= max_tokens:
            merged[-1] = prev + " " + ch
        else:
            merged.append(ch)

    return merged
