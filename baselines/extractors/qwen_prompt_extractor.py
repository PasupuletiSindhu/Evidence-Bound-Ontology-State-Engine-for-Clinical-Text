# Qwen prompt -> (subject, relation, object). One model, no NER/relation.
import re
from typing import Dict, List, Tuple, Optional

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    _TRANSFORMERS_AVAILABLE = True
except ImportError:
    _TRANSFORMERS_AVAILABLE = False

DEFAULT_QWEN_MODEL = "Qwen/Qwen2.5-0.5B"

_EXTRACT_PROMPT = """Extract knowledge triples (subject, relation, object) from the following biomedical or clinical text. Output one triple per line in the format: subject | relation | object. If there are no triples, output "none".

Text:
{text}

Triples:
"""


def _parse_triple_lines(output: str) -> List[Tuple[str, str, str]]:
    triples = []
    for line in output.strip().splitlines():
        line = line.strip()
        if not line or line.lower() == "none" or "|" not in line:
            continue
        parts = [p.strip() for p in line.split("|", 2)]
        if len(parts) >= 3 and parts[0] and parts[2]:
            triples.append((parts[0], parts[1] or "related_to", parts[2]))
        elif len(parts) == 2 and parts[0] and parts[1]:
            triples.append((parts[0], "related_to", parts[1]))
    return triples


class QwenPromptExtractor:
    def __init__(
        self,
        model_name: str = DEFAULT_QWEN_MODEL,
        max_new_tokens: int = 256,
        do_sample: bool = False,
        device: Optional[str] = None,
    ):
        if not _TRANSFORMERS_AVAILABLE:
            raise RuntimeError("transformers and torch required for QwenPromptExtractor")
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.do_sample = do_sample
        hub_kw = {"trust_remote_code": True} if "qwen" in (model_name or "").lower() else {}
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, **hub_kw)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **hub_kw)
        self.model.eval()
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self._device)

    def extract(self, text: str) -> List[Tuple[str, str, str]]:
        """Return list of (subject, relation, object) triples from text."""
        if not text or not text.strip():
            return []
        prompt = _EXTRACT_PROMPT.format(text=text.strip()[:4000])
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        ).to(self._device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=self.do_sample,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        decoded = self.tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return _parse_triple_lines(decoded)

    def __call__(self, text: str) -> List[Tuple[str, str, str]]:
        return self.extract(text)

    def generate_from_messages(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: int = 128,
    ) -> str:
        """
        Chat-style generation for QA / reasoning (not triple extraction).
        messages: list of {"role": "system"|"user"|"assistant", "content": str}
        """
        if not messages:
            return ""
        if self.tokenizer.chat_template is None:
            # Fallback: concatenate
            prompt = "\n\n".join(f"{m['role']}: {m['content']}" for m in messages) + "\nassistant:"
        else:
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=3072,
        ).to(self._device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=self.do_sample,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        gen = out[0][inputs["input_ids"].shape[1] :]
        return self.tokenizer.decode(gen, skip_special_tokens=True).strip()


def qwen_extract_triples(
    text: str,
    model_name: str = DEFAULT_QWEN_MODEL,
    max_new_tokens: int = 256,
) -> List[Tuple[str, str, str]]:
    """One-off extraction using Qwen 2.5. Loads model each call; for repeated use, use QwenPromptExtractor."""
    ex = QwenPromptExtractor(model_name=model_name, max_new_tokens=max_new_tokens)
    return ex.extract(text)
