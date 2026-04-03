"""
Relation extractor — Qwen 2.5 (default) or BERT backbone + entity marker strategy for head/tail.
Softmax or BCE classification for relation type.
extract_relations(text, entities) -> List[Dict] with head, tail, relation, confidence.
Single GPU only; no DataParallel.
"""

# CID - Chemical-Disease Relation and NR - No Relation (BC5CDR)
"""Original: "Aspirin may reduce the risk of heart disease."
Marked: "[E1] Aspirin [/E1] may reduce the risk of [E2] heart disease [/E2]."""


import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import torch
import numpy as np
from torch.utils.data import Dataset
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    BertConfig,
    BertForSequenceClassification,
    BertTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from typing import List, Dict, Any, Optional
import inspect

DEFAULT_REL_MODEL = "Qwen/Qwen2.5-0.5B"


def _is_qwen_backbone(model_name: str) -> bool:
    n = (model_name or "").lower()
    return "qwen" in n


def _hub_kwargs(model_name: str) -> dict:
    """Kwargs for from_pretrained (e.g. trust_remote_code for Qwen2)."""
    return {"trust_remote_code": True} if _is_qwen_backbone(model_name) else {}


def _tokenizer_kwargs(model_name: str) -> dict:
    """Kwargs for tokenizer from_pretrained. use_fast=False avoids needing sentencepiece/tiktoken for BERT."""
    out = dict(_hub_kwargs(model_name))
    if not _is_qwen_backbone(model_name):
        out["use_fast"] = False
    return out


def _remap_layernorm_gamma_beta_to_weight_bias(state_dict: dict) -> dict:
    """Remap LayerNorm.gamma -> weight, LayerNorm.beta -> bias (BioBERT checkpoints use gamma/beta)."""
    out = {}
    for k, v in state_dict.items():
        if k.endswith(".LayerNorm.gamma"):
            out[k.replace(".LayerNorm.gamma", ".LayerNorm.weight")] = v
        elif k.endswith(".LayerNorm.beta"):
            out[k.replace(".LayerNorm.beta", ".LayerNorm.bias")] = v
        else:
            out[k] = v
    return out


def _replace_layernorm_gamma_beta_with_weight_bias(module: torch.nn.Module) -> None:
    """
    Replace any LayerNorm submodules that use .gamma/.beta (BertLayerNorm style) with
    nn.LayerNorm (weight/bias) and copy values in-place. This ensures the Trainer
    saves/loads checkpoints with weight/bias so no missing/unexpected keys.
    """
    for name, child in list(module.named_children()):
        _replace_layernorm_gamma_beta_with_weight_bias(child)
        if hasattr(child, "gamma") and hasattr(child, "beta"):
            # This is a gamma/beta LayerNorm (e.g. BertLayerNorm)
            gamma = getattr(child, "gamma")
            beta = getattr(child, "beta")
            normalized_shape = gamma.shape
            eps = getattr(child, "eps", 1e-12)
            new_ln = torch.nn.LayerNorm(normalized_shape, eps=eps)
            new_ln.weight.data.copy_(gamma.data)
            new_ln.bias.data.copy_(beta.data)
            setattr(module, name, new_ln)


def _load_state_dict_from_checkpoint(checkpoint_dir: str) -> dict:
    """Load state dict from checkpoint dir (pytorch_model.bin or model.safetensors)."""
    from pathlib import Path
    p = Path(checkpoint_dir)
    if (p / "pytorch_model.bin").exists():
        return torch.load(p / "pytorch_model.bin", map_location="cpu")
    if (p / "model.safetensors").exists():
        from safetensors.torch import load_file
        return load_file(p / "model.safetensors")
    # Sharded safetensors
    import glob
    shards = sorted(glob.glob(str(p / "model*.safetensors")))
    if shards:
        from safetensors.torch import load_file
        state = {}
        for path in shards:
            state.update(load_file(path))
        return state
    raise FileNotFoundError(f"No pytorch_model.bin or model.safetensors in {checkpoint_dir}")


def _load_tokenizer(model_name: str, checkpoint_dir: str = None):
    """Load tokenizer; use BertTokenizer (slow) for BERT/BioBERT to avoid sentencepiece/tiktoken."""
    path = checkpoint_dir or model_name
    if _is_qwen_backbone(model_name or path):
        return AutoTokenizer.from_pretrained(path, **_hub_kwargs(model_name or path))
    return BertTokenizer.from_pretrained(path, use_fast=False)


class BackboneForSequenceClassification(torch.nn.Module):
    """Encoder (Qwen 2.5, BERT, etc.) + linear head for sequence classification. Used when AutoModelForSequenceClassification is not available (e.g. Qwen2)."""

    def __init__(self, model_name: str, num_labels: int):
        super().__init__()
        self.model_name = model_name
        hub_kw = _hub_kwargs(model_name)
        config = AutoConfig.from_pretrained(model_name, **hub_kw)
        config.num_labels = num_labels
        self.config = config
        self.num_labels = num_labels
        self.backbone = AutoModel.from_pretrained(model_name, **hub_kw)
        hidden_size = config.hidden_size
        self.dropout = torch.nn.Dropout(getattr(config, "hidden_dropout_prob", getattr(config, "emb_dropout_prob", 0.1)))
        self.classifier = torch.nn.Linear(hidden_size, num_labels)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        # Decoder-only (Qwen): use last non-padding token; encoder (BERT): use pooled or first token
        last_hidden = outputs.last_hidden_state
        if attention_mask is not None:
            idx = (attention_mask.sum(1) - 1).clamp(0).unsqueeze(1).unsqueeze(2).expand(-1, -1, last_hidden.size(-1))
            pooled = last_hidden.gather(1, idx).squeeze(1)
        else:
            pooled = last_hidden[:, -1, :]
        if pooled.dtype != self.classifier.weight.dtype:
            pooled = pooled.to(self.classifier.weight.dtype)
        logits = self.classifier(self.dropout(pooled))
        loss = None
        if labels is not None:
            if self.num_labels == 1:
                loss = torch.nn.functional.binary_cross_entropy_with_logits(logits.view(-1), labels.view(-1).float())
            else:
                loss = torch.nn.functional.cross_entropy(logits, labels.long().view(-1))
        return type("Output", (), {"loss": loss, "logits": logits})()

    def save_pretrained(self, save_directory: str):
        from pathlib import Path
        import json
        p = Path(save_directory)
        p.mkdir(parents=True, exist_ok=True)
        (p / "rel_config.json").write_text(json.dumps({
            "use_backbone_rel": True,
            "model_name": self.model_name,
            "num_labels": self.num_labels,
            "label_list": getattr(self, "label_list", None),
        }))
        torch.save(self.state_dict(), p / "pytorch_model.bin")

    @classmethod
    def from_pretrained(cls, save_directory: str):
        from pathlib import Path
        import json
        p = Path(save_directory)
        cfg = json.loads((p / "rel_config.json").read_text())
        model = cls(cfg["model_name"], cfg["num_labels"])
        model.load_state_dict(torch.load(p / "pytorch_model.bin", map_location="cpu"))
        return model

# Support both older (evaluation_strategy) and newer (eval_strategy) transformers
try:
    _params = inspect.signature(TrainingArguments).parameters
    _EVAL_STRATEGY_KEY = "eval_strategy" if "eval_strategy" in _params else "evaluation_strategy"
except Exception:
    _EVAL_STRATEGY_KEY = "evaluation_strategy"

# Binary relation (0=no relation, 1=relation): use BCEWithLogitsLoss; labels must be FloatTensor, shape (batch_size, 1).

def _mark_entities(text: str, head: Dict, tail: Dict) -> str:
    """
    Insert entity markers for relation classification.

    Marker style matters: training/example strings in this repo use a *space-padded* format:
      "[E1] Aspirin [/E1] ... [E2] gastric bleeding [/E2]"

    Set REL_MARK_STYLE=compact to force the old no-space format:
      "[E1]Aspirin[/E1]..."
    """
    style = os.environ.get("REL_MARK_STYLE", "spaced").strip().lower()
    spaced = style != "compact"
    first, second = (head, tail) if head["start"] <= tail["start"] else (tail, head)
    (o1, c1), (o2, c2) = ("[E1]", "[/E1]"), ("[E2]", "[/E2]")
    if head["start"] > tail["start"]:
        (o1, c1), (o2, c2) = (o2, c2), (o1, c1)
    parts = []
    last = 0
    for start, end, (o, c) in [
        (first["start"], first["end"], (o1, c1)),
        (second["start"], second["end"], (o2, c2)),
    ]:
        parts.append(text[last:start])
        if spaced:
            parts.append(o + " ")
            parts.append(text[start:end])
            parts.append(" " + c)
        else:
            parts.append(o)
            parts.append(text[start:end])
            parts.append(c)
        last = end
    parts.append(text[last:])
    return "".join(parts)


def build_relation_examples(
    examples: List[Dict],
    relation_key: str = "type",
    add_negatives: bool = True,
    max_negatives_ratio: float = 1.0,
) -> List[Dict]:
    """
    Build relation examples from documents. If add_negatives=True, adds NR (no-relation)
    pairs for chemical–disease pairs not in the gold relations, so the model sees both classes.
    """
    out = []
    for ex in examples:
        text = ex["text"]
        entities = ex.get("entities", [])
        relations = ex.get("relations", [])
        positive_pairs = set()
        for rel in relations:
            h, t = rel["head"], rel["tail"]
            if h >= len(entities) or t >= len(entities):
                continue
            positive_pairs.add((h, t))
            out.append({
                "text": text,
                "head": entities[h],
                "tail": entities[t],
                "label": rel.get(relation_key, "CID"),
            })
        # Add negative examples (chemical–disease pairs not in relations) so eval is not trivial.
        if add_negatives and entities and max_negatives_ratio > 0:
            # BC5CDR: type is "Chemical" or "Disease"; assume chemical–disease pairs are valid candidates.
            chem_ids = [i for i, e in enumerate(entities) if e.get("type") == "Chemical"]
            dis_ids = [i for i, e in enumerate(entities) if e.get("type") == "Disease"]
            if not chem_ids or not dis_ids:
                chem_ids = list(range(len(entities)))
                dis_ids = list(range(len(entities)))
            max_neg = max(1, int(len(positive_pairs) * max_negatives_ratio))
            added = 0
            for h in chem_ids:
                for t in dis_ids:
                    if h == t or (h, t) in positive_pairs or (t, h) in positive_pairs:
                        continue
                    out.append({
                        "text": text,
                        "head": entities[h],
                        "tail": entities[t],
                        "label": "NR",
                    })
                    added += 1
                    if added >= max_neg:
                        break
                if added >= max_neg:
                    break
    return out


class _RelationDataset(Dataset):
    def __init__(self, examples, tokenizer, label2id, max_len=128, use_binary_bce: bool = False):
        self.use_binary_bce = use_binary_bce
        self.data = []
        for ex in examples:
            marked = _mark_entities(ex["text"], ex["head"], ex["tail"])
            enc = tokenizer(
                marked,
                truncation=True,
                max_length=max_len,
                padding="max_length",
                return_tensors=None,
            )
            label_id = label2id.get(ex["label"], 0)
            self.data.append({
                "input_ids": enc["input_ids"],
                "attention_mask": enc["attention_mask"],
                "labels": label_id,
            })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        d = self.data[i]
        # BCEWithLogitsLoss requires FloatTensor; shape (1,) so batch is (batch_size, 1).
        if self.use_binary_bce:
            labels = torch.tensor(d["labels"], dtype=torch.float32).unsqueeze(-1)
        else:
            labels = torch.tensor(d["labels"], dtype=torch.long)
        return {
            "input_ids": torch.tensor(d["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(d["attention_mask"], dtype=torch.long),
            "labels": labels,
        }


class _BCERelationTrainer(Trainer):
    """Trainer that uses BCEWithLogitsLoss for binary relation (labels float, shape (B, 1))."""

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits  # (B, 1)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits.view(-1), labels.view(-1).float()
        )
        return (loss, outputs) if return_outputs else loss


class BERTRelationModel:
    def __init__(
        self,
        model_name: str = DEFAULT_REL_MODEL,
        label_list: Optional[List[str]] = None,
        use_binary_bce: bool = True,
    ):
        self.model_name = model_name
        self.tokenizer = _load_tokenizer(model_name)
        self.label_list = label_list or ["NR", "CID"]
        self.label2id = {l: i for i, l in enumerate(self.label_list)}
        self.id2label = {i: l for l, i in self.label2id.items()}
        self.model = None
        self.use_binary_bce = use_binary_bce

    def prepare_from_data(self, relation_examples: List[Dict]):
        labels = set(ex["label"] for ex in relation_examples)
        self.label_list = sorted(labels)
        # Binary BCE: 0 = no relation (NR), 1 = relation (CID); one output, BCEWithLogitsLoss.
        if self.use_binary_bce and len(self.label_list) == 2:
            self._num_labels = 1
            # Order so NR=0, CID=1 (positive class last).
            if "NR" in self.label_list and "CID" in self.label_list:
                self.label_list = ["NR", "CID"]
            self.label2id = {l: i for i, l in enumerate(self.label_list)}
            self.id2label = {i: l for l, i in self.label2id.items()}
        else:
            self._num_labels = len(self.label_list)
            self.label2id = {l: i for i, l in enumerate(self.label_list)}
            self.id2label = {i: l for l, i in self.label2id.items()}
        if _is_qwen_backbone(self.model_name):
            self.model = BackboneForSequenceClassification(self.model_name, self._num_labels)
            self.model.label_list = list(self.label_list)
        else:
            self.model = BertForSequenceClassification.from_pretrained(
                self.model_name, num_labels=self._num_labels
            )
            # BioBERT (and some Hub checkpoints) use LayerNorm with gamma/beta; Trainer expects weight/bias.
            # Replace those modules in-place so saved checkpoints use weight/bias and load_best_model_at_end works.
            _replace_layernorm_gamma_beta_with_weight_bias(self.model)
        return self

    def train(
        self,
        relation_examples: List[Dict],
        dev_examples: Optional[List[Dict]] = None,
        output_dir: str = "./out_rel",
        num_epochs: int = 3,
        batch_size: int = 16,
        lr: float = 2e-5,
    ):
        if self.model is None:
            self.prepare_from_data(relation_examples)
        use_bce = getattr(self, "_num_labels", len(self.label_list)) == 1
        train_ds = _RelationDataset(
            relation_examples, self.tokenizer, self.label2id, use_binary_bce=use_bce
        )
        eval_ds = (
            _RelationDataset(dev_examples, self.tokenizer, self.label2id, use_binary_bce=use_bce)
            if dev_examples
            else None
        )
        args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=lr,
            **{_EVAL_STRATEGY_KEY: "epoch" if eval_ds else "no"},
            save_strategy="epoch",
            load_best_model_at_end=bool(eval_ds),
            metric_for_best_model="f1",
        )

        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            from sklearn.metrics import f1_score, accuracy_score
            if isinstance(labels, torch.Tensor):
                labels = labels.cpu().numpy()
            logits = np.array(logits)
            if logits.ndim == 2 and logits.shape[-1] == 1:
                # BCE: logits (N, 1), labels float (N, 1)
                preds = (logits.squeeze(-1) > 0).astype(np.int64)
                labels_int = labels.squeeze().astype(np.int64)
            else:
                preds = np.argmax(logits, axis=-1)
                labels_int = labels.squeeze().astype(np.int64)
            return {
                "accuracy": float(accuracy_score(labels_int, preds)),
                "f1": float(f1_score(labels_int, preds, average="weighted", zero_division=0)),
            }

        trainer_cls = _BCERelationTrainer if use_bce else Trainer
        trainer = trainer_cls(
            model=self.model,
            args=args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            compute_metrics=compute_metrics,
        )
        trainer.train()
        trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        return trainer

    def load(self, checkpoint_dir: str):
        from pathlib import Path
        p = Path(checkpoint_dir)
        if (p / "rel_config.json").exists():
            import json
            cfg = json.loads((p / "rel_config.json").read_text())
            if cfg.get("use_backbone_rel"):
                self.model = BackboneForSequenceClassification.from_pretrained(checkpoint_dir)
                self.tokenizer = _load_tokenizer(cfg.get("model_name", ""), checkpoint_dir)
                self._num_labels = self.model.num_labels
                self.label_list = list(cfg.get("label_list") or (["NR", "CID"] if self._num_labels == 1 else list(range(self._num_labels))))
                self.id2label = {i: self.label_list[i] for i in range(len(self.label_list))}
                self.label2id = {v: k for k, v in self.id2label.items()}
                return self
        # Load config and weights; remap LayerNorm gamma/beta -> weight/bias for BioBERT checkpoints.
        state = _load_state_dict_from_checkpoint(checkpoint_dir)
        state = _remap_layernorm_gamma_beta_to_weight_bias(state)
        if "classifier.weight" not in state:
            # Checkpoint has wrong architecture (e.g. MLM head cls.* instead of classifier). Load with HF and warn.
            print("[WARNING] Relation checkpoint missing classifier.* (has cls.*/pooler?). Loading with from_pretrained; classifier may be wrong. Re-train relation model.")
            self.model = BertForSequenceClassification.from_pretrained(checkpoint_dir, ignore_mismatched_sizes=True)
        else:
            config = BertConfig.from_pretrained(checkpoint_dir)
            self.model = BertForSequenceClassification(config)
            self.model.load_state_dict(state, strict=True)
        self.tokenizer = _load_tokenizer("bert", checkpoint_dir)
        self.id2label = {int(k): v for k, v in self.model.config.id2label.items()}
        self.label2id = {v: k for k, v in self.id2label.items()}
        self.label_list = list(self.id2label.values())
        self._num_labels = getattr(self.model.config, "num_labels", len(self.label_list))
        return self

    def extract_relations(
        self,
        text: str,
        entities: List[Dict[str, Any]],
        *,
        positive_threshold: float | None = None,
    ) -> List[Dict[str, Any]]:
        """
        Return list of {"head": entity_id, "tail": entity_id, "relation": str, "confidence": float}.
        entity_id is index into entities.
        """
        if self.model is None:
            raise RuntimeError("Model not loaded; train or load first.")
        self.model.eval()
        if positive_threshold is None:
            try:
                positive_threshold = float(os.environ.get("REL_POS_THRESHOLD", "0.5"))
            except Exception:
                positive_threshold = 0.5
        debug_scores = os.environ.get("REL_DEBUG_SCORES", "").strip().lower() in ("1", "true", "yes", "y")
        include_negative = os.environ.get("REL_INCLUDE_NR", "").strip().lower() in ("1", "true", "yes", "y")
        out = []
        for i, h in enumerate(entities):
            for j, t in enumerate(entities):
                if i >= j:
                    continue
                marked = _mark_entities(text, h, t)
                enc = self.tokenizer(
                    marked,
                    truncation=True,
                    max_length=128,
                    return_tensors="pt",
                )
                with torch.no_grad():
                    device = next(self.model.parameters()).device
                    logits = self.model(
                        input_ids=enc["input_ids"].to(device),
                        attention_mask=enc["attention_mask"].to(device),
                    ).logits
                num_labels = getattr(self.model.config, "num_labels", 2)
                if num_labels == 1:
                    # BCE: single logit, positive = chemical–disease relation (CID) per train.py.
                    # HF checkpoints often only define id2label {0: "LABEL_0"}; id2label.get(1) was always
                    # missing → "NR" → zero extracted triples for Baselines 1 & 2.
                    prob = torch.sigmoid(logits[0, 0]).item()
                    if prob >= float(positive_threshold):
                        rel = "CID"
                        conf = prob
                    else:
                        rel = "NR"
                        conf = 1.0 - prob
                    if debug_scores:
                        print(
                            f"[REL_DEBUG_SCORES] pair=({i},{j}) prob_pos={prob:.4f} thr={float(positive_threshold):.4f} pred={rel}",
                            file=os.sys.stderr,
                        )
                else:
                    probs = torch.softmax(logits, dim=-1)[0]
                    pred_id = logits.argmax(-1).item()
                    # Some configs only store LABEL_0; treat pred_id==1 as positive even if id2label is incomplete.
                    if pred_id == 1 and ("LABEL_1" not in (self.id2label or {}).values()):
                        rel = "LABEL_1"
                    else:
                        rel = self.id2label.get(pred_id, "NR")
                    conf = probs[pred_id].item()
                    if debug_scores:
                        print(
                            f"[REL_DEBUG_SCORES] pair=({i},{j}) pred_id={pred_id} conf={conf:.4f} pred={rel}",
                            file=os.sys.stderr,
                        )
                if rel != "NR" or include_negative:
                    out.append({"head": i, "tail": j, "relation": rel, "confidence": conf})
        return out
