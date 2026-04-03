"""
NER backbone — HuggingFace Transformers (Qwen 2.5 by default; BERT supported).
Token classification, training loop, evaluation (precision, recall, F1).
Inference: extract_entities(text) -> List[Dict] with text, start, end, label, confidence.
Option to freeze encoder layers. Single GPU only; no DataParallel.
"""


"""State: The current snapshot of the extracted knowledge graph (the set of triples at time t); 
a trajectory is a sequence of such states."""



import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    AutoModelForTokenClassification,
    BertConfig,
    BertModel,
    BertTokenizer,
    BertForTokenClassification,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    DataCollatorForTokenClassification,
)
from collections import Counter
import inspect

# Support both older (evaluation_strategy) and newer (eval_strategy) transformers
try:
    _params = inspect.signature(TrainingArguments).parameters
    _EVAL_STRATEGY_KEY = "eval_strategy" if "eval_strategy" in _params else "evaluation_strategy"
except Exception:
    _EVAL_STRATEGY_KEY = "evaluation_strategy"
from seqeval.metrics import f1_score, precision_score, recall_score
from typing import List, Dict, Any, Optional, Tuple

try:
    from torchcrf import CRF as TorchCRF
    _CRF_AVAILABLE = True
except ImportError:
    TorchCRF = None
    _CRF_AVAILABLE = False

# UMLS semantic type (Txxx) -> SemGroup abbreviation for reduced label set (MedMentions).
# Covers types in SemGroups.txt so ~15 labels instead of 125+.
UMLS_T_TO_GROUP: Dict[str, str] = {
    "T052": "ACTI", "T053": "ACTI", "T056": "ACTI", "T051": "ACTI", "T064": "ACTI",
    "T055": "ACTI", "T066": "ACTI", "T057": "ACTI", "T054": "ACTI",
    "T017": "ANAT", "T029": "ANAT", "T023": "ANAT", "T030": "ANAT", "T031": "ANAT",
    "T022": "ANAT", "T025": "ANAT", "T026": "ANAT", "T018": "ANAT", "T021": "ANAT", "T024": "ANAT",
    "T116": "CHEM", "T195": "CHEM", "T123": "CHEM", "T122": "CHEM", "T103": "CHEM",
    "T120": "CHEM", "T104": "CHEM", "T200": "CHEM", "T196": "CHEM", "T126": "CHEM",
    "T131": "CHEM", "T125": "CHEM", "T129": "CHEM", "T130": "CHEM", "T197": "CHEM",
    "T114": "CHEM", "T109": "CHEM", "T121": "CHEM", "T192": "CHEM", "T127": "CHEM",
    "T185": "CONC", "T077": "CONC", "T169": "CONC", "T102": "CONC", "T078": "CONC",
    "T170": "CONC", "T171": "CONC", "T080": "CONC", "T081": "CONC", "T089": "CONC", "T082": "CONC", "T079": "CONC",
    "T203": "DEVI", "T074": "DEVI", "T075": "DEVI",
    "T020": "DISO", "T190": "DISO", "T049": "DISO", "T019": "DISO", "T047": "DISO",
    "T050": "DISO", "T033": "DISO", "T037": "DISO", "T048": "DISO", "T191": "DISO", "T046": "DISO", "T184": "DISO",
    "T087": "GENE", "T088": "GENE", "T028": "GENE", "T085": "GENE", "T086": "GENE",
    "T083": "GEOG",
    "T100": "LIVB", "T011": "LIVB", "T008": "LIVB", "T194": "LIVB", "T007": "LIVB",
    "T012": "LIVB", "T204": "LIVB", "T099": "LIVB", "T013": "LIVB", "T004": "LIVB",
    "T096": "LIVB", "T016": "LIVB", "T015": "LIVB", "T001": "LIVB", "T101": "LIVB",
    "T002": "LIVB", "T098": "LIVB", "T097": "LIVB", "T014": "LIVB", "T010": "LIVB", "T005": "LIVB",
    "T071": "OBJC", "T168": "OBJC", "T073": "OBJC", "T072": "OBJC", "T167": "OBJC",
    "T091": "OCCU", "T090": "OCCU",
    "T093": "ORGA", "T092": "ORGA", "T094": "ORGA", "T095": "ORGA",
    "T038": "PHEN", "T069": "PHEN", "T068": "PHEN", "T034": "PHEN", "T070": "PHEN", "T067": "PHEN",
    "T043": "PHYS", "T201": "PHYS", "T045": "PHYS", "T041": "PHYS", "T044": "PHYS",
    "T032": "PHYS", "T040": "PHYS", "T042": "PHYS", "T039": "PHYS",
    "T060": "PROC", "T065": "PROC", "T058": "PROC", "T059": "PROC", "T063": "PROC",
    "T062": "PROC", "T061": "PROC",
}


def _normalize_entity_tag(raw: str) -> str:
    """Single label for BIO: take first semantic type if comma-separated (MedMentions)."""
    if not raw or raw == "O":
        return ""
    tag = (raw.strip().split(",")[0] or "").strip()[:30]
    return tag if tag else ""


def _resolve_tag(raw: str, semantic_group_map: Optional[Dict[str, str]]) -> str:
    tag = _normalize_entity_tag(raw)
    if not tag:
        return ""
    if semantic_group_map is not None:
        tag = semantic_group_map.get(tag, "OTHER")
    return tag


# 3-label mode for capstone (graph/state/drift): B-ENTITY, I-ENTITY, O only.
ENTITY_ONLY_LABEL_LIST = ["B-ENTITY", "I-ENTITY", "O"]


def _bio_labels(
    examples: List[Dict],
    label_key: str = "semantic_type",
    semantic_group_map: Optional[Dict[str, str]] = None,
    collapse_to_entity: bool = False,
) -> List[str]:
    if collapse_to_entity:
        return list(ENTITY_ONLY_LABEL_LIST)
    labels = set()
    for ex in examples:
        for e in ex.get("entities", []):
            raw = e.get(label_key) or e.get("type") or ""
            tag = _resolve_tag(raw, semantic_group_map)
            if tag:
                labels.add(f"B-{tag}")
                labels.add(f"I-{tag}")
    labels.discard("B-O")
    labels.discard("I-O")
    return sorted(labels) + ["O"]


_IGNORE_LABEL = "__IGNORE__"


def _align_to_tokens(
    text: str,
    entities: List[Dict],
    tokenizer,
    label_key: str = "semantic_type",
    semantic_group_map: Optional[Dict[str, str]] = None,
    collapse_to_entity: bool = False,
):
    enc = tokenizer(text, return_offsets_mapping=True, truncation=True, max_length=512)
    offsets = enc["offset_mapping"]
    labels = ["O"] * len(offsets)
    for ent in entities:
        ent_start, ent_end = int(ent["start"]), int(ent["end"])
        if collapse_to_entity:
            tag = "ENTITY"
        else:
            raw = ent.get(label_key) or ent.get("type") or ""
            tag = _resolve_tag(raw, semantic_group_map)
        if not tag:
            continue
        first = True
        for i, (tok_start, tok_end) in enumerate(offsets):
            if tok_start == 0 and tok_end == 0:
                continue
            if tok_end <= ent_start or tok_start >= ent_end:
                continue
            labels[i] = f"B-{tag}" if first else f"I-{tag}"
            first = False
    for i, (tok_start, tok_end) in enumerate(offsets):
        if tok_start == 0 and tok_end == 0:
            labels[i] = _IGNORE_LABEL
    return enc["input_ids"], enc["attention_mask"], labels


class _NERDataset(Dataset):
    def __init__(
        self,
        examples,
        tokenizer,
        label2id,
        label_key="semantic_type",
        semantic_group_map: Optional[Dict[str, str]] = None,
        collapse_to_entity: bool = False,
    ):
        self.data = []
        for ex in examples:
            ids, mask, labels = _align_to_tokens(
                ex["text"],
                ex.get("entities", []),
                tokenizer,
                label_key,
                semantic_group_map=semantic_group_map,
                collapse_to_entity=collapse_to_entity,
            )
            self.data.append({
                "input_ids": ids,
                "attention_mask": mask,
                "labels": [-100 if l == _IGNORE_LABEL else label2id.get(l, label2id["O"]) for l in labels],
            })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]


def debug_alignment(
    example: Dict,
    tokenizer,
    label2id: Dict,
    id2label: Dict,
    label_key: str = "semantic_type",
    max_tokens: int = 50,
    semantic_group_map: Optional[Dict[str, str]] = None,
    collapse_to_entity: bool = False,
):
    """Print label2id, id2label, and aligned labels for one example (for debugging zero F1 / span alignment)."""
    print("[DEBUG] label2id:", label2id)
    print("[DEBUG] id2label:", id2label)
    print("[DEBUG] example keys:", list(example.keys()))
    print("[DEBUG] example text length:", len(example.get("text", "")))
    print("[DEBUG] example entities count:", len(example.get("entities", [])))
    ids, mask, str_labels = _align_to_tokens(
        example["text"],
        example.get("entities", []),
        tokenizer,
        label_key,
        semantic_group_map=semantic_group_map,
        collapse_to_entity=collapse_to_entity,
    )
    label_ids = [-100 if l == _IGNORE_LABEL else label2id.get(l, label2id["O"]) for l in str_labels]
    # Gold label sequence: if all O (except ignore), span alignment or sentence split is wrong.
    gold_non_ignore = [l for l in str_labels if l != _IGNORE_LABEL]
    gold_non_o = [l for l in gold_non_ignore if l != "O"]
    print("[DEBUG] gold_labels (first %d, exclude special tokens):" % max_tokens, gold_non_ignore[:max_tokens])
    print("[DEBUG] gold non-O count:", len(gold_non_o), "/", len(gold_non_ignore), "tokens")
    if example.get("entities") and len(gold_non_o) == 0:
        print("[DEBUG] >>> SPAN ALIGNMENT BUG: example has entities but gold labels are all O. Check entity start/end vs token offsets.")
    print("[DEBUG] aligned label_ids (first %d):" % max_tokens, label_ids[:max_tokens])
    non_o = sum(1 for i in label_ids if i != -100 and id2label.get(i) != "O")
    print("[DEBUG] first example: non-O (non-ignore) label count:", non_o)


class _LossLoggingCallback(TrainerCallback):
    """Print train_loss and eval_loss so you can see if the model is learning (loss decreasing)."""

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        if "loss" in logs:
            print(f"  [loss] train_loss = {logs['loss']:.4f} (step {state.global_step})")
        if "eval_loss" in logs:
            print(f"  [loss] eval_loss = {logs['eval_loss']:.4f}")


def _compute_label_weights(dataset: _NERDataset, num_labels: int) -> Optional[torch.Tensor]:
    """Compute inverse-frequency weights from dataset (exclude -100). Returns tensor of shape (num_labels,) or None."""
    counts = np.zeros(num_labels, dtype=np.float64)
    for i in range(len(dataset)):
        for lid in dataset.data[i]["labels"]:
            if lid != -100 and 0 <= lid < num_labels:
                counts[lid] += 1.0
    if counts.sum() == 0:
        return None
    # Inverse frequency, clip and normalize so min weight 0.1, max 10, then scale so mean ~1
    inv = np.zeros_like(counts)
    np.divide(1.0, counts, out=inv, where=counts > 0)
    inv[counts == 0] = 0.0
    inv = np.clip(inv, 0.1, 10.0)
    if inv.max() > 0:
        inv = inv / (inv.sum() / max(1, (inv > 0).sum()))
    return torch.tensor(inv, dtype=torch.float32)


class _WeightedNERTrainer(Trainer):
    def __init__(self, **kwargs):
        self._label_weights = kwargs.pop("label_weights", None)
        super().__init__(**kwargs)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        num_labels = getattr(model, "num_labels", getattr(model.config, "num_labels", logits.shape[-1]))
        # CRF with emissions: add auxiliary weighted CE on emissions to boost rare labels (F1).
        emissions = getattr(outputs, "emissions", None)
        if self._label_weights is not None and emissions is not None and outputs.loss is not None:
            w = self._label_weights.to(emissions.device)
            loss_fct = torch.nn.CrossEntropyLoss(weight=w, ignore_index=-100)
            aux = loss_fct(emissions.view(-1, num_labels), labels.view(-1))
            loss = outputs.loss + 0.2 * aux
        elif self._label_weights is not None:
            loss_fct = torch.nn.CrossEntropyLoss(weight=self._label_weights.to(logits.device), ignore_index=-100)
            loss = loss_fct(logits.view(-1, num_labels), labels.view(-1))
        else:
            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(logits.view(-1, num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


# Default backbone: Qwen 2.5 (replace BioBERT)
DEFAULT_NER_MODEL = "Qwen/Qwen2.5-0.5B"


def _is_qwen_backbone(model_name: str) -> bool:
    """True if model is Qwen2/Qwen2.5 (no built-in AutoModelForTokenClassification)."""
    n = (model_name or "").lower()
    return "qwen" in n


def _hub_kwargs(model_name: str) -> dict:
    """Kwargs for from_pretrained when loading from hub (e.g. trust_remote_code for Qwen2)."""
    return {"trust_remote_code": True} if _is_qwen_backbone(model_name) else {}


def _load_config_and_backbone_class(model_name: str, hub_kw: dict):
    """Load config and backbone class. Use BertConfig/BertModel for BERT/BioBERT (Hub config may lack model_type)."""
    name_lower = (model_name or "").lower()
    if "bert" in name_lower or "biobert" in name_lower:
        config = BertConfig.from_pretrained(model_name)
        backbone = BertModel
    else:
        config = AutoConfig.from_pretrained(model_name, **hub_kw)
        backbone = AutoModel
    return config, backbone


def _tokenizer_kwargs(model_name: str) -> dict:
    """Kwargs for tokenizer from_pretrained. use_fast=False avoids needing sentencepiece/tiktoken for BERT."""
    out = dict(_hub_kwargs(model_name))
    if not _is_qwen_backbone(model_name):
        out["use_fast"] = False
    return out


def _remap_layernorm_gamma_beta_to_weight_bias(state_dict: dict) -> dict:
    """Remap LayerNorm.gamma -> weight, LayerNorm.beta -> bias (BioBERT/some checkpoints use gamma/beta)."""
    out = {}
    for k, v in state_dict.items():
        if k.endswith(".LayerNorm.gamma"):
            out[k.replace(".LayerNorm.gamma", ".LayerNorm.weight")] = v
        elif k.endswith(".LayerNorm.beta"):
            out[k.replace(".LayerNorm.beta", ".LayerNorm.bias")] = v
        else:
            out[k] = v
    return out


def _load_tokenizer(model_name: str, checkpoint_dir: str = None):
    """Load tokenizer; use BertTokenizer (slow) for BERT/BioBERT to avoid sentencepiece/tiktoken."""
    path = checkpoint_dir or model_name
    if _is_qwen_backbone(model_name or path):
        return AutoTokenizer.from_pretrained(path, **_hub_kwargs(model_name or path))
    return BertTokenizer.from_pretrained(path, use_fast=False)


class _CRFOutput:
    """Output container with .loss, .logits, optional .emissions; subscriptable so Trainer gets output[0]=loss, output[1]=logits."""
    __slots__ = ("loss", "logits", "emissions")

    def __init__(self, loss=None, logits=None, emissions=None):
        self.loss = loss
        self.logits = logits
        self.emissions = emissions

    def __getitem__(self, idx):
        if idx == 0:
            return self.loss
        if idx == 1:
            return self.logits
        if isinstance(idx, slice):
            return (self.loss, self.logits)[idx]
        raise IndexError(idx)


class BackboneForTokenClassification(torch.nn.Module):
    """Any encoder (Qwen 2.5, BERT, etc.) + linear head for token classification. Used when AutoModelForTokenClassification is not available (e.g. Qwen2)."""

    def __init__(self, model_name: str, num_labels: int, freeze_encoder: bool = False):
        super().__init__()
        self.model_name = model_name
        hub_kw = _hub_kwargs(model_name)
        config, BackboneClass = _load_config_and_backbone_class(model_name, hub_kw)
        config.num_labels = num_labels
        self.config = config
        self.num_labels = num_labels
        hidden_dropout = getattr(config, "hidden_dropout_prob", getattr(config, "emb_dropout_prob", 0.1))
        self.backbone = BackboneClass.from_pretrained(model_name, **hub_kw)
        self.dropout = torch.nn.Dropout(hidden_dropout)
        self.classifier = torch.nn.Linear(config.hidden_size, num_labels)
        if freeze_encoder:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        seq = outputs.last_hidden_state
        if seq.dtype != self.classifier.weight.dtype:
            seq = seq.to(self.classifier.weight.dtype)
        logits = self.classifier(self.dropout(seq))
        loss = None
        if labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        return _CRFOutput(loss=loss, logits=logits)

    def save_pretrained(self, save_directory: str):
        import json
        from pathlib import Path
        p = Path(save_directory)
        p.mkdir(parents=True, exist_ok=True)
        config = {
            "use_backbone_ner": True,
            "num_labels": self.num_labels,
            "model_name": self.model_name,
            "label_list": getattr(self, "label_list", None),
        }
        (p / "ner_config.json").write_text(json.dumps(config))
        torch.save(self.state_dict(), p / "pytorch_model.bin")

    @classmethod
    def from_pretrained(cls, save_directory: str, freeze_encoder: bool = False):
        import json
        from pathlib import Path
        p = Path(save_directory)
        config = json.loads((p / "ner_config.json").read_text())
        model = cls(config["model_name"], config["num_labels"], freeze_encoder=freeze_encoder)
        state = torch.load(p / "pytorch_model.bin", map_location="cpu")
        state = _remap_layernorm_gamma_beta_to_weight_bias(state)
        model.load_state_dict(state, strict=True)
        return model


class BERTCRFForTokenClassification(torch.nn.Module):
    """Backbone + linear emissions + CRF for BIO (works with Qwen 2.5 or BERT). Use with use_crf=True."""

    def __init__(self, model_name: str, num_labels: int, freeze_encoder: bool = False):
        super().__init__()
        if not _CRF_AVAILABLE:
            raise RuntimeError("pytorch-crf is required for use_crf. Install with: pip install pytorch-crf")
        self.model_name = model_name
        hub_kw = _hub_kwargs(model_name)
        config, BackboneClass = _load_config_and_backbone_class(model_name, hub_kw)
        self.config = config
        self.num_labels = num_labels
        self.bert = BackboneClass.from_pretrained(model_name, **hub_kw)
        self.dropout = torch.nn.Dropout(getattr(config, "hidden_dropout_prob", 0.1))
        self.classifier = torch.nn.Linear(config.hidden_size, num_labels)
        self.crf = TorchCRF(num_labels, batch_first=True)
        if freeze_encoder:
            for p in self.bert.parameters():
                p.requires_grad = False

    def state_dict(self, *args, **kwargs):
        # Safetensors / some save paths require contiguous tensors (e.g. BERT attention layers).
        out = super().state_dict(*args, **kwargs)
        return {k: v.contiguous() if isinstance(v, torch.Tensor) and not v.is_contiguous() else v for k, v in out.items()}

    def save_pretrained(self, save_directory: str):
        import json
        from pathlib import Path
        p = Path(save_directory)
        p.mkdir(parents=True, exist_ok=True)
        config = {
            "use_crf": True,
            "num_labels": self.num_labels,
            "model_name": self.model_name,
            "label_list": getattr(self, "label_list", None),
        }
        (p / "ner_config.json").write_text(json.dumps(config))
        torch.save(self.state_dict(), p / "pytorch_model.bin")

    @classmethod
    def from_pretrained(cls, save_directory: str, freeze_encoder: bool = False):
        import json
        from pathlib import Path
        p = Path(save_directory)
        config = json.loads((p / "ner_config.json").read_text())
        model = cls(config["model_name"], config["num_labels"], freeze_encoder=freeze_encoder)
        state = torch.load(p / "pytorch_model.bin", map_location="cpu")
        state = _remap_layernorm_gamma_beta_to_weight_bias(state)
        model.load_state_dict(state, strict=True)
        return model

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        seq = outputs.last_hidden_state  # (batch, seq_len, hidden_size)
        if seq.dtype != self.classifier.weight.dtype:
            seq = seq.to(self.classifier.weight.dtype)
        emissions = self.classifier(self.dropout(seq))  # (batch, seq_len, num_labels)

        # CRF expects mask: 1 = valid token, 0 = pad. First timestep must be valid.
        # Use boolean mask (PyTorch / torchcrf require bool for torch.where).
        if labels is not None:
            mask = (labels != -100).to(torch.bool)
            mask[:, 0] = True  # pytorch-crf requires first timestep valid
            tags = labels.clone()
            tags[labels == -100] = 0
            nll = -self.crf(emissions, tags, mask=mask, reduction="mean")
        else:
            nll = None

        # Decode for logits so Trainer / _compute_metrics get tag ids via argmax
        if attention_mask is not None:
            infer_mask = attention_mask.to(torch.bool)
        else:
            infer_mask = emissions.new_ones(emissions.shape[:2], dtype=torch.bool)
        decoded = self.crf.decode(emissions, mask=infer_mask)  # List[List[int]]
        # Pad to max length and convert to one-hot logits (batch, seq_len, num_labels)
        max_len = emissions.size(1)
        batch_size = emissions.size(0)
        logits = emissions.new_zeros(batch_size, max_len, self.num_labels)
        for b, seq_tags in enumerate(decoded):
            for i, tid in enumerate(seq_tags):
                if i < max_len:
                    logits[b, i, tid] = 1.0

        return _CRFOutput(loss=nll, logits=logits, emissions=emissions)


def _compute_metrics(eval_pred, id2label):
    preds, labels = eval_pred
    preds = np.argmax(preds, axis=-1)
    true_list, pred_list = [], []
    total_correct, total_tokens = 0, 0
    pred_labels_flat = []
    for p_seq, l_seq in zip(preds, labels):
        t, pr = [], []
        for p, l in zip(p_seq, l_seq):
            if l == -100:
                continue
            total_tokens += 1
            if p == l:
                total_correct += 1
            lab = id2label.get(p, "O")
            pred_labels_flat.append(lab)
            t.append(id2label.get(l, "O"))
            pr.append(lab)
        true_list.append(t)
        pred_list.append(pr)
    # Confirm collapse: if all O, model collapsed to majority class
    print("Predicted label distribution:", dict(Counter(pred_labels_flat)))
    accuracy = total_correct / total_tokens if total_tokens else 0.0
    return {
        "accuracy": accuracy,
        "f1": f1_score(true_list, pred_list),
        "precision": precision_score(true_list, pred_list),
        "recall": recall_score(true_list, pred_list),
    }


class BERTNERModel:
    def __init__(
        self,
        model_name: str = DEFAULT_NER_MODEL,
        label_list: Optional[List[str]] = None,
        label_key: str = "semantic_type",
        freeze_encoder: bool = False,
        use_semantic_groups: bool = False,
        class_weight: bool = False,
        collapse_to_entity: bool = False,
        use_crf: bool = False,
    ):
        self.model_name = model_name
        self.label_key = label_key
        self.tokenizer = _load_tokenizer(model_name)
        self.label_list = label_list or ["O"]
        self.label2id = {l: i for i, l in enumerate(self.label_list)}
        self.id2label = {i: l for l, i in self.label2id.items()}
        self.model = None
        self.freeze_encoder = freeze_encoder
        self.use_semantic_groups = use_semantic_groups
        self.class_weight = class_weight
        self.collapse_to_entity = collapse_to_entity
        self.use_crf = use_crf and _CRF_AVAILABLE
        self.semantic_group_map: Optional[Dict[str, str]] = UMLS_T_TO_GROUP if use_semantic_groups else None

    def prepare_from_data(self, train_examples: List[Dict], dev_examples: Optional[List[Dict]] = None):
        self.label_list = _bio_labels(
            train_examples + (dev_examples or []),
            self.label_key,
            semantic_group_map=self.semantic_group_map,
            collapse_to_entity=self.collapse_to_entity,
        )
        self.label2id = {l: i for i, l in enumerate(self.label_list)}
        self.id2label = {i: l for l, i in self.label2id.items()}
        num_labels = len(self.label_list)
        if self.collapse_to_entity and self.label_list == ENTITY_ONLY_LABEL_LIST:
            print(f"[NER] 3-label mode: B-ENTITY, I-ENTITY, O (num_labels={num_labels}) — sufficient for graph/state/drift.")
        if self.use_crf:
            print("[NER] Using CRF layer on top of backbone (BIO boundary enforcement).")
            self.model = BERTCRFForTokenClassification(
                self.model_name, num_labels, freeze_encoder=self.freeze_encoder
            )
            self.model.label_list = list(self.label_list)
        elif _is_qwen_backbone(self.model_name):
            print("[NER] Using Qwen 2.5 backbone + token classification head.")
            self.model = BackboneForTokenClassification(
                self.model_name, num_labels, freeze_encoder=self.freeze_encoder
            )
            self.model.label_list = list(self.label_list)
        else:
            self.model = BertForTokenClassification.from_pretrained(
                self.model_name, num_labels=num_labels
            )
            if self.freeze_encoder:
                for p in self.model.bert.parameters():
                    p.requires_grad = False
        return self

    def train(
        self,
        train_examples: List[Dict],
        dev_examples: Optional[List[Dict]] = None,
        output_dir: str = "./out_ner",
        num_epochs: int = 3,
        batch_size: int = 16,
        lr: float = 2e-5,
        warmup_ratio: float = 0.1,
        use_amp: bool = True,
        dataloader_num_workers: int = 4,
        entity_weight: Optional[float] = None,
    ):
        if self.model is None:
            self.prepare_from_data(train_examples, dev_examples)
        train_ds = _NERDataset(
            train_examples,
            self.tokenizer,
            self.label2id,
            self.label_key,
            semantic_group_map=self.semantic_group_map,
            collapse_to_entity=self.collapse_to_entity,
        )
        eval_ds = (
            _NERDataset(
                dev_examples,
                self.tokenizer,
                self.label2id,
                self.label_key,
                semantic_group_map=self.semantic_group_map,
                collapse_to_entity=self.collapse_to_entity,
            )
            if dev_examples
            else None
        )
        collator = DataCollatorForTokenClassification(self.tokenizer, padding=True)
        fn = lambda pred: _compute_metrics(pred, self.id2label)
        # Mixed precision: bf16 on Ampere+, else fp16 (faster training). Disable with use_amp=False if NaNs.
        bf16 = use_amp and getattr(torch.cuda, "is_bf16_supported", lambda: False)()
        fp16 = use_amp and not bf16
        if use_amp:
            print("[NER] Mixed precision:", "bf16" if bf16 else "fp16")
        args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=lr,
            warmup_ratio=warmup_ratio,
            fp16=fp16,
            bf16=bf16,
            dataloader_num_workers=max(0, dataloader_num_workers),
            dataloader_pin_memory=(dataloader_num_workers > 0),
            **{_EVAL_STRATEGY_KEY: "epoch" if eval_ds else "no"},
            save_strategy="epoch",
            load_best_model_at_end=bool(eval_ds),
            metric_for_best_model="f1",
        )
        label_weights = None
        if entity_weight is not None and self.label_list == ENTITY_ONLY_LABEL_LIST and not self.use_crf:
            # Manual weights: B-ENTITY and I-ENTITY get entity_weight, O gets 1.0 (3-label only, no CRF).
            label_weights = torch.tensor(
                [float(entity_weight), float(entity_weight), 1.0],
                dtype=torch.float32,
            )
            print(f"[NER] Using manual entity weights (B=I={entity_weight}, O=1.0).")
        elif self.class_weight:
            label_weights = _compute_label_weights(train_ds, len(self.label_list))
            if label_weights is not None:
                if self.use_crf:
                    print("[NER] Using class-weighted auxiliary CE on emissions (5-label CRF F1 boost).")
                else:
                    print("[NER] Using class-weighted loss (inverse frequency).")
        trainer_cls = _WeightedNERTrainer if label_weights is not None else Trainer
        trainer_kw = dict(
            model=self.model,
            args=args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            data_collator=collator,
            compute_metrics=fn,
            callbacks=[_LossLoggingCallback()],
        )
        if label_weights is not None:
            trainer_kw["label_weights"] = label_weights
        trainer = trainer_cls(**trainer_kw)
        trainer.train()
        # Save in our format so load() finds ner_config.json (CRF/backbone). Trainer.save_model()
        # writes HF format only and no ner_config.json, so load() would fall back to 3-label BERT and fail.
        if isinstance(self.model, (BERTCRFForTokenClassification, BackboneForTokenClassification)):
            self.model.save_pretrained(output_dir)
        else:
            trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        return trainer

    def load(self, checkpoint_dir: str):
        from pathlib import Path
        p = Path(checkpoint_dir)
        if (p / "ner_config.json").exists():
            import json
            cfg = json.loads((p / "ner_config.json").read_text())
            if cfg.get("from_hub"):
                return self.load_pretrained(cfg["from_hub"])
            if cfg.get("use_crf"):
                self.model = BERTCRFForTokenClassification.from_pretrained(
                    checkpoint_dir, freeze_encoder=self.freeze_encoder
                )
                self.use_crf = True
                self.label_list = list(cfg.get("label_list") or ["B-ENTITY", "I-ENTITY", "O"])
                self.label2id = {l: i for i, l in enumerate(self.label_list)}
                self.id2label = {i: l for l, i in self.label2id.items()}
                self.tokenizer = _load_tokenizer(cfg.get("model_name", ""), checkpoint_dir)
                return self
            if cfg.get("use_backbone_ner"):
                self.model = BackboneForTokenClassification.from_pretrained(
                    checkpoint_dir, freeze_encoder=self.freeze_encoder
                )
                self.label_list = list(cfg.get("label_list") or ["B-ENTITY", "I-ENTITY", "O"])
                self.label2id = {l: i for i, l in enumerate(self.label_list)}
                self.id2label = {i: l for l, i in self.label2id.items()}
                self.tokenizer = _load_tokenizer(cfg.get("model_name", ""), checkpoint_dir)
                return self
        self.model = BertForTokenClassification.from_pretrained(checkpoint_dir)
        self.tokenizer = _load_tokenizer("bert", checkpoint_dir)
        self.label_list = list(self.model.config.id2label.values())
        self.label2id = self.model.config.label2id
        self.id2label = self.model.config.id2label
        return self

    def load_pretrained(self, hub_model_id: str):
        """Load a pretrained NER model from HuggingFace Hub (no training). Use for Baseline 1/2 with ready-made NER."""
        from transformers import AutoTokenizer, AutoConfig
        import json

        def _lora_base_then_adapter():
            """Load base as BertForTokenClassification then apply PEFT adapter (correct for Francesco-A–style LoRA)."""
            from peft import PeftModel
            from huggingface_hub import hf_hub_download
            adapter_path = hf_hub_download(hub_model_id, "adapter_config.json")
            with open(adapter_path) as f:
                adapter_cfg = json.load(f)
            base_name = adapter_cfg.get("base_model_name_or_path") or hub_model_id
            config_path = hf_hub_download(hub_model_id, "config.json")
            with open(config_path) as f:
                hub_config = json.load(f)
            id2label = hub_config.get("id2label")
            if id2label is not None:
                id2label = {int(k): str(v) for k, v in id2label.items()}
            else:
                id2label = {0: "O", 1: "B-Chemical", 2: "I-Chemical", 3: "B-Disease", 4: "I-Disease"}
            num_labels = len(id2label)
            config = AutoConfig.from_pretrained(base_name)
            config.num_labels = num_labels
            config.id2label = id2label
            config.label2id = {v: k for k, v in id2label.items()}
            base_model = BertForTokenClassification.from_pretrained(base_name, config=config)
            self.model = PeftModel.from_pretrained(base_model, hub_model_id)
            self.model = self.model.merge_and_unload()
            self.tokenizer = AutoTokenizer.from_pretrained(hub_model_id)
            self.id2label = id2label
            self.label2id = {v: k for k, v in self.id2label.items()}
            self.label_list = list(self.id2label.values())
            return True

        try:
            if _lora_base_then_adapter():
                print("[NER] Loaded LoRA adapter (base + PEFT merge) from", hub_model_id)
                return self
        except Exception as e:
            print("[NER] LoRA base+adapter load failed:", e, "— falling back to AutoPeft/Auto.")
        try:
            from peft import AutoPeftModelForTokenClassification
            self.model = AutoPeftModelForTokenClassification.from_pretrained(hub_model_id)
            self.model = self.model.merge_and_unload()
            self.tokenizer = AutoTokenizer.from_pretrained(hub_model_id)
        except (ImportError, Exception):
            try:
                self.model = AutoModelForTokenClassification.from_pretrained(hub_model_id)
                self.tokenizer = AutoTokenizer.from_pretrained(hub_model_id)
            except Exception as e2:
                err = str(e2).lower()
                if (
                    "does not appear to have" in err
                    or "pytorch_model.bin" in err
                    or "model.safetensors" in err
                ):
                    raise RuntimeError(
                        f"NER hub id {hub_model_id!r} is a LoRA adapter repo (no full model files at root). "
                        "Install dependencies and retry: pip install peft huggingface_hub "
                        "(see baselines/requirements.txt). "
                        f"Original error: {e2}"
                    ) from e2
                raise RuntimeError(
                    f"Failed to load {hub_model_id}. For LoRA adapters: pip install peft huggingface_hub. "
                    f"Error: {e2}"
                ) from e2
        id2label = getattr(self.model.config, "id2label", None)
        if id2label is not None and isinstance(id2label, dict) and len(id2label) > 0:
            self.id2label = {int(k): v for k, v in id2label.items()}
        else:
            num_labels = getattr(self.model.config, "num_labels", 5)
            if num_labels == 5:
                self.id2label = {0: "O", 1: "B-Chemical", 2: "I-Chemical", 3: "B-Disease", 4: "I-Disease"}
            else:
                self.id2label = {i: f"LABEL_{i}" for i in range(num_labels)}
        self.label2id = {v: k for k, v in self.id2label.items()}
        self.label_list = list(self.id2label.values())
        return self

    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """
        Return list of {"text", "start", "end", "label", "confidence"}.
        """
        if self.model is None:
            raise RuntimeError("Model not loaded; train or load first.")
        self.model.eval()
        enc = self.tokenizer(
            text,
            return_offsets_mapping=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        device = next(self.model.parameters()).device
        with torch.no_grad():
            if isinstance(self.model, (BERTCRFForTokenClassification, BackboneForTokenClassification)):
                out = self.model(
                    input_ids=enc["input_ids"].to(device),
                    attention_mask=enc["attention_mask"].to(device),
                )
                logits = out.logits
                if isinstance(self.model, BERTCRFForTokenClassification):
                    preds = logits.argmax(-1)[0].tolist()
                    prob_list = logits[0].max(-1).values.tolist()
                else:
                    probs = torch.softmax(logits, dim=-1)
                    preds = logits.argmax(-1)[0].tolist()
                    prob_list = probs[0].max(-1).values.tolist()
            else:
                logits = self.model(
                    input_ids=enc["input_ids"].to(device),
                    attention_mask=enc["attention_mask"].to(device),
                ).logits
                probs = torch.softmax(logits, dim=-1)
                preds = logits.argmax(-1)[0].tolist()
                prob_list = probs[0].max(-1).values.tolist()
        offsets = enc["offset_mapping"][0].tolist()
        out = []
        cur = None
        for i, (pid, (s, e)) in enumerate(zip(preds, offsets)):
            if s == 0 and e == 0:
                continue
            lab = self.id2label.get(pid, "O")
            conf = prob_list[i]
            if lab.startswith("B-"):
                if cur:
                    out.append(cur)
                cur = {"start": s, "end": e, "label": lab[2:], "confidence": conf, "text": text[s:e]}
            elif lab.startswith("I-") and cur and lab[2:] == cur["label"]:
                cur["end"] = e
                cur["text"] = text[cur["start"]:e]
                cur["confidence"] = max(cur["confidence"], conf)
            else:
                if cur:
                    out.append(cur)
                cur = None
        if cur:
            out.append(cur)
        return out
