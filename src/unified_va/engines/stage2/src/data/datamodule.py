"""Torch Dataset, collate, and tokenizer caching for DimABSA2026."""
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler, Sampler, WeightedRandomSampler

from src.data.sample_keys import candidate_sample_keys
from src.data.translation_sidecar import lookup_translation_record


DEFAULT_SMALL_LLM_PROMPT = (
    "You are a multilingual valence-arousal regression backbone.\n"
    "Read the text and the target aspect, then produce a hidden representation suitable for continuous affect prediction.\n"
    "Aspect: {aspect}\n"
    "Text: {text}"
)


def _render_prompt_template(sample: dict, prompt_template: str) -> str:
    opinion = sample.get("opinion") or ""
    category = sample.get("category") or ""
    track = sample.get("track") or ""
    lang = sample.get("lang") or ""
    domain = sample.get("domain") or ""
    return prompt_template.format(
        id=sample.get("id", ""),
        aspect=sample["aspect"],
        text=sample["text"],
        aspect_en=sample.get("aspect_en", sample.get("aspect_pivot", sample["aspect"])),
        text_en=sample.get("text_en", sample.get("text_pivot", sample["text"])),
        aspect_pivot=sample.get("aspect_pivot", sample.get("aspect_en", sample["aspect"])),
        text_pivot=sample.get("text_pivot", sample.get("text_en", sample["text"])),
        opinion=opinion,
        category=category,
        track=track,
        lang=lang,
        domain=domain,
        source_lang=sample.get("source_lang", lang),
        pivot_lang=sample.get("pivot_lang", sample.get("target_lang", "eng")),
        mt_conf=sample.get("mt_conf", 1.0),
    )


def render_dimasr_input(sample: dict, input_format: str = "aspect_sep_text",
                        prompt_template: str | None = None) -> str:
    """Render aspect-conditioned model input text."""
    opinion = sample.get("opinion")
    if prompt_template:
        return _render_prompt_template(sample, prompt_template)
    if input_format == "instruction_va_v1":
        return _render_prompt_template(sample, DEFAULT_SMALL_LLM_PROMPT)
    if input_format == "aspect_colon_text":
        return f"{sample['aspect']} : {sample['text']}"
    if input_format == "tagged_aspect":
        return f"The aspect is {sample['aspect']}. {sample['text']}"
    if input_format == "text_then_aspect":
        return f"{sample['text']} [SEP] {sample['aspect']}"
    if input_format == "asp_opi_sep_text" and opinion:
        return f"{sample['aspect']} [SEP] {opinion} [SEP] {sample['text']}"
    return f"{sample['aspect']} [SEP] {sample['text']}"


class DimASRDataset(Dataset):
    """Dataset for DimABSA2026 aspect-level VA regression."""

    def __init__(self, samples: list[dict], tokenizer, max_len: int = 128,
                 input_format: str = "aspect_sep_text",
                 proto_sidecar: dict | None = None,
                 fn_sidecar: dict | None = None,
                 teacher_map: dict | None = None,
                 llm_sidecar=None,
                 translation_map: dict | None = None,
                 translation_pivot: dict | None = None,
                 prompt_template: str | None = None,
                 teacher_explanation_as_input: bool = False,
                 teacher_explanation_prefix: str = "teacher_explanation",
                 teacher_explanation_min_quality: float = 0.0):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.input_format = input_format
        self.prompt_template = prompt_template
        self._cache = {}
        self.proto_sidecar = proto_sidecar
        self.proto_k = 0
        if proto_sidecar:
            first = next(iter(proto_sidecar.values()))
            self.proto_k = len(first["proto_q"])
        self.fn_sidecar = fn_sidecar
        self.fn_join_miss_count = 0
        self.teacher_map = teacher_map
        self.llm_sidecar = llm_sidecar
        self.llm_dim = getattr(llm_sidecar, "dim", 0) if llm_sidecar is not None else 0
        self.llm_join_miss_count = 0
        self.translation_map = translation_map
        self.translation_pivot = translation_pivot or {}
        self.translation_enabled = bool(self.translation_pivot.get("enabled", False)) and translation_map is not None
        self.translation_join_miss_count = 0
        self.teacher_explanation_as_input = teacher_explanation_as_input
        self.teacher_explanation_prefix = teacher_explanation_prefix
        self.teacher_explanation_min_quality = teacher_explanation_min_quality

    def __len__(self):
        return len(self.samples)

    def _make_input(self, sample: dict, teacher_rec: dict | None = None) -> str:
        base = render_dimasr_input(sample, self.input_format, prompt_template=self.prompt_template)
        if self.teacher_explanation_as_input and teacher_rec:
            if teacher_rec.get("teacher_mask", 0):
                expl = teacher_rec.get("teacher_explanation")
                if isinstance(expl, str):
                    expl = expl.strip()
                    if expl:
                        eq = float(teacher_rec.get("teacher_expl_quality", 1.0))
                        if eq >= self.teacher_explanation_min_quality:
                            return f"{base} [SEP] {self.teacher_explanation_prefix}: {expl}"
        return base

    def _make_pivot_sample(self, sample: dict, translation_rec: dict | None) -> dict:
        pivot_lang = str(self.translation_pivot.get("pivot_lang", "eng") or "eng")
        pivot_sample = dict(sample)
        pivot_sample["source_lang"] = sample.get("lang", "unknown")
        pivot_sample["pivot_lang"] = pivot_lang

        if translation_rec is None:
            pivot_sample["lang"] = pivot_lang
            pivot_sample["aspect_en"] = sample["aspect"]
            pivot_sample["text_en"] = sample["text"]
            pivot_sample["aspect_pivot"] = sample["aspect"]
            pivot_sample["text_pivot"] = sample["text"]
            pivot_sample["mt_conf"] = 0.0
            return pivot_sample

        aspect_en = translation_rec.get("aspect_en") or sample["aspect"]
        text_en = translation_rec.get("text_en") or sample["text"]
        pivot_sample["lang"] = pivot_lang
        pivot_sample["aspect"] = aspect_en
        pivot_sample["text"] = text_en
        pivot_sample["aspect_en"] = aspect_en
        pivot_sample["text_en"] = text_en
        pivot_sample["aspect_pivot"] = aspect_en
        pivot_sample["text_pivot"] = text_en
        pivot_sample["mt_conf"] = float(translation_rec.get("mt_conf", 1.0))
        return pivot_sample

    def _make_pivot_input(self, sample: dict, translation_rec: dict | None) -> str:
        pivot_sample = self._make_pivot_sample(sample, translation_rec)
        pivot_input_format = self.translation_pivot.get("pivot_input_format", self.input_format)
        pivot_prompt_template = self.translation_pivot.get("pivot_prompt_template")
        return render_dimasr_input(
            pivot_sample,
            pivot_input_format,
            prompt_template=pivot_prompt_template,
        )

    def _tokenize(self, text: str):
        # This is a process-local tokenizer cache, not an integrity check.
        if text not in self._cache:
            if getattr(self.tokenizer, "pad_token", None) is None and getattr(self.tokenizer, "eos_token", None) is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            enc = self.tokenizer(
                text,
                truncation=True,
                max_length=self.max_len,
                padding="max_length",
                return_tensors="pt",
            )
            self._cache[text] = {k: v.squeeze(0) for k, v in enc.items()}
        return self._cache[text]

    def __getitem__(self, idx):
        sample = self.samples[idx]
        teacher_rec = None
        for sample_key in candidate_sample_keys(sample["id"], sample["aspect_index"], sample.get("lang")):
            if self.teacher_map is not None:
                teacher_rec = self.teacher_map.get(sample_key)
                if teacher_rec is not None:
                    break

        tok = self._tokenize(self._make_input(sample, teacher_rec=teacher_rec))
        item = {
            "input_ids": tok["input_ids"],
            "attention_mask": tok["attention_mask"],
            "idx": idx,
        }
        if sample.get("v") is not None and sample.get("a") is not None:
            item["va"] = torch.tensor([sample["v"], sample["a"]], dtype=torch.float)
        if sample.get("is_synthetic"):
            item["is_synthetic"] = torch.tensor(1.0, dtype=torch.float)
            item["synth_uncertainty"] = torch.tensor(
                [sample.get("synth_uncertainty_v", 1.0), sample.get("synth_uncertainty_a", 1.0)],
                dtype=torch.float,
            )
        else:
            item["is_synthetic"] = torch.tensor(0.0, dtype=torch.float)
            item["synth_uncertainty"] = torch.zeros(2, dtype=torch.float)
        item["is_auxiliary"] = torch.tensor(1.0 if sample.get("_is_auxiliary") else 0.0, dtype=torch.float)
        item["sample_weight"] = torch.tensor(float(sample.get("_sample_weight", 1.0)), dtype=torch.float)
        item["error_hardness"] = torch.tensor(float(sample.get("_error_hardness", 0.0)), dtype=torch.float)
        item["error_hard_mask"] = torch.tensor(float(sample.get("_error_hard_mask", 0.0)), dtype=torch.float)
        item["error_easy_mask"] = torch.tensor(float(sample.get("_error_easy_mask", 0.0)), dtype=torch.float)
        item["lang"] = sample.get("lang", "unknown")
        item["domain"] = sample.get("domain", "unknown")
        item["lang_metadata"] = sample.get("lang", "unknown")
        item["domain_metadata"] = sample.get("domain", "unknown")
        item["repair_mask"] = torch.tensor(float(sample.get("_repair_mask", 0.0)), dtype=torch.float)
        item["repair_weight"] = torch.tensor(
            float(sample.get("_repair_weight", sample.get("_repair_mask", 0.0))),
            dtype=torch.float,
        )
        item["repair_score"] = torch.tensor(float(sample.get("_repair_score", 0.0)), dtype=torch.float)
        item["protected_mask"] = torch.tensor(float(sample.get("_protected_mask", 0.0)), dtype=torch.float)
        item["protected_weight"] = torch.tensor(
            float(sample.get("_protected_weight", sample.get("_protected_mask", 0.0))),
            dtype=torch.float,
        )
        item["protected_va"] = torch.tensor(
            [
                float(sample.get("_protected_teacher_v", 0.0)),
                float(sample.get("_protected_teacher_a", 0.0)),
            ],
            dtype=torch.float,
        )

        if self.proto_sidecar is not None:
            rec = None
            for sample_key in candidate_sample_keys(sample["id"], sample["aspect_index"], sample.get("lang")):
                rec = self.proto_sidecar.get(sample_key)
                if rec is not None:
                    break
            if rec:
                item["proto_q"] = torch.tensor(rec["proto_q"], dtype=torch.float)
                item["proto_stab"] = torch.tensor(rec["proto_stab"], dtype=torch.float)
                item["proto_mask"] = torch.tensor(1.0, dtype=torch.float)
            else:
                item["proto_q"] = torch.zeros(self.proto_k, dtype=torch.float)
                item["proto_stab"] = torch.tensor(0.0, dtype=torch.float)
                item["proto_mask"] = torch.tensor(0.0, dtype=torch.float)

        if self.fn_sidecar is not None:
            fn_rec = None
            for sample_key in candidate_sample_keys(sample["id"], sample["aspect_index"], sample.get("lang")):
                fn_rec = self.fn_sidecar.get(sample_key)
                if fn_rec is not None:
                    break
            if fn_rec:
                item["fn_risk"] = torch.tensor(fn_rec["fn_risk"], dtype=torch.float)
                item["fn_weight"] = torch.tensor(fn_rec["fn_weight"], dtype=torch.float)
                item["fn_active"] = torch.tensor(float(fn_rec["fn_active"]), dtype=torch.float)
            else:
                self.fn_join_miss_count += 1
                item["fn_risk"] = torch.tensor(0.0, dtype=torch.float)
                item["fn_weight"] = torch.tensor(1.0, dtype=torch.float)
                item["fn_active"] = torch.tensor(0.0, dtype=torch.float)

        if self.teacher_map is not None:
            if teacher_rec and teacher_rec.get("teacher_mask", 0):
                item["teacher_va"] = torch.tensor([teacher_rec["teacher_v"], teacher_rec["teacher_a"]], dtype=torch.float)
                item["teacher_weight"] = torch.tensor(teacher_rec.get("teacher_weight", 1.0), dtype=torch.float)
                item["teacher_mask"] = torch.tensor(1.0, dtype=torch.float)
                item["teacher_expl_quality"] = torch.tensor(teacher_rec.get("teacher_expl_quality", 1.0), dtype=torch.float)
            else:
                item["teacher_va"] = torch.zeros(2, dtype=torch.float)
                item["teacher_weight"] = torch.tensor(0.0, dtype=torch.float)
                item["teacher_mask"] = torch.tensor(0.0, dtype=torch.float)
                item["teacher_expl_quality"] = torch.tensor(0.0, dtype=torch.float)

        if self.llm_sidecar is not None:
            llm_emb = None
            for sample_key in candidate_sample_keys(sample["id"], sample["aspect_index"], sample.get("lang")):
                llm_emb = self.llm_sidecar.get(sample_key)
                if llm_emb is not None:
                    break
            if llm_emb is not None:
                item["llm_emb"] = torch.from_numpy(np.asarray(llm_emb, dtype=np.float32))
                item["llm_mask"] = torch.tensor(1.0, dtype=torch.float)
            else:
                self.llm_join_miss_count += 1
                item["llm_emb"] = torch.zeros(self.llm_dim, dtype=torch.float)
                item["llm_mask"] = torch.tensor(0.0, dtype=torch.float)

        if self.translation_enabled:
            translation_rec, _ = lookup_translation_record(self.translation_map, sample)
            if translation_rec is None:
                self.translation_join_miss_count += 1
            tok_pivot = self._tokenize(self._make_pivot_input(sample, translation_rec))
            item["pivot_input_ids"] = tok_pivot["input_ids"]
            item["pivot_attention_mask"] = tok_pivot["attention_mask"]

            enable_for_english = bool(self.translation_pivot.get("enable_for_english", False))
            lang = (sample.get("lang") or "").strip()
            pivot_active = translation_rec is not None and (enable_for_english or lang != "eng")
            item["pivot_mask"] = torch.tensor(1.0 if pivot_active else 0.0, dtype=torch.float)

            if pivot_active:
                conf = float(translation_rec.get("mt_conf", self.translation_pivot.get("default_confidence", 1.0)))
                conf = max(conf, float(self.translation_pivot.get("confidence_min", 0.0)))
            else:
                conf = 0.0
            item["pivot_weight"] = torch.tensor(conf, dtype=torch.float)
        return item

    def update_fn_weights(self, weight_map: dict) -> dict:
        """Dynamically update FN weights (for online/hybrid FN risk)."""
        if self.fn_sidecar is None:
            self.fn_sidecar = {}

        dataset_keys = set()
        for sample in self.samples:
            for sample_key in candidate_sample_keys(sample["id"], sample["aspect_index"], sample.get("lang")):
                dataset_keys.add(sample_key)

        n_hit = 0
        for key, rec in weight_map.items():
            self.fn_sidecar[key] = rec
            if key in dataset_keys:
                n_hit += 1

        hit_rate = n_hit / max(len(dataset_keys), 1)
        if hit_rate < 0.95:
            import logging
            logging.getLogger(__name__).warning(
                f"update_fn_weights: low hit rate {hit_rate:.3f} ({n_hit}/{len(dataset_keys)} dataset samples matched)"
            )

        return {
            "n_map": len(weight_map),
            "n_dataset": len(dataset_keys),
            "n_hit": n_hit,
            "hit_rate": round(hit_rate, 4),
        }


def dimasr_collate(batch: list[dict], disable_explicit_metadata: bool = False) -> dict:
    out = {}
    for k in batch[0]:
        vals = [b[k] for b in batch]
        if isinstance(vals[0], torch.Tensor):
            out[k] = torch.stack(vals)
        else:
            out[k] = vals
    if disable_explicit_metadata:
        # Keep the fields present for compatibility, but scrub true metadata so
        # no downstream path can accidentally consume gold lang/domain labels.
        if "lang" in out:
            out["lang"] = ["__meta_disabled__"] * len(out["lang"])
        if "domain" in out:
            out["domain"] = ["__meta_disabled__"] * len(out["domain"])
    return out


class EpochSeededRandomSampler(Sampler):
    """Sample order depends only on the explicit sampler seed and epoch.

    Separate from global/model initialization RNG and DataLoader worker RNG.
    Calling set_epoch(e) reproduces that epoch without replaying prior epochs.
    """

    def __init__(self, data_source, seed: int):
        self.data_source = data_source
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int):
        if int(epoch) < 0:
            raise ValueError("sampler epoch must be nonnegative")
        self.epoch = int(epoch)

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        return iter(torch.randperm(len(self.data_source), generator=generator).tolist())

    def __len__(self):
        return len(self.data_source)


def build_dimasr_dataloader(samples, tokenizer, batch_size=8, shuffle=True,
                            max_len=128, input_format="aspect_sep_text",
                            proto_sidecar=None, fn_sidecar=None,
                            teacher_map=None, llm_sidecar=None,
                            translation_map: dict | None = None,
                            translation_pivot: dict | None = None,
                            prompt_template: str | None = None,
                            teacher_explanation_as_input: bool = False,
                            teacher_explanation_prefix: str = "teacher_explanation",
                            teacher_explanation_min_quality: float = 0.0,
                            sampler=None,
                            sampler_seed: int | None = None,
                            distributed_num_replicas: int | None = None,
                            distributed_rank: int | None = None,
                            disable_explicit_metadata: bool = False,
                            num_workers: int = 0,
                            pin_memory: bool = False,
                            persistent_workers: bool = False):
    ds = DimASRDataset(
        samples,
        tokenizer,
        max_len,
        input_format,
        proto_sidecar=proto_sidecar,
        fn_sidecar=fn_sidecar,
        teacher_map=teacher_map,
        llm_sidecar=llm_sidecar,
        translation_map=translation_map,
        translation_pivot=translation_pivot,
        prompt_template=prompt_template,
        teacher_explanation_as_input=teacher_explanation_as_input,
        teacher_explanation_prefix=teacher_explanation_prefix,
        teacher_explanation_min_quality=teacher_explanation_min_quality,
    )
    if sampler is None and distributed_num_replicas and distributed_num_replicas > 1:
        sampler = DistributedSampler(
            ds,
            num_replicas=int(distributed_num_replicas),
            rank=int(distributed_rank or 0),
            shuffle=bool(shuffle),
            drop_last=False,
            **({"seed": int(sampler_seed)} if sampler_seed is not None else {}),
        )
    elif sampler is None and shuffle and sampler_seed is not None:
        sampler = EpochSeededRandomSampler(ds, seed=int(sampler_seed))

    effective_num_workers = max(int(num_workers), 0)
    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": bool(shuffle) if sampler is None else False,
        "collate_fn": lambda batch: dimasr_collate(
            batch,
            disable_explicit_metadata=bool(disable_explicit_metadata),
        ),
        "drop_last": False,
        "sampler": sampler,
        "num_workers": effective_num_workers,
        "pin_memory": bool(pin_memory),
    }
    if sampler_seed is not None:
        worker_generator = torch.Generator()
        worker_generator.manual_seed(int(sampler_seed))
        loader_kwargs["generator"] = worker_generator
    if effective_num_workers > 0:
        loader_kwargs["persistent_workers"] = bool(persistent_workers)
    return DataLoader(ds, **loader_kwargs)
