""" ProLIP
Copyright (c) 2024-present NAVER Cloud Corp.
MIT license

Reference code: https://github.com/mlfoundations/open_clip/blob/v2.24.0/src/open_clip/zero_shot_classifier.py
"""
from functools import partial
from itertools import islice
from typing import Callable, Optional, Sequence, Union

import torch


def batched(iterable, n):
    """Batch data into lists of length *n*. The last batch may be shorter.
    NOTE based on more-itertools impl, to be replaced by python 3.12 itertools.batched impl
    """
    it = iter(iterable)
    while True:
        batch = list(islice(it, n))
        if not batch:
            break
        yield batch


def build_zero_shot_classifier(
        model,
        tokenizer,
        classnames: Sequence[str],
        templates: Sequence[Union[Callable, str]],
        num_classes_per_batch: Optional[int] = 10,
        device: Union[str, torch.device] = 'cpu',
        use_tqdm: bool = False,
):
    """ Build zero-shot classifier weights by iterating over class names in batches
    Args:
        model: CLIP model instance
        tokenizer: CLIP tokenizer instance
        classnames: A sequence of class (label) names
        templates: A sequence of callables or format() friendly strings to produce templates per class name
        num_classes_per_batch: The number of classes to batch together in each forward, all if None
        device: Device to use.
        use_tqdm: Enable TQDM progress bar.
    """
    assert isinstance(templates, Sequence) and len(templates) > 0
    assert isinstance(classnames, Sequence) and len(classnames) > 0
    use_format = isinstance(templates[0], str)
    num_templates = len(templates)
    num_classes = len(classnames)
    if use_tqdm:
        import tqdm
        num_iter = 1 if num_classes_per_batch is None else ((num_classes - 1) // num_classes_per_batch + 1)
        iter_wrap = partial(tqdm.tqdm, total=num_iter, unit_scale=num_classes_per_batch)
    else:
        iter_wrap = iter

    def _process_batch(batch_classnames):
        num_batch_classes = len(batch_classnames)
        texts = [template.format(c) if use_format else template(c) for c in batch_classnames for template in templates]
        texts = tokenizer(texts).to(device)
        class_embeddings = model.encode_text(texts, normalize=True)
        class_embeddings = class_embeddings["mean"]
        class_embeddings = class_embeddings.reshape(num_batch_classes, num_templates, -1).mean(dim=1)
        class_embeddings = class_embeddings / class_embeddings.norm(dim=1, keepdim=True)
        class_embeddings = class_embeddings.T
        return class_embeddings

    with torch.no_grad():
        if num_classes_per_batch:
            batched_embeds = [_process_batch(batch) for batch in iter_wrap(batched(classnames, num_classes_per_batch))]
            zeroshot_weights = torch.cat(batched_embeds, dim=1)
        else:
            zeroshot_weights = _process_batch(classnames)
    return zeroshot_weights


def build_zero_shot_classifier_with_stds(
        model,
        tokenizer,
        classnames: Sequence[str],
        templates: Sequence[Union[Callable, str]],
        num_classes_per_batch: Optional[int] = 10,
        device: Union[str, torch.device] = 'cpu',
        use_tqdm: bool = False,
        no_aggregate: bool = False,
):
    """ Build zero-shot classifier weights by iterating over class names in batches
    Args:
        model: CLIP model instance
        tokenizer: CLIP tokenizer instance
        classnames: A sequence of class (label) names
        templates: A sequence of callables or format() friendly strings to produce templates per class name
        num_classes_per_batch: The number of classes to batch together in each forward, all if None
        device: Device to use.
        use_tqdm: Enable TQDM progress bar.
    """
    assert isinstance(templates, Sequence) and len(templates) > 0
    assert isinstance(classnames, Sequence) and len(classnames) > 0
    use_format = isinstance(templates[0], str)
    num_templates = len(templates)
    num_classes = len(classnames)
    if use_tqdm:
        import tqdm
        num_iter = 1 if num_classes_per_batch is None else ((num_classes - 1) // num_classes_per_batch + 1)
        iter_wrap = partial(tqdm.tqdm, total=num_iter, unit_scale=num_classes_per_batch)
    else:
        iter_wrap = iter

    def _process_batch(batch_classnames):
        num_batch_classes = len(batch_classnames)
        texts = [template.format(c=c) if use_format else template(c) for c in batch_classnames for template in templates]
        texts = tokenizer(texts).to(device)
        class_embeddings = model.encode_text(texts, normalize=True)
        std_embeddings = class_embeddings["std"]
        if std_embeddings is not None:
            std_embeddings = std_embeddings.reshape(num_batch_classes, num_templates, -1).exp()
        else:
            std_embeddings = torch.zeros_like(class_embeddings["mean"]).reshape(num_batch_classes, num_templates, -1)
        if not no_aggregate:
            std_embeddings = std_embeddings.mean(dim=1)
            std_embeddings = std_embeddings.T
        else:
            pass
        class_embeddings = class_embeddings["mean"]
        class_embeddings = class_embeddings.reshape(num_batch_classes, num_templates, -1)
        if not no_aggregate:
            class_embeddings = class_embeddings.mean(dim=1)
            class_embeddings = class_embeddings / class_embeddings.norm(dim=1, keepdim=True)
            class_embeddings = class_embeddings.T
        else:
            pass
        return class_embeddings, std_embeddings

    with torch.no_grad():
        if num_classes_per_batch:
            batched_embeds = [_process_batch(batch) for batch in iter_wrap(batched(classnames, num_classes_per_batch))]
            if not no_aggregate:
                zeroshot_weights = torch.cat([weight for weight, std in batched_embeds], dim=1)
                zeroshot_stds = torch.cat([std for weight, std in batched_embeds], dim=1)
            else:
                zeroshot_weights = torch.cat([weight for weight, std in batched_embeds], dim=0)
                zeroshot_stds = torch.cat([std for weight, std in batched_embeds], dim=0)
        else:
            zeroshot_weights, zeroshot_stds = _process_batch(classnames)
    return zeroshot_weights, zeroshot_stds
