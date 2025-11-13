""" ProLIP zero shot evaluation
This code is highly similar to the original code, but some of the codes were updated for uncertainty eval.

Original code: https://github.com/mlfoundations/open_clip/blob/v2.24.0/src/training/zero_shot.py
"""
import logging

import torch
from tqdm import tqdm

from prolip import get_input_dtype, get_tokenizer, build_zero_shot_classifier, \
    IMAGENET_CLASSNAMES, OPENAI_IMAGENET_TEMPLATES
from .precision import get_autocast
from .zero_shot_unc import eval_clsf_unc, run_ret


def accuracy(output, target, topk=(1,)):
    return accuracy_with_correct(output, target, topk)[0]


def accuracy_with_correct(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk], correct


def eval_zeroshot_classifier(model, classifier, dataloader,
                             unc_weight=None, n_bins=10,
                             precision="amp", device="cuda", batch_size=32,
                             verbose=True, verbose_prefix=""):
    autocast = get_autocast(precision)
    input_dtype = get_input_dtype(precision)

    # [768, num_classes]
    if classifier.shape[1] < 5:
        # prevent index error
        topk = (1, classifier.shape[1])
    else:
        topk = (1, 5)

    with torch.no_grad():
        top1, top5, n = 0., 0., 0.
        stds, corrects_top1, corrects_top5 = [], [], []
        for images, target in tqdm(dataloader, unit_scale=batch_size, disable=not verbose):
            images = images.to(device=device, dtype=input_dtype)
            target = target.to(device)

            with autocast():
                # predict
                output = model(image=images)
                output = output['image_features'] if isinstance(output, dict) else output[0]
                image_features = output['mean']
                logits = 100. * image_features @ classifier
                if unc_weight is not None:
                    logits = logits - 50.0 * unc_weight

                if output["std"] is not None:
                    image_stds = output["std"]
                    stds.extend(torch.exp(image_stds).sum(dim=-1).tolist())

            # measure accuracy
            (acc1, acc5), correct = accuracy_with_correct(logits, target, topk=topk)
            corrects_top1.extend(correct[0].tolist())
            corrects_top5.extend(correct.sum(0).tolist())
            top1 += acc1
            top5 += acc5
            n += images.size(0)

    top1 = (top1 / n)
    top5 = (top5 / n)
    unc_vs_acc_coef, std_stats = eval_clsf_unc(stds, corrects_top1, corrects_top5,
                                               n_bins, verbose, verbose_prefix)

    return top1, top5, unc_vs_acc_coef, std_stats


def run(model, classifier, dataloader, args):
    return eval_zeroshot_classifier(
        model, classifier, dataloader,
        precision=args.precision,
        device=args.device,
        batch_size=args.batch_size,
        verbose_prefix="ImageNet-"
    )


def zero_shot_eval(model, data, epoch, args, tokenizer=None):
    if "imagenet-val" not in data and "imagenet-v2" not in data:
        return {}
    if args.zeroshot_frequency == 0:
        return {}
    if (epoch % args.zeroshot_frequency) != 0 and epoch != args.epochs:
        return {}
    if args.distributed and not args.horovod:
        model = model.module

    logging.info('Starting zero-shot evaluation.')

    results = {}
    if "coco-test" in data:
        logging.info("Building zero-shot retrieval")
        recalls = run_ret(model, tokenizer, data["coco-test"].dataloader, args)
        results.update(recalls)
        logging.info('Finished zero-shot retrieval.')

    if tokenizer is None:
        tokenizer = get_tokenizer(args.model)

    logging.info('Building zero-shot classifier')
    autocast = get_autocast(args.precision)
    with autocast():
        classifier = build_zero_shot_classifier(
            model,
            tokenizer=tokenizer,
            classnames=IMAGENET_CLASSNAMES,
            templates=OPENAI_IMAGENET_TEMPLATES,
            num_classes_per_batch=10,
            device=args.device,
            use_tqdm=True,
        )

    logging.info('Using classifier')
    if 'imagenet-val' in data:
        top1, top5, unc_vs_acc, std_stats = run(model, classifier, data['imagenet-val'].dataloader, args)
        results['imagenet-zeroshot-val-top1'] = top1
        results['imagenet-zeroshot-val-top5'] = top5
        if unc_vs_acc:
            results['unc/imagenet-zeroshot-val-unc-vs-acc-top1'] = unc_vs_acc["top1"]
            results['unc/imagenet-zeroshot-val-unc-vs-acc-top5'] = unc_vs_acc["top5"]
            results.update(std_stats)
    if 'imagenet-v2' in data:
        top1, top5, unc_vs_acc, std_stats = run(model, classifier, data['imagenet-v2'].dataloader, args)
        results['imagenetv2-zeroshot-val-top1'] = top1
        results['imagenetv2-zeroshot-val-top5'] = top5
        if unc_vs_acc:
            results['unc/imagenetv2-zeroshot-val-unc-vs-acc-top1'] = unc_vs_acc["top1"]
            results['unc/imagenetv2-zeroshot-val-unc-vs-acc-top5'] = unc_vs_acc["top5"]
            results.update({k.replace("imagenet", "imagenetv2"): v for k, v in std_stats.items()})

    logging.info('Finished zero-shot imagenet.')

    return results
