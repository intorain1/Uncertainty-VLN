""" ProLIP zero-shot evaluation for uncertainty
"""
import logging

import numpy as np
import torch

from src.base import get_input_dtype
from .precision import get_autocast

from eccv_caption import Metrics


def eval_clsf_unc(stds, corrects_top1, corrects_top5,
                  n_bins=10, verbose=True, verbose_prefix=""):
    if not len(stds):
        return {}, {}

    corrects_top1 = np.array(corrects_top1)
    corrects_top5 = np.array(corrects_top5)
    stds = np.array(stds)
    sorted_inds = np.argsort(stds)
    bin_size = len(sorted_inds) // n_bins

    unc_vs_acc = []
    unc_vs_acc_top5 = []
    for hist_idx in range(n_bins):
        cur_inds = sorted_inds[range(hist_idx * bin_size, (hist_idx + 1) * bin_size)]
        unc_vs_acc.append(np.mean(corrects_top1[cur_inds]))
        unc_vs_acc_top5.append(np.mean(corrects_top5[cur_inds]))

    unc_vs_acc_str = [f"{v * 100.:.2f}" for v in unc_vs_acc]
    unc_vs_acc_top5_str = [f"{v * 100.:.2f}" for v in unc_vs_acc_top5]

    unc_vs_acc_coef = {
        "top1": np.corrcoef(np.arange(n_bins), unc_vs_acc)[0, 1],
        "top5": np.corrcoef(np.arange(n_bins), unc_vs_acc_top5)[0, 1],
    }
    std_stats = {
        "imagenet-sigma-min": np.min(stds),
        "imagenet-sigma-max": np.max(stds),
        "imagenet-sigma-mean": np.mean(stds),
        "imagenet-sigma-median": np.median(stds)
    }
    std_stats = {f"stats/{k}": v for k, v in std_stats.items()}
    if verbose:
        logging.info(f"{verbose_prefix}unc_vs_acc top1: {unc_vs_acc_str}")
        logging.info(f"{verbose_prefix}unc_vs_acc top5: {unc_vs_acc_top5_str}")
        logging.info(f"{verbose_prefix}sigma-stats: {std_stats}")

    return unc_vs_acc_coef, std_stats


def eval_ret_unc(istds, sstds, iids, sids, i2t, t2i, ret, n_bins=10):
    if not len(istds):
        return

    iinds = np.argsort(istds)
    sinds = np.argsort(sstds)
    i_bin_size = len(iinds) // n_bins
    s_bin_size = len(sinds) // n_bins

    METRICS = ("eccv_r1", "eccv_map_at_r", "coco_5k_r1")
    unc_vs_scores = {m: {"img": [], "cap": []} for m in METRICS}
    for hist_idx in range(n_bins):
        metric = Metrics()
        cur_iinds = iinds[range(hist_idx * i_bin_size, (hist_idx + 1) * i_bin_size)]
        cur_iids = set([iids[idx] for idx in cur_iinds])
        _i2t = {iid: _sims for iid, _sims in i2t.items() if iid in cur_iids}

        cur_sinds = sinds[range(hist_idx * s_bin_size, (hist_idx + 1) * s_bin_size)]
        cur_sids = set([sids[idx] for idx in cur_sinds])
        _t2i = {sid: _sims for sid, _sims in t2i.items() if sid in cur_sids}

        # reset GTs
        coco_gts = {}
        coco_gts["i2t"] = {k: v for k, v in metric.coco_gts["i2t"].items() if int(k) in _i2t}
        coco_gts["t2i"] = {k: v for k, v in metric.coco_gts["t2i"].items() if int(k) in _t2i}
        metric.coco_gts = coco_gts

        eccv_gts = {}
        eccv_gts["i2t"] = {k: v for k, v in metric.eccv_gts["i2t"].items() if int(k) in _i2t}
        eccv_gts["t2i"] = {k: v for k, v in metric.eccv_gts["t2i"].items() if int(k) in _t2i}
        metric.eccv_gts = eccv_gts

        cur_scores = metric.compute_all_metrics(
            _i2t, _t2i,
            target_metrics=METRICS,
            Ks=(1,),
            verbose=False,
        )
        for m in METRICS:
            unc_vs_scores[m]["img"].append(cur_scores[m]["i2t"])
            unc_vs_scores[m]["cap"].append(cur_scores[m]["t2i"])

    for m in METRICS:
        unc_vs_perf_str = [f"{v * 100.:.2f}" for v in unc_vs_scores[m]["img"]]
        logging.info(f"COCO-unc_vs_{m}-img: {unc_vs_perf_str}")
        unc_vs_perf_str = [f"{v * 100.:.2f}" for v in unc_vs_scores[m]["cap"]]
        logging.info(f"COCO-unc_vs_{m}-cap: {unc_vs_perf_str}")

    for m in METRICS:
        for mod in ("img", "cap"):
            ret[f"unc/{m}_{mod}"] = np.corrcoef(
                np.arange(n_bins), unc_vs_scores[m][mod])[0, 1]

    std_stats = {
        "coco-img-sigma-min": np.min(istds),
        "coco-img-sigma-max": np.max(istds),
        "coco-img-sigma-mean": np.mean(istds),
        "coco-img-sigma-median": np.median(istds),
        "coco-txt-sigma-min": np.min(sstds),
        "coco-txt-sigma-max": np.max(sstds),
        "coco-txt-sigma-mean": np.mean(sstds),
        "coco-txt-sigma-median": np.median(sstds)
    }
    logging.info(f"COCO-sigma-stats: {std_stats}")

    ret.update({f"stats/{k}": v for k, v in std_stats.items()})


def run_ret(model, tokenizer, dataloader, args):
    autocast = get_autocast(args.precision)
    input_dtype = get_input_dtype(args.precision)

    with torch.no_grad():
        iids, sids, iembs, sembs = list(), list(), list(), list()
        istds, sstds = list(), list()
        for images, target in dataloader:
            images = images.to(device=args.device, dtype=input_dtype)
            caps = [t.split("<|cap|>") for t in target]
            _iids = [int(c[0].split("<|iid|>")[1]) for c in caps]
            iids += _iids

            _caps = list()
            for c in caps:
                cc = [cc.split("<|sid|>") for cc in c[1:]]
                _caps += cc

            caps, _sids = zip(*_caps)
            _sids = [int(s) for s in _sids]
            sids += _sids
            caps = tokenizer(caps)
            caps = caps.to(args.device)

            with autocast():
                output = model(text=caps)
                text_features = output["text_features"]["mean"] if isinstance(output, dict) else output[1]
                sembs.append(text_features.cpu())
                if output["text_features"]["std"] is not None:
                    text_stds = torch.exp(output["text_features"]["std"]).sum(dim=-1).tolist()
                    sstds.extend(text_stds)

                output = model(image=images)
                image_features = output["image_features"]["mean"] if isinstance(output, dict) else output[0]
                iembs.append(image_features.cpu())
                if output["image_features"]["std"] is not None:
                    image_stds = torch.exp(output["image_features"]["std"]).sum(dim=-1).tolist()
                    istds.extend(image_stds)

        iembs = torch.cat(iembs, dim=0).float()
        sembs = torch.cat(sembs, dim=0).float()
        istds = np.array(istds)
        sstds = np.array(sstds)

        logging.info("Computing retrieval logits")
        logits_per_image = iembs @ sembs.T
        logits_per_text = logits_per_image.T

        if len(istds):
            logits_per_image = logits_per_image - sstds.reshape(1, -1) / 2
            logits_per_text = logits_per_text - istds.reshape(1, -1) / 2

        sims_per_image = logits_per_image.numpy()
        sims_per_text = logits_per_text.numpy()

        logging.info("Sorting retrieval scores")
        i2t_idxs = np.argsort(-sims_per_image, axis=1, kind="stable")
        t2i_idxs = np.argsort(-sims_per_text, axis=1, kind="stable")
        rank_sids = torch.tensor(sids)[i2t_idxs]
        rank_iids = torch.tensor(iids)[t2i_idxs]

        logging.info("Computing retrieval scores")
        i2t, t2i = dict(), dict()
        for iid, sid in zip(iids, rank_sids):
            i2t[iid] = sid.tolist()
        for sid, iid in zip(sids, rank_iids):
            t2i[sid] = iid.tolist()

    ret = dict()

    metric = Metrics()

    scores = metric.compute_all_metrics(
        i2t,
        t2i,
        target_metrics=(
            "eccv_r1",
            "eccv_map_at_r",
            "eccv_rprecision",
            # "coco_1k_recalls",
            "coco_5k_recalls",
            "cxc_recalls",
        ),
        Ks=(1, 5, 10),
        verbose=False,
    )

    for k, v in scores.items():
        for kk, vv in v.items():
            ret[f"coco_recalls/{k}_{kk}"] = vv

    eval_ret_unc(istds, sstds, iids, sids, i2t, t2i, ret, n_bins=10)

    del iids, sids, iembs, sembs
    del istds, sstds
    del i2t, t2i

    return ret
