"""Training entry point for both stages."""
import argparse
import json
import os
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from emsrseg.config import load_config
from emsrseg.models import build_model
from emsrseg.checkpoints import initialize_stage2, set_training_mode
from emsrseg.losses import training_loss
from emsrseg.training_state import capture_rng_state, resume_progress, training_iterator
from emsrseg.evaluation import seed_everything, infer_volume, segment_and_score, sr_metrics
from data.data_provider_labeled_3d import Train
from data.data_sampler import DistIterSampler
from data.provider_valid_3d import Provider_valid
from model.lr_scheduler import CosineAnnealingLR_Restart

SEED = 0

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", "-opt", required=True)
    parser.add_argument("--local-rank", "--local_rank", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--stage1-checkpoint")
    parser.add_argument("--resume", help="本入口保存的包含优化器状态的 .state 文件")
    parser.add_argument("--iterations", type=int, help="覆盖停止迭代数；不改变学习率周期")
    parser.add_argument("--batch-size", type=int, help="所有 GPU 合计 batch")
    parser.add_argument("--workers", type=int)
    args = parser.parse_args()
    cfg = load_config(args.config, args.data_root)
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank))
    if not torch.cuda.is_available():
        raise RuntimeError("Training requires CUDA/MMCV ops.")
    torch.cuda.set_device(local_rank)
    if world > 1:
        dist.init_process_group("nccl")
    device = torch.device("cuda", local_rank)
    output = Path(args.output).resolve()
    
    exists = output.exists()
    if world > 1:
        dist.barrier()
    if exists and not args.resume:
        raise FileExistsError(f"Refusing to overwrite an existing run: {output}")
    if rank == 0:
        output.mkdir(parents=True, exist_ok=bool(args.resume))
    if world > 1:
        dist.barrier()
    stage = cfg["stage"]
    training = cfg["train"]
    dataset_cfg = cfg["datasets"]["train"]
    if args.batch_size is not None:
        dataset_cfg["batch_size"] = args.batch_size
    if args.workers is not None:
        dataset_cfg["n_workers"] = args.workers
    batch_size = dataset_cfg["batch_size"]
    if batch_size <= 0 or batch_size % world:
        raise ValueError("Global batch size must be a positive multiple of world size.")
    seed_everything(SEED)
    torch.backends.cudnn.benchmark = True
    dataset = Train(dataset_cfg, cfg["scale"])
    sampler = DistIterSampler(dataset, world, rank, ratio=200)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size//world, sampler=sampler, drop_last=True,
        num_workers=dataset_cfg["n_workers"])
    model = build_model(cfg)
    frozen = []
    if stage == 2:
        frozen = initialize_stage2(
            model, build_model(cfg,stage=1),
            args.stage1_checkpoint or cfg["pretrain"]["stage1_model_path"])
    if world > 1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.to(device)
    groups = []
    lr = training["lr_G"]
    if stage == 1:
        for sr_group in (True, False):
            params = [p for n,p in model.named_parameters() if p.requires_grad and ("sr" in n.lower()) == sr_group]
            groups.append(dict(params=params, lr=lr if sr_group else lr*training["other_lr_multiplier"]))
    else:
        groups = [dict(params=[p for p in model.parameters() if p.requires_grad],lr=lr)]
    optimizer = torch.optim.Adam(groups, betas=(training["beta1"],training["beta2"]),
                                 weight_decay=training.get("weight_decay_G") or 0)
    scheduler = CosineAnnealingLR_Restart(
        optimizer, training["T_period"], restarts=training["restarts"],
        weights=training["restart_weights"], eta_min=training["eta_min"])
    iteration, epoch = 0, 0
    progress = None
    if args.resume:
        state = torch.load(args.resume,map_location=device)
        if state["setting"] != cfg["setting"] or state["stage"] != stage:
            raise ValueError("Resume setting/stage mismatch")
        model.load_state_dict(state["model"],strict=True)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        iteration, epoch = state["iteration"], state["epoch"]
        progress = resume_progress(
            state, rank, world, batch_size, dataset_cfg["n_workers"])
        del state
    network = DistributedDataParallel(model,device_ids=[local_rank],find_unused_parameters=False) if world > 1 else model
    if rank == 0:
        (output/"config.json").write_text(json.dumps(cfg,indent=2),encoding="utf-8")
        validation = Provider_valid(cfg["datasets"]["val"],cfg["scale"])
    stop = args.iterations or int(training["niter"])
    while iteration < stop:
        sampler.set_epoch(epoch)
        iterator, completed, epoch_rng = training_iterator(loader, progress)
        progress = None
        for batch_index, batch in enumerate(iterator, start=completed + 1):
            if iteration >= stop:
                break
            set_training_mode(model,frozen)
            batch = {k:v.to(device) if isinstance(v,torch.Tensor) else v for k,v in batch.items()}
            optimizer.zero_grad()
            result = network(batch["lr"],batch["hr"])
            loss, parts = training_loss(result,batch,stage,training.get("loss_prior_factor",0.1))
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite loss at iteration {iteration+1}")
            loss.backward()
            optimizer.step()
            scheduler.step()
            iteration += 1
            if rank == 0 and (iteration == 1 or iteration % training["logger_freq"] == 0):
                print(json.dumps(dict(iteration=iteration,loss=float(loss.detach()),
                                      **{k:float(v.detach()) for k,v in parts.items()})),flush=True)
            validate = iteration % int(training["val_freq"]) == 0 and iteration >= training.get("validation_start",0)
            if validate:
                if world > 1: dist.barrier()
                if rank == 0:
                    
                    model.eval()
                    torch.backends.cudnn.benchmark = False
                    affinities,sr = infer_volume(model,validation,device,stage)
                    _,scores = segment_and_score(affinities,validation.get_gt_lb().astype(np.uint32))
                    scores.update(sr_metrics(sr,validation.get_hr_data()))
                    scores["iteration"] = iteration
                    with (output/"validation.jsonl").open("a",encoding="utf-8") as stream:
                        stream.write(json.dumps(scores)+"\n")
                    torch.backends.cudnn.benchmark = True
                if world > 1: dist.barrier()
            if iteration % int(cfg["save_checkpoint_freq"]) == 0 or iteration == stop:
                local_progress = dict(
                    epoch_rng=epoch_rng, rng=capture_rng_state(),
                    batches_completed=batch_index)
                rank_progress = [None] * world
                if world > 1:
                    dist.all_gather_object(rank_progress, local_progress)
                else:
                    rank_progress[0] = local_progress
                if rank == 0:
                    torch.save(model.state_dict(),output/f"{iteration}_G.pth")
                    torch.save(dict(
                        model=model.state_dict(),optimizer=optimizer.state_dict(),
                        scheduler=scheduler.state_dict(),iteration=iteration,epoch=epoch,
                        setting=cfg["setting"],stage=stage,
                        training_progress=dict(
                            world_size=world,batch_size=batch_size,
                            workers=dataset_cfg["n_workers"],ranks=rank_progress)),
                        output/f"{iteration}.state")
            if iteration == stop:
                break
        epoch += 1
    if world > 1:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
