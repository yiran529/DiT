"""
Sanity check run on two CIFAR-10 images from different classes.
Loads one image per class, runs a single forward/backward step.
"""
import argparse
import logging
import os

import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
from torchvision.datasets import CIFAR10

from diffusion import create_diffusion
from models import DiT_models


def create_logger(log_dir: str):
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(), logging.FileHandler(os.path.join(log_dir, "log.txt"))],
    )
    return logging.getLogger(__name__)


def compute_grad_norm(parameters):
    total = 0.0
    for param in parameters:
        if param.grad is not None:
            grad = param.grad.detach()
            total += grad.norm(2).item() ** 2
    return total ** 0.5 if total > 0 else 0.0


def find_one_per_class(dataset, classes):
    picked = {}
    for idx, (_, y) in enumerate(dataset):
        if y in classes and y not in picked:
            picked[y] = idx
        if len(picked) == len(classes):
            break
    return [picked[c] for c in classes]


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.set_device(0)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    logger = create_logger(args.results_dir)

    assert args.image_size % args.patch_size == 0, "image_size must be divisible by patch_size."
    model = DiT_models[args.model](
        input_size=args.image_size,
        num_classes=args.num_classes,
        num_latents_basic=args.num_latents_basic,
        num_latents_optional=args.num_latents_optional,
        cross_attn_interval=args.cross_attn_interval,
        optional_weight_temperature=args.optional_weight_temperature,
        optional_weight_bias=args.optional_weight_bias,
        optional_target_mean=args.optional_target_mean,
        # patch_size=args.patch_size,
    ).to(device)
    diffusion = create_diffusion(timestep_respacing="")

    transform = transforms.Compose(
        [
            transforms.Resize(args.image_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
        ]
    )
    dataset = CIFAR10(root=args.data_path, train=True, download=True, transform=transform)

    classes = args.classes
    if len(classes) != 2 or classes[0] == classes[1]:
        raise ValueError("Please provide two different class indices via --classes (e.g., 0 1).")

    indices = find_one_per_class(dataset, classes)
    images, labels = zip(*[dataset[i] for i in indices])
    images_tensor = torch.stack(images, dim=0)
    labels_tensor = torch.tensor(labels)
    data_loader = DataLoader(
        TensorDataset(images_tensor, labels_tensor),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
    )

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0)
    logger.info(
        f"Training for {args.epochs} epochs on {len(images_tensor)} samples (classes={classes})."
        f"config: {args}"
    )

    for epoch in range(args.epochs):
        epoch_loss = 0.0
        batch_count = 0
        last_grad_norm = 0.0
        for batch_x, batch_y in data_loader:
            batch_count += 1
            x = batch_x.to(device)
            y = batch_y.to(device)
            t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)
            model_kwargs = dict(y=y)
            loss_dict = diffusion.training_losses(model, x, t, model_kwargs)
            loss = loss_dict["loss"].mean()
            aux = getattr(model, "last_optional_aux_loss", None)
            if aux is not None:
                loss = loss + args.optional_aux_weight * aux

            opt.zero_grad()
            loss.backward()
            last_grad_norm = compute_grad_norm(model.parameters())
            opt.step()

            epoch_loss += loss.item()

        if not (epoch % 10 == 0 or epoch == args.epochs - 1): 
            continue

        avg_loss = epoch_loss / max(batch_count, 1)
        weight_mean = getattr(model, "last_optional_weight_mean", None)
        if isinstance(weight_mean, torch.Tensor):
            weight_mean_val = float(weight_mean.detach().cpu().item())
        elif weight_mean is None:
            weight_mean_val = float('nan')
        else:
            weight_mean_val = float(weight_mean)
        aux_val = getattr(model, "last_optional_aux_loss", None)
        aux_log = float('nan') if aux_val is None else float(aux_val.detach().cpu().item())
        logger.info(
            f"Epoch {epoch + 1}/{args.epochs}: loss={avg_loss:.4f}, grad_norm={last_grad_norm:.4f}, "
            f"weight_mean={weight_mean_val:.4f}, aux={aux_log:.4f}"
        )

    ckpt_path = os.path.join(args.results_dir, f"ckpt_sanity_check.pt")
    torch.save(
        {
            "model": model.state_dict(),
            "args": vars(args),
        },
        ckpt_path,
    )
    logger.info(f"Saved checkpoint to {ckpt_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True, help="CIFAR-10 root directory")
    parser.add_argument("--results-dir", type=str, default="results_cifar10_sanity")
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-Tiny/4")
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--patch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=3000, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=2, help="Mini-batch size for the sanity loop.")
    parser.add_argument("--num-latents-basic", type=int, default=24)
    parser.add_argument("--num-latents-optional", type=int, default=8)
    parser.add_argument("--cross-attn-interval", type=int, default=4)
    parser.add_argument("--optional-weight-temperature", type=float, default=1.0)
    parser.add_argument("--optional-weight-bias", type=float, default=0.0)
    parser.add_argument("--optional-target-mean", type=float, default=0.5)
    parser.add_argument("--optional-aux-weight", type=float, default=0.01)
    parser.add_argument(
        "--classes",
        type=int,
        nargs=2,
        default=[0, 1],
        help="Two different CIFAR-10 class indices (e.g., 0 1).",
    )
    args = parser.parse_args()
    main(args)
