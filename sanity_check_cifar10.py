"""
Sanity check run on two CIFAR-10 images from different classes.
Loads one image per class, runs a single forward/backward step.
"""
import argparse
import logging
import os

import torch
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
        patch_size=args.patch_size,
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
    x = torch.stack(images, dim=0).to(device)
    y = torch.tensor(labels, device=device)

    t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)
    model_kwargs = dict(y=y)
    loss_dict = diffusion.training_losses(model, x, t, model_kwargs)
    loss = loss_dict["loss"].mean()
    aux = getattr(model, "last_optional_aux_loss", None)
    if aux is not None:
        loss = loss + args.optional_aux_weight * aux

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0)
    opt.zero_grad()
    loss.backward()
    opt.step()

    logger.info(
        f"Sanity check complete. classes={classes}, loss={loss.item():.4f}, "
        f"weight_mean={getattr(model, 'last_optional_weight_mean', torch.tensor(float('nan')))}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True, help="CIFAR-10 root directory")
    parser.add_argument("--results-dir", type=str, default="results_cifar10_sanity")
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-B/2")
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--patch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-latents-basic", type=int, default=32)
    parser.add_argument("--num-latents-optional", type=int, default=32)
    parser.add_argument("--cross-attn-interval", type=int, default=4)
    parser.add_argument("--optional-weight-temperature", type=float, default=1.0)
    parser.add_argument("--optional-weight-bias", type=float, default=0.0)
    parser.add_argument("--optional-target-mean", type=float, default=0.5)
    parser.add_argument("--optional-aux-weight", type=float, default=0.1)
    parser.add_argument(
        "--classes",
        type=int,
        nargs=2,
        default=[0, 1],
        help="Two different CIFAR-10 class indices (e.g., 0 1).",
    )
    args = parser.parse_args()
    main(args)
