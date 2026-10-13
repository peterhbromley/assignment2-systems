import argparse
from timeit import default_timer as timer

from jaxtyping import Int
import numpy as np
import torch
from torch import Tensor 

from llm.transformer import Transformer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark and profile an LLM.")

    # Model params.
    parser.add_argument("--vocab-size", type=int, default=10000)
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--num-heads", type=int, default=16)
    parser.add_argument("--d-ff", type=int, default=1344)
    parser.add_argument("--rope-theta", type=float, default=10000.0)

    # Benchmark options.
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument(
        "--mode",
        choices=["forward", "forward-backward", "full"],
        default="forward",
    )

    # Runtime.
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cpu", help="For example: cpu, mps, or cuda:0.")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    # Make sure d_model divides evenly across heads
    if args.d_model % args.num_heads:
        parser.error("d_model must divide evenly across heads")

    # Make sure d_model // num_heads (d_k and d_v) is even (for RoPE)
    if (args.d_model // args.num_heads) % 2:
        parser.error("d_model // num_heads must be even (for RoPE implementation).")

    # Validate rope theta value
    if not 0 < args.rope_theta:
        parser.error("--rope-theta must be > 0.")

    return args


def get_random_batch(
    vocab_size: int,
    batch_size: int, 
    context_length: int, 
    device: torch.device,
) -> Int[Tensor, " batch_size context_length"]:
    seqs = np.random.randint(0, vocab_size, (batch_size, context_length))
    return torch.from_numpy(seqs).to(device=device, dtype=torch.long)


def main() -> None:
    args = parse_args()
    print(args)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device)
    dtype = torch.float32

    def synchronize():
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        if device.type == "mps":
            torch.mps.synchronize()

    model = Transformer.from_config(vars(args), device=device, dtype=dtype)
    data = get_random_batch(args.vocab_size, args.batch_size, args.context_length, device)

    fwd_times = []
    for step in range(args.warmup_steps + args.steps):
        synchronize()

        start = timer()
        output = model(data)
        synchronize()
        elapsed = timer() - start
        del output

        if step >= args.warmup_steps:
            fwd_times.append(elapsed)

    fwd_times_ms = np.asarray(fwd_times) * 1000
    print(f"Forward: {fwd_times_ms.mean():.2f} ± {fwd_times_ms.std():.2f} ms")


if __name__ == "__main__":
    main()
