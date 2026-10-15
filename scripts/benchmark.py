import argparse
from contextlib import nullcontext
from timeit import default_timer as timer

from jaxtyping import Int
import numpy as np
import torch
from torch.cuda import nvtx
from torch import Tensor 

from llm.loss import cross_entropy_loss
from llm.optimizer import AdamW
from llm.transformer import Transformer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark and profile an LLM.")

    # Model params.
    parser.add_argument("--vocab-size", type=int, default=10000)
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=12)
    parser.add_argument("--d-model", type=int, default=768)
    parser.add_argument("--num-heads", type=int, default=12)
    parser.add_argument("--d-ff", type=int, default=3072)
    parser.add_argument("--rope-theta", type=float, default=10000.0)

    # Benchmark options.
    parser.add_argument("--profile", action="store_true")
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

    if args.steps <= 0 or args.warmup_steps < 0:
        parser.error("--steps must be > 0 and --warmup-steps must be >= 0.")
    if args.num_heads <= 0:
        parser.error("--num-heads must be > 0.")

    # Make sure d_model divides evenly across heads
    if args.d_model % args.num_heads:
        parser.error("d_model must divide evenly across heads")

    # Make sure d_model // num_heads (d_k and d_v) is even (for RoPE)
    if (args.d_model // args.num_heads) % 2:
        parser.error("d_model // num_heads must be even (for RoPE implementation).")

    # Validate rope theta value
    if not 0 < args.rope_theta:
        parser.error("--rope-theta must be > 0.")

    # If --profile is passed, device should be cuda
    if args.profile and (torch.device(args.device).type != "cuda"):
        parser.error("--profile must only be used with cuda")

    return args


def get_random_batch(
    vocab_size: int,
    batch_size: int, 
    context_length: int, 
    device: torch.device,
) -> tuple[Int[Tensor, " batch_size context_length"], Int[Tensor, " batch_size context_length"]]:
    seqs = np.random.randint(0, vocab_size, (batch_size, context_length + 1))
    return (
        torch.from_numpy(seqs[:, :-1]).to(device=device, dtype=torch.long), 
        torch.from_numpy(seqs[:, 1:]).to(device=device, dtype=torch.long)
    )


def main() -> None:
    args = parse_args()
    print(args)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device)
    dtype = torch.float32

    def profile_range(label: str):
        return nvtx.range(label) if args.profile else nullcontext()

    def synchronize():
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        if device.type == "mps":
            torch.mps.synchronize()

    model = Transformer.from_config(
        vars(args), device=device, dtype=dtype, profile=args.profile,
    )
    inputs, targets = get_random_batch(args.vocab_size, args.batch_size, args.context_length, device)

    run_backward = args.mode in {"forward-backward", "full"}
    run_optimizer = args.mode == "full"
    optimizer = AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1) if run_optimizer else None

    timings = {"Forward": []}
    if run_backward:
        timings["Backward"] = []
    if run_optimizer:
        timings["Optimizer"] = []

    # Run warmup steps with no timing
    for _ in range(args.warmup_steps):
        if run_backward:
            model.zero_grad(set_to_none=True)
        logits = model(inputs)
        if run_backward:
            loss = cross_entropy_loss(logits, targets)
            loss.backward()
            del loss
        if optimizer is not None:
            optimizer.step()
        del logits
    synchronize()

    # Main measurement loop
    with profile_range("full_profiling_run"):
        for _ in range(args.steps):
            with profile_range("single_step"):
                if run_backward:
                    model.zero_grad(set_to_none=True)
                synchronize()

                # Run and time forward pass
                with profile_range("forward_pass"):
                    start = timer()
                    logits = model(inputs)
                    synchronize()
                    timings["Forward"].append(timer() - start)

                if run_backward:
                    with profile_range("cross_entropy_loss"):
                        loss = cross_entropy_loss(logits, targets)
                        synchronize()

                    with profile_range("backward_pass"):
                        start = timer()
                        loss.backward()
                        synchronize()
                        timings["Backward"].append(timer() - start)
                    del loss

                if optimizer is not None:
                    with profile_range("optimizer_step"):
                        start = timer()
                        optimizer.step()
                        synchronize()
                        timings["Optimizer"].append(timer() - start)

                # Release the output and graph outside the timed regions.
                del logits

    for phase, samples in timings.items():
        times_ms = np.asarray(samples) * 1000
        print(f"{phase}: {times_ms.mean():.2f} ± {times_ms.std():.2f} ms")


if __name__ == "__main__":
    main()
