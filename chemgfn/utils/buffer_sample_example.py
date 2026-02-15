#!/usr/bin/env python
"""
Buffer Sampling Usage Example

This script demonstrates how to create and use buffer samples.
"""

import torch
from transformers import AutoTokenizer


def create_buffer_samples(
    tokenizer_name: str = "meta-llama/Meta-Llama-3-8B-Instruct",
    num_samples: int = 100,
    seq_len: int = 10,
    output_path: str = "buffer_samples.pt",
):
    """
    Create example buffer samples file.

    Args:
        tokenizer_name: Tokenizer name
        num_samples: Number of samples to generate
        seq_len: Length of each sample
        output_path: Output file path
    """
    print(f"Creating buffer samples...")
    print(f"  Tokenizer: {tokenizer_name}")
    print(f"  Number of samples: {num_samples}")
    print(f"  Sequence length: {seq_len}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    vocab_size = len(tokenizer)

    # Generate random token IDs (in practice, should be meaningful sequences)
    buffer_samples = torch.randint(0, vocab_size, (num_samples, seq_len), dtype=torch.long)

    # Ensure each sequence ends with EOS token
    buffer_samples[:, -1] = tokenizer.eos_token_id

    # Save
    torch.save(buffer_samples, output_path)

    print(f"\nOK: Buffer samples saved to: {output_path}")
    print(f"  Shape: {buffer_samples.shape}")
    print(f"  Dtype: {buffer_samples.dtype}")
    print(f"\nExample sample:")
    print(f"  Token IDs: {buffer_samples[0].tolist()}")
    print(f"  Decoded: {tokenizer.decode(buffer_samples[0], skip_special_tokens=False)}")


def verify_buffer_samples(buffer_path: str):
    """
    Verify that the buffer samples file format is correct.

    Args:
        buffer_path: Buffer file path
    """
    print(f"\nVerifying buffer samples: {buffer_path}")

    try:
        # Try to load
        buffer = torch.load(buffer_path)

        # Check type
        if isinstance(buffer, torch.Tensor):
            print(f"OK: Type: Tensor")
            print(f"OK: Shape: {buffer.shape}")
            print(f"OK: Dtype: {buffer.dtype}")
            print(f"OK: Number of elements: {buffer.numel()}")

            if buffer.numel() == 0:
                print("WARN: Buffer is empty!")
                return False

        elif isinstance(buffer, list):
            print(f"OK: Type: List")
            print(f"OK: Length: {len(buffer)}")

            if len(buffer) == 0:
                print("WARN: Buffer is empty!")
                return False

            print(f"OK: First element type: {type(buffer[0])}")
            if isinstance(buffer[0], torch.Tensor):
                print(f"OK: First element shape: {buffer[0].shape}")
        else:
            print(f"ERROR: Unknown type: {type(buffer)}")
            return False

        print(f"\nOK: Buffer samples verification passed!")
        return True

    except FileNotFoundError:
        print(f"ERROR: File not found: {buffer_path}")
        return False
    except Exception as e:
        print(f"ERROR: Loading failed: {e}")
        return False


def create_empty_buffer(output_path: str = "empty_buffer.pt"):
    """
    Create an empty buffer file for testing automatic detection.

    Args:
        output_path: Output file path
    """
    print(f"\nCreating empty buffer: {output_path}")

    # Create empty tensor
    empty_buffer = torch.tensor([], dtype=torch.long)

    # Save
    torch.save(empty_buffer, output_path)

    print(f"OK: Empty buffer created: {output_path}")
    print(
        f"  This file will be automatically detected as invalid, buffer sampling will be disabled"
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Buffer Sampling Tool")
    parser.add_argument(
        "--action",
        type=str,
        choices=["create", "verify", "create-empty"],
        default="create",
        help="Action to perform",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="buffer_samples.pt",
        help="Output file path",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=100,
        help="Number of samples",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=10,
        help="Sequence length",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="meta-llama/Meta-Llama-3-8B-Instruct",
        help="Tokenizer name",
    )

    args = parser.parse_args()

    if args.action == "create":
        create_buffer_samples(
            tokenizer_name=args.tokenizer,
            num_samples=args.num_samples,
            seq_len=args.seq_len,
            output_path=args.output,
        )
    elif args.action == "verify":
        verify_buffer_samples(args.output)
    elif args.action == "create-empty":
        create_empty_buffer(args.output)

    print("\n" + "=" * 60)
    print("Usage examples:")
    print("=" * 60)
    print("\n1. Create buffer samples:")
    print(
        "   python buffer_sample_example.py --action create --output my_buffer.pt --num-samples 1000"
    )
    print("\n2. Verify buffer samples:")
    print("   python buffer_sample_example.py --action verify --output my_buffer.pt")
    print("\n3. Create empty buffer (test automatic detection):")
    print("   python buffer_sample_example.py --action create-empty --output empty.pt")
    print("\n4. Use in training:")
    print(
        "   python chemgfn/train.py experiment=SMILES_basic/SMILES_cfg_TB "
        "data.buffer_sample_path=my_buffer.pt"
    )
    print()
