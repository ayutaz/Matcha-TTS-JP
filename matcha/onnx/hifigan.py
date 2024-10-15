from matcha.cli import load_vocoder, load_matcha, process_text

from argparse import ArgumentParser

import numpy as np
import torch
from torch import nn


class Vocoder(nn.Module):
    def __init__(self, hifigan):
        super().__init__()
        self.hifigan = hifigan

    def forward(self, mel, mel_lengths):
        mel = mel.detach().clone()
        mel_lengths = mel_lengths.detach().clone()
        wavs = self.hifigan(mel)
        lengths = mel_lengths * 256
        return wavs.squeeze(1), lengths


def main():
    parser = ArgumentParser(description="")
    parser.add_argument("--vocoder_path", type=str, required=True)
    parser.add_argument("--vocoder_name", type=str, required=True)

    args = parser.parse_args()

    vocoder, _ = load_vocoder(args.vocoder_name, args.vocoder_path, "cpu")

    dummy_input = torch.randn(1, 80, 93)
    dummy_input_lengths = torch.tensor([93])

    model = Vocoder(vocoder)
    exporter = torch.onnx.dynamo_export(model, dummy_input, dummy_input_lengths)
    exporter.save("vocoder.onnx")


if __name__ == "__main__":
    main()