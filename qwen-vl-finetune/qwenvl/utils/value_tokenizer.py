"""
value_tokenizer.py

Extension class; wraps base LLM/VLM tokenizer with logic to discretize and tokenize continuous values.
"""

from typing import List, Union

import os
import numpy as np
from transformers import AutoTokenizer


class ValueTokenizer:
    def __init__(
        self, llm_path: str = "/project/peilab/junhao/Value_Function/qwen-vl-finetune/checkpoints/Qwen2.5-VL-3B-Instruct-resize", 
        bins: int = 201, min_value: float = -1.0, max_value: float = 0.0
    ) -> None:
        """
        Discretizes continuous values into N bins per dimension and maps to the least used tokens.

        NOTE =>> by default, assumes a BPE-style tokenizer akin to the LlamaTokenizer, where *the least used tokens*
                 appear at the end of the vocabulary!

        :param llm_path: LLM path to find tokenizer.
        :param bins: Number of bins for each continuous value; we'll adopt a uniform binning strategy.
        :param min_value: Minimum value (for clipping, setting lower bound on bin interval).
        :param max_value: Maximum value (for clipping, setting upper bound on bin interval).
        """
        self.tokenizer = AutoTokenizer.from_pretrained(llm_path)  # 使用AutoTokenizer兼容Qwen2.5-VL
        self.n_bins, self.min_value, self.max_value = bins, min_value, max_value

        # Create Uniform Bins + Compute Bin Centers
        self.bins = np.linspace(min_value, max_value, self.n_bins + 1)
        self.bin_centers = (self.bins[:-1] + self.bins[1:]) / 2.0

        # [Contract] Set "value_token_begin_idx" based on `self.tokenizer.vocab_size - (self.n_bins + 1)`
        #   =>> Assumes we're always overwriting the final `n_bins` tokens of the vocabulary!
        # self.value_token_begin_idx: int = int(self.tokenizer.vocab_size - (self.n_bins + 1))
        self.vocab_size = len(self.tokenizer)

        # Build mapping from bin indices to extra_id token IDs
        # Since we added <extra_id_0> to <extra_id_200>, map bin index i to <extra_id_i>
        self.extra_id_token_ids = []
        self.token_id_to_bin_idx = {}
        for i in range(self.n_bins):
            token_name = f"<extra_id_{i}>"
            tid = self.tokenizer.convert_tokens_to_ids(token_name)
            if tid is None or tid == self.tokenizer.unk_token_id:
                raise ValueError(f"Could not find token {token_name} in tokenizer. "
                               f"Make sure the model was trained with extra_id tokens added via add_token.py")
            self.extra_id_token_ids.append(tid)
            self.token_id_to_bin_idx[tid] = i
        self.extra_id_token_ids = np.array(self.extra_id_token_ids)

    def __call__(self, value: np.ndarray) -> Union[str, List[str]]:
        """
        Clip & bin values to *the last `n_bins` tokens* of the vocabulary.

        np.digitize returns indices in range [1, n_bins+1] for values in [min_value, max_value].
        We map these to token IDs in range [vocab_size - n_bins, vocab_size - 1].
        """
        value = np.clip(value, a_min=float(self.min_value), a_max=float(self.max_value))
        discretized_value = np.digitize(value, self.bins)

        # np.digitize returns [1, n_bins+1], we need to map to [0, n_bins-1] then to token IDs
        # discretized_value - 1 gives [0, n_bins]
        # Clip to ensure we're within valid range [0, n_bins-1]
        bin_indices = np.clip(discretized_value - 1, 0, self.n_bins - 1)

        # Map bin indices to <extra_id_{i}> token IDs
        token_ids = self.extra_id_token_ids[bin_indices]

        # Handle single element vs. batch
        if value.ndim == 0 or (value.ndim == 1 and value.shape[0] == 1):
            # Single value: return decoded string
            token_id = int(token_ids.item() if token_ids.ndim > 0 else token_ids)
            return self.tokenizer.decode([token_id])
        else:
            # Batch: return list of decoded strings
            return self.tokenizer.batch_decode(token_ids.astype(int).tolist())

    def decode_token_ids_to_values(self, value_token_ids: np.ndarray) -> np.ndarray:
        """
        Returns continuous values for discrete value token IDs.

        Token IDs correspond to <extra_id_{i}> tokens where i is the bin index.
        We convert back to bin indices [0, n_bins-1] and then to bin centers.
        """
        try:
            bin_indices = np.array([self.token_id_to_bin_idx[tid] for tid in value_token_ids])
            return self.bin_centers[bin_indices]
        except KeyError as error:
            tid = error.args[0]
            token_str = self.tokenizer.decode([tid])
            raise ValueError(f"Token ID {tid} ({token_str}) is not a valid extra_id token. "
                           f"Expected one of: {self.extra_id_token_ids}")

# Test, test ...
# if __name__ == "__main__":
#     value = np.array([-1.0, -0.8, -0.5, -0.2, 0.0])
#     value_tokenizer = ValueTokenizer()
#     print(value_tokenizer(value))