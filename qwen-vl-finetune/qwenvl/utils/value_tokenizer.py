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
        self, llm_path: str = "/project/peilab/junhao/Qwen3-VL/checkpoints/Qwen2.5-VL-3B-Instruct-resize", 
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
        # 确保从本地加载 tokenizer，不触发网络下载
        local_files_only = os.path.exists(llm_path) if os.path.isabs(llm_path) or not llm_path.startswith(('http://', 'https://')) else False
        self.tokenizer = AutoTokenizer.from_pretrained(
            llm_path,
            local_files_only=local_files_only
        )  # 使用AutoTokenizer以兼容Qwen2.5-VL
        self.n_bins, self.min_value, self.max_value = bins, min_value, max_value

        # Create Uniform Bins + Compute Bin Centers (fix off-by-one: use n_bins + 1 edges)
        self.bin_edges = np.linspace(min_value, max_value, self.n_bins + 1)
        self.bin_centers = (self.bin_edges[:-1] + self.bin_edges[1:]) / 2.0

        # [Contract] Set "value_token_begin_idx" based on `self.tokenizer.vocab_size - (self.n_bins + 1)`
        #   =>> Assumes we're always overwriting the final `n_bins` tokens of the vocabulary!
        # self.value_token_begin_idx: int = int(self.tokenizer.vocab_size - (self.n_bins + 1))
        self.vocab_size = len(self.tokenizer)

        # Build mapping from bin indices to extra_id token IDs
        # Since we added <extra_id_0> to <extra_id_200>, map bin index i to <extra_id_i>
        self.extra_id_token_ids = []
        for i in range(self.n_bins):
            token_name = f"<extra_id_{i}>"
            tid = self.tokenizer.convert_tokens_to_ids(token_name)
            if tid is None or tid == self.tokenizer.unk_token_id:
                raise ValueError(f"Could not find token {token_name} in tokenizer. "
                               f"Make sure the model was trained with extra_id tokens added via add_token.py")
            self.extra_id_token_ids.append(tid)
        self.extra_id_token_ids = np.array(self.extra_id_token_ids)

    def __call__(self, value: np.ndarray) -> Union[str, List[str]]:
        """
        Clip & bin values to *the last `n_bins` tokens* of the vocabulary.

        np.digitize returns indices in range [1, n_bins+1] for values in [min_value, max_value].
        We map these to token IDs in range [vocab_size - n_bins, vocab_size - 1].
        """
        # Ensure value is a numpy array
        value = np.asarray(value)
        value = np.clip(value, a_min=float(self.min_value), a_max=float(self.max_value))
        discretized_value = np.digitize(value, self.bin_edges)

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
            result = self.tokenizer.decode([token_id])


            return result
        else:
            # Batch: return list of decoded strings
            result = self.tokenizer.batch_decode(token_ids.astype(int).tolist())


            return result

    def decode_token_ids_to_values(self, value_token_ids: np.ndarray) -> np.ndarray:
        """
        Returns continuous values for discrete value token IDs.

        Token IDs correspond to <extra_id_{i}> tokens where i is the bin index.
        We convert back to bin indices [0, n_bins-1] and then to bin centers.
        """
        # Convert token IDs back to bin indices by finding their position in extra_id_token_ids
        bin_indices = []
        for tid in value_token_ids:
            try:
                bin_idx = np.where(self.extra_id_token_ids == tid)[0][0]
                bin_indices.append(bin_idx)
            except IndexError:
                token_str = self.tokenizer.decode([tid])
                raise ValueError(f"Token ID {tid} ({token_str}) is not a valid extra_id token. "
                               f"Expected one of: {self.extra_id_token_ids}")

        bin_indices = np.array(bin_indices)

        return self.bin_centers[bin_indices]

# Test, test ...
# if __name__ == "__main__":
#     value = np.array([-0.5, -0.3, -0.8, -0.2, -0.9, -0.1, -1.0])
#     value_tokenizer = ValueTokenizer()
#     print(value_tokenizer(value))

