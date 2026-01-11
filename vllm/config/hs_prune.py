from pydantic.dataclasses import dataclass
from pydantic import Field
from .utils import config

@config
@dataclass
class HSPruneConfig:
    enable: bool = True
    classifier_path: str | None = None
    stop_thinking_tokenID: int | None = None
    double_new_line_tokenID: list[int] | None = None
    def compute_hash(self) -> str:
        return str((
            self.enable,
            self.classifier_path,
            self.stop_thinking_tokenID,
            self.double_new_line_tokenID,
        ))

def set_hs_prune_token_ids(hs_prune_config, tokenizer):
    target = "\n\n"
    matching_tokens = []
    for token_id in range(tokenizer.vocab_size):
        token_text = tokenizer.decode([token_id], clean_up_tokenization_spaces=False)
        if target in token_text:
            matching_tokens.append((token_id, token_text))
    hs_prune_config.double_new_line_tokenID = [token_id for token_id, _ in matching_tokens]
    hs_prune_config.stop_thinking_tokenID = tokenizer.added_tokens_encoder.get("</think>", None)

