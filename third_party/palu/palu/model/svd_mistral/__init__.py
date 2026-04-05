import transformers
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from .configuration_palu_mistral import PaluMistralConfig
from .modeling_palu_mistral import PaluMistralForCausalLM

_PALU_MISTRAL_TOKENIZER = getattr(transformers, "MistralTokenizer", None)
if _PALU_MISTRAL_TOKENIZER is None:
    _PALU_MISTRAL_TOKENIZER = getattr(transformers, "MistralCommonTokenizer", None)
if _PALU_MISTRAL_TOKENIZER is None:
    raise ImportError("Neither MistralTokenizer nor MistralCommonTokenizer is available in transformers.")

AutoConfig.register("palumistral", PaluMistralConfig)
AutoModelForCausalLM.register(PaluMistralConfig, PaluMistralForCausalLM)
AutoTokenizer.register(PaluMistralConfig, _PALU_MISTRAL_TOKENIZER)
