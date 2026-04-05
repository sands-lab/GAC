#llama
from .svd_llama import (
    PaluLlamaConfig,
    PaluLlamaForCausalLM
)

#mistral
from .svd_mistral import (
    PaluMistralConfig,
    PaluMistralForCausalLM
)

#qwen
from .svd_qwen import (
    PaluQwen2Config,
    PaluQwen2ForCausalLM
)

#modules
from .modules import (
    HeadwiseLowRankModule
)

_MODEL_FAMILIES = {
    'llama': {
        'config': PaluLlamaConfig,
        'ModelForCausalLM': PaluLlamaForCausalLM,
        'aliases': ('llama', 'palullama'),
    },
    'mistral': {
        'config': PaluMistralConfig,
        'ModelForCausalLM': PaluMistralForCausalLM,
        'aliases': ('mistral', 'palumistral'),
    },
    'qwen2': {
        'config': PaluQwen2Config,
        'ModelForCausalLM': PaluQwen2ForCausalLM,
        'aliases': ('qwen2', 'paluqwen2'),
    },
}

AVAILABLE_MODELS = {}
for family in _MODEL_FAMILIES.values():
    entry = {
        'config': family['config'],
        'ModelForCausalLM': family['ModelForCausalLM'],
    }
    for alias in family['aliases']:
        AVAILABLE_MODELS[alias] = entry
