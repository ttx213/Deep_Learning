'''
    code by TaeHwan Jung(@graykode)
    Original Paper and repository here : https://github.com/openai/gpt-2
    GPT2 Pytorch Model : https://github.com/huggingface/pytorch-pretrained-BERT
'''
import logging

logger = logging.getLogger(__name__)

def load_weight(model, state_dict, model_type):
    old_keys = []
    new_keys = []
    for key in state_dict.keys():
        new_key = None
        if "gamma" in key:
            if new_key == None:
                new_key = key.replace("gamma", "weight")
            else:
                new_key = new_key.replace("gamma", "weight")
        if "beta" in key:
            if new_key == None:
                new_key = key.replace("beta", "bias")
            else:
                new_key = new_key.replace("beta", "bias")
        if "bert.encoder.layer" in key:
            if new_key == None:
                new_key = key.replace("bert.encoder.layer", "g_layers")
            else:
                new_key = new_key.replace("bert.encoder.layer", "g_layers")
        if "attention" in key:
            if new_key == None:
                new_key = key.replace("attention", "self_attn")
            else:
                new_key = new_key.replace("attention", "self_attn")
        if new_key:
            old_keys.append(key)
            new_keys.append(new_key)
    
    for old_key, new_key in zip(old_keys, new_keys):
        state_dict[new_key] = state_dict.pop(old_key)
    

    
    expected_keys = list(model.state_dict().keys())
    loaded_keys = list(state_dict.keys())


    missing_keys = []
    unexpected_keys = []
    error_msgs = []
    # copy state_dict so _load_from_state_dict can modify it
    metadata = getattr(state_dict, "_metadata", None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    def load(module, prefix=""):
        local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})

        module._load_from_state_dict(
            state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs
        )
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + ".")

    start_model = model
    if hasattr(model, "transformer") and all(not s.startswith('transformer.') for s in state_dict.keys()):
        start_model = model.transformer
    
    expected_keys = list(start_model.state_dict().keys())
    loaded_keys = list(state_dict.keys())

    missing_keys = list(set(expected_keys) - set(loaded_keys))
    unexpected_keys = list(set(loaded_keys) - set(expected_keys))

    load(start_model, prefix="")

    # Make sure we are still sharing the output and input embeddings after loading weights
    if model_type == 'gpt2':
        model.set_tied()
    return model