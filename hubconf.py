import os
import sys
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    AutoModelForQuestionAnswering,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    add_start_docstrings,
)

# Set up the path for importing modules
SRC_DIR = os.path.join(os.path.dirname(__file__), "src")
sys.path.append(SRC_DIR)

# List of dependencies
dependencies = [
    "torch", "numpy", "tokenizers", "filelock", "requests", "tqdm", 
    "regex", "sentencepiece", "sacremoses", "importlib_metadata", 
    "huggingface_hub"
]

@add_start_docstrings(AutoConfig.__doc__)
def config(*args, **kwargs):
    """
    # Using torch.hub
    import torch

    config = torch.hub.load('huggingface/transformers', 'config', 'google-bert/bert-base-uncased')
    config = torch.hub.load('huggingface/transformers', 'config', './test/bert_saved_model/')
    config = torch.hub.load('huggingface/transformers', 'config', './test/bert_saved_model/my_configuration.json')
    config = torch.hub.load('huggingface/transformers', 'config', 'google-bert/bert-base-uncased', output_attentions=True, foo=False)
    assert config.output_attentions == True
    config, unused_kwargs = torch.hub.load('huggingface/transformers', 'config', 'google-bert/bert-base-uncased', output_attentions=True, foo=False, return_unused_kwargs=True)
    assert config.output_attentions == True
    assert unused_kwargs == {'foo': False}
    
    # New testing methodology: Testing on different datasets
    # Example usage: config('model-name', additional_param='value')
    """
    return AutoConfig.from_pretrained(*args, **kwargs)


@add_start_docstrings(AutoTokenizer.__doc__)
def tokenizer(*args, **kwargs):
    """
    # Using torch.hub
    import torch

    tokenizer = torch.hub.load('huggingface/transformers', 'tokenizer', 'google-bert/bert-base-uncased')
    tokenizer = torch.hub.load('huggingface/transformers', 'tokenizer', './test/bert_saved_model/')
    
    # New testing methodology: Parameter experiments
    # Example usage: tokenizer('model-name', special_tokens=True)
    """
    return AutoTokenizer.from_pretrained(*args, **kwargs)


@add_start_docstrings(AutoModel.__doc__)
def model(*args, **kwargs):
    """
    # Using torch.hub
    import torch

    model = torch.hub.load('huggingface/transformers', 'model', 'google-bert/bert-base-uncased')
    model = torch.hub.load('huggingface/transformers', 'model', './test/bert_model/')
    model = torch.hub.load('huggingface/transformers', 'model', 'google-bert/bert-base-uncased', output_attentions=True)
    assert model.config.output_attentions == True

    # Loading from a TF checkpoint file
    config = AutoConfig.from_pretrained('./tf_model/bert_tf_model_config.json')
    model = torch.hub.load('huggingface/transformers', 'model', './tf_model/bert_tf_checkpoint.ckpt.index', from_tf=True, config=config)

    # New testing methodology: Additional models
    # Example usage: model('model-name', additional_model='MLP')
    """
    return AutoModel.from_pretrained(*args, **kwargs)


@add_start_docstrings(AutoModelForCausalLM.__doc__)
def modelForCausalLM(*args, **kwargs):
    """
    # Using torch.hub
    import torch

    model = torch.hub.load('huggingface/transformers', 'modelForCausalLM', 'openai-community/gpt2')
    model = torch.hub.load('huggingface/transformers', 'modelForCausalLM', './test/saved_model/')
    model = torch.hub.load('huggingface/transformers', 'modelForCausalLM', 'openai-community/gpt2', output_attentions=True)
    assert model.config.output_attentions == True

    # Loading from a TF checkpoint file
    config = AutoConfig.from_pretrained('./tf_model/gpt_tf_model_config.json')
    model = torch.hub.load('huggingface/transformers', 'modelForCausalLM', './tf_model/gpt_tf_checkpoint.ckpt.index', from_tf=True, config=config)

    # New testing methodology: Testing on different datasets
    # Example usage: modelForCausalLM('model-name', new_param='value')
    """
    return AutoModelForCausalLM.from_pretrained(*args, **kwargs)


@add_start_docstrings(AutoModelForMaskedLM.__doc__)
def modelForMaskedLM(*args, **kwargs):
    """
    # Using torch.hub
    import torch

    model = torch.hub.load('huggingface/transformers', 'modelForMaskedLM', 'google-bert/bert-base-uncased')
    model = torch.hub.load('huggingface/transformers', 'modelForMaskedLM', './test/bert_model/')
    model = torch.hub.load('huggingface/transformers', 'modelForMaskedLM', 'google-bert/bert-base-uncased', output_attentions=True)
    assert model.config.output_attentions == True

    # Loading from a TF checkpoint file
    config = AutoConfig.from_pretrained('./tf_model/bert_tf_model_config.json')
    model = torch.hub.load('huggingface/transformers', 'modelForMaskedLM', './tf_model/bert_tf_checkpoint.ckpt.index', from_tf=True, config=config)

    # New testing methodology: Parameter experiments
    # Example usage: modelForMaskedLM('model-name', parameter_experiment='value')
    """
    return AutoModelForMaskedLM.from_pretrained(*args, **kwargs)


@add_start_docstrings(AutoModelForSequenceClassification.__doc__)
def modelForSequenceClassification(*args, **kwargs):
    """
    # Using torch.hub
    import torch

    model = torch.hub.load('huggingface/transformers', 'modelForSequenceClassification', 'google-bert/bert-base-uncased')
    model = torch.hub.load('huggingface/transformers', 'modelForSequenceClassification', './test/bert_model/')
    model = torch.hub.load('huggingface/transformers', 'modelForSequenceClassification', 'google-bert/bert-base-uncased', output_attentions=True)
    assert model.config.output_attentions == True

    # Loading from a TF checkpoint file
    config = AutoConfig.from_pretrained('./tf_model/bert_tf_model_config.json')
    model = torch.hub.load('huggingface/transformers', 'modelForSequenceClassification', './tf_model/bert_tf_checkpoint.ckpt.index', from_tf=True, config=config)

    # New testing methodology: Additional models
    # Example usage: modelForSequenceClassification('model-name', additional_model='MLP')
    """
    return AutoModelForSequenceClassification.from_pretrained(*args, **kwargs)


@add_start_docstrings(AutoModelForQuestionAnswering.__doc__)
def modelForQuestionAnswering(*args, **kwargs):
    """
    # Using torch.hub
    import torch

    model = torch.hub.load('huggingface/transformers', 'modelForQuestionAnswering', 'google-bert/bert-base-uncased')
    model = torch.hub.load('huggingface/transformers', 'modelForQuestionAnswering', './test/bert_model/')
    model = torch.hub.load('huggingface/transformers', 'modelForQuestionAnswering', 'google-bert/bert-base-uncased', output_attentions=True)
    assert model.config.output_attentions == True

    # Loading from a TF checkpoint file
    config = AutoConfig.from_pretrained('./tf_model/bert_tf_model_config.json')
    model = torch.hub.load('huggingface/transformers', 'modelForQuestionAnswering', './tf_model/bert_tf_checkpoint.ckpt.index', from_tf=True, config=config)

    # New testing methodology: Testing on different datasets
    # Example usage: modelForQuestionAnswering('model-name', new_param='value')
    """
    return AutoModelForQuestionAnswering.from_pretrained(*args, **kwargs)
