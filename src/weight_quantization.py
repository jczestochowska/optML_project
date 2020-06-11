import torch


def quantize_float16(model_dict):
    '''
    This function performs the 16-bit quantization  

    :param model_dict: Model's state dict with default 32-bit float parameters
    :return: model_dict = Model's state dict with 16-bit float parameters
    '''

    for name, param in model_dict.items():
        model_dict[name] = param.half()
    return model_dict


def quantize_int8(model_dict):
    '''
    This function performs the 8-bit quantization  

    :param model_dict: Model's state dict with default 32-bit float parameters
    :return: model_dict = Model's state dict with 8-bit int parameters
    '''

    # Find maximum parameter
    max_param = 0
    for name, param in model_dict.items():
        new_max = param.abs().max()
        if new_max > max_param:
            max_param = new_max
    # Scale the maximum value to the max of an int8
    multiplier = 127 / max_param
    for name, param in model_dict.items():
        model_dict[name] = (param * multiplier).to(torch.int8)
    return model_dict, multiplier


def decode_quantized_model_int8(model_dict, multiplier):
    '''
    This function decodes model's weights back to float32 from int8

    :param model_dict: Model's state dict with quantized parameters
    :param multiplier: Multiplier number needed to decode
                       quantized weights back to float32
    :return:
    '''
    for name, param in model_dict.items():
        model_dict[name] = param.to(torch.float32) / multiplier
    return model_dict


def no_quantization(model_dict):
    '''
    This function performs no quantization and just returns the model's state dicts

    :param model_dict: Model's state dict with default 32-bit float parameters
    :return: model_dict = Model's state dict with default 32-bit float parameters
    '''

    return model_dict
