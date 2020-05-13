def quantize_float16(model_dict):
    """
    :param model: Model's state dict with default 32-bit float parameters
    :return: model's state dict with 16-bit float parameters
    """
    for name, param in model_dict.items():
        model_dict[name] = param.half()
    return model_dict


def quantize_int8(model_dict):
    for name, param in model_dict.items():
        pass


def no_quantization(model_dict):
    return model_dict
