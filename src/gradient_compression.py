

def sparsify_gradient_topk(model):
    for param in model.parameters():
        with torch.no_grad():


def sparsify_gradient_randomly(model):
    return 1


def quantize_gradient_float16(model):
    """Input: Model with default 32-bit float parameters
    Return: model with 16-bit float parameters
            number of saved bytes due to quantization"""
    saved_bytes = 0
    with torch.no_grad():
        for param in model.parameters():
            saved_bytes += 2*p.nelements()
    return model.half(), saved_bytes

def quantize_gradient_float32(model):
    return model.float()


def quantize_gradient2(model):
    return 1
