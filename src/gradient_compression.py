

def sparsify_gradient_topk(model):
    for param in model.parameters():
        param.grad *= 2


def sparsify_gradient_randomly(model):
    # for param in model.parameters():
    #     param.grad *= C
    pass


def quantize_gradient1(model):
    # for param in model.parameters():
    #     param.grad *= C
    pass


def quantize_gradient2(model):
    # for param in model.parameters():
    #     param.grad *= C
    pass
