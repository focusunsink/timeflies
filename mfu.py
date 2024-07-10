

TF32_COMP = 156.0 * 2 ** 40
FP32_COMP = 19.5 * 2 ** 40


def mfu_computes(b, l, s, h, V):

    ret = 72 * b * l * s * h * h

    ret = ret / (1 + s/(6 * h) + V / (12 * h * l))

    return ret 
     
def mfu_cost(F, N, T):
    return F * N * T


def mfu(comp, resource):
    return comp / resource


def case_1():
    time = 2 * 60 * 60 + 0 * 60 + 43
    b = 14416
    l = 32 
    s = 2048
    h = 2048
    V = 125824





    comp = mfu_computes(b, l, s, h, V)
    resource = mfu_cost(FP32_COMP, 8, time)


    mfu_ratio = mfu(comp, resource)


    print(mfu_ratio)




def case_2():
    time = 91 * 24 * 60 * 60
    b = 3 * 10**9 / 2048
    l = 32 
    s = 2048
    h = 2048
    V = 125824

    comp = mfu_computes(b, l, s, h, V)
    resource = mfu_cost(FP32_COMP, 8, time)

    mfu_ratio = mfu(comp, resource)

    print(mfu_ratio)


case_2()
