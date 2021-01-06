from common.equivalence import equivalence


def model_compare(hypothesis_pre, hypothesis_now, upper_guard, system):
    # Do not compare for the first time
    if hypothesis_pre is None:
        return True, []

    eq_flag, ctx = equivalence(hypothesis_now, hypothesis_pre, upper_guard)  # ctx is DTWs
    if eq_flag:
        raise Exception('eq_flag must be false!')
    flag = True
    DRTWs_real, outputs_real = system.test_DTWs(ctx)
    DRTWs_now, outputs_now = hypothesis_now.test_DTWs(ctx)
    # if (value_real == 1 and value_now != 1) or (value_real != 1 and value_now == 1):
    if outputs_real != outputs_now:
        flag = False
    return flag, ctx
