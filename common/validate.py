from copy import deepcopy
from common.equivalence import equivalence
from common.sampling import sample_generation_main_old_1
from common.system import build_canonicalOTA


def validate(learned_system, system, upper_guard):
    new_system = build_canonicalOTA(deepcopy(system))
    # 比较是否等价
    correct_flag, ctx = equivalence(learned_system, new_system, upper_guard)

    # 测试通过率
    if correct_flag:
        passingRate = 1
    else:
        failNum = 0
        testNum = 20000
        for i in range(testNum):
            sample = sample_generation_main_old_1(learned_system.actions, upper_guard, len(learned_system.states))
            system_DRTWs, real_outputs = new_system.test_DTWs(sample)
            hypothesis_DRTWs, outputs = learned_system.test_DTWs(sample)
            if real_outputs != outputs:
                failNum += 1
        passingRate = (testNum - failNum) / testNum
    return correct_flag, passingRate
