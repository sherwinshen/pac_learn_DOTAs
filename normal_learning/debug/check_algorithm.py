from common.system import build_canonicalOTA
from common.TimedWord import ResetTimedWord, TimedWord
import copy


# 检查当前观察表是否是正确的
def check_guess_table_correct(table, system):
    system = build_canonicalOTA(copy.deepcopy(system))
    for s in table.S:
        flag = is_LRTWs_right(s.LRTWs, system)
        if not flag:
            return False
        for index in range(len(table.E)):
            LRTWs = s.LRTWs + combine_LRTWs_with_LTWs(table.E[index], s.suffixes_resets[index])
            flag = is_LRTWs_value_right(LRTWs, s.values[index], system)
            if not flag:
                return False
    for r in table.R:
        flag = is_LRTWs_right(r.LRTWs, system)
        if not flag:
            return False
        for index in range(len(table.E)):
            LRTWs = r.LRTWs + combine_LRTWs_with_LTWs(table.E[index], r.suffixes_resets[index])
            flag = is_LRTWs_value_right(LRTWs, r.values[index], system)
            if not flag:
                return False
    return True


# 检查当前暂存猜测观察表中是否有猜对的表存在
def check_guess(current_table, need_to_explore, system):
    system = build_canonicalOTA(copy.deepcopy(system))
    need_to_explore = need_to_explore.queue
    flag = check_guess_table_correct(current_table, system)
    if flag:
        return True
    else:
        for i in range(len(need_to_explore)):
            current_table = need_to_explore[i]
            flag = check_guess_table_correct(current_table, system)
            if flag:
                return True
    raise Exception('No Right')


def is_LRTWs_right(LRTWs, system):
    if not LRTWs:
        return True
    else:
        now_time = 0
        cur_state = system.init_state
        for lrtw in LRTWs:
            if lrtw.time < now_time:
                cur_state = system.sink_state
                if not lrtw.reset:
                    return False
            else:
                LRTW = ResetTimedWord(lrtw.action, lrtw.time, lrtw.reset)
                flag, cur_state, reset = is_passing_tran(LRTW, cur_state, system)
                if not flag:
                    return False
                if reset:
                    now_time = 0
                else:
                    now_time = lrtw.time
        return True


def is_LRTWs_value_right(LRTWs, realValue, system):
    if not LRTWs:
        if system.init_state in system.accept_states:
            value = 1
        else:
            value = 0
    else:
        now_time = 0
        cur_state = system.init_state
        for lrtw in LRTWs:
            if lrtw.time < now_time:
                cur_state = system.sink_state
                if not lrtw.reset:
                    return False
            else:
                LRTW = ResetTimedWord(lrtw.action, lrtw.time, lrtw.reset)
                flag, cur_state, reset = is_passing_tran(LRTW, cur_state, system)
                if not flag:
                    return False
                if reset:
                    now_time = 0
                else:
                    now_time = lrtw.time
        if cur_state in system.accept_states:
            value = 1
        elif cur_state == system.sink_state:
            value = -1
        else:
            value = 0
    if value != realValue:
        return False
    return True


def is_passing_tran(lrtw, cur_state, system):
    for tran in system.trans:
        if tran.source == cur_state:
            if lrtw.action == tran.action and tran.reset == lrtw.reset:
                for guard in tran.guards:
                    if guard.is_in_interval(lrtw.time):
                        return True, tran.target, tran.reset
    return False, None, None


def combine_LRTWs_with_LTWs(LTWs, reset):
    LRTWs = []
    for i in range(len(LTWs)):
        LRTWs.append(ResetTimedWord(LTWs[i].action, LTWs[i].time, reset[i]))
    return LRTWs
