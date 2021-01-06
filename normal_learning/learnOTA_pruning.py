import random
import copy
from normal_learning.teacher import EQs
from common.hypothesis import struct_discreteOTA, struct_hypothesisOTA
from normal_learning.obsTable import init_table_normal, make_closed, make_consistent, deal_ctx
from normal_learning.pac import minimize_counterexample_normal


def learnOTA_normal_pruning(system, actions, upper_guard, epsilon, delta, state_num, debug_flag):
    t_number = 0  # Current number of tables
    target = None
    prev_ctx = []  # List of existing counterexamples

    width = 10  # 每一层的观察表数
    expand_factor = 5  # 观察表扩展的保留数量
    level = 0  # 轮数

    eq_flag = False

    init_tables = init_table_normal(actions, system)
    if len(init_tables) <= width:
        need_to_explore = init_tables
    else:
        need_to_explore = random.sample(init_tables, width)

    while len(need_to_explore) != 0:
        level += 1

        next_to_explore = []
        for i in range(min(len(need_to_explore), width)):
            for j in range(expand_factor):
                if level == 1:
                    current_table, ctx = need_to_explore[i], None
                else:
                    current_table, ctx = need_to_explore[i]
                current_table = copy.deepcopy(current_table)
                if debug_flag:
                    print('———————————————— cur_table ————————————————')
                    current_table.show()
                if ctx is not None:
                    temp_tables = deal_ctx(current_table, ctx, system)
                    if len(temp_tables) > 0:
                        current_table = random.sample(temp_tables, 1)[0]
                    else:
                        continue
                if debug_flag:
                    print('———————————————— cur_table after deal ctx ————————————————')
                    current_table.show()

                flag, current_table = random_steps(current_table, actions, prev_ctx, system, debug_flag)
                if debug_flag:
                    print('———————————————— cur_table after one step ————————————————')
                    current_table.show()

                if not flag:
                    continue
                else:
                    t_number += 1
                    # If prepared, check conversion to FA
                    discreteOTA = struct_discreteOTA(current_table, actions)
                    if discreteOTA is None:
                        continue
                    if debug_flag:
                        print("***************** discreteOTA_" + str(system.eq_num + 1) + " is as follow. *******************")
                        discreteOTA.show_discreteOTA()

                    # Convert FA to OTA
                    hypothesisOTA = struct_hypothesisOTA(discreteOTA)
                    if debug_flag:
                        print("***************** Hypothesis_" + str(system.eq_num + 1) + " is as follow. *******************")
                        hypothesisOTA.show_OTA()

                    system.eq_num += 1
                    equivalent, ctx = EQs(hypothesisOTA, upper_guard, epsilon, delta, level, state_num, system)

                    if not equivalent:
                        if ctx not in prev_ctx:
                            prev_ctx.append(ctx)
                        if debug_flag:
                            print("***************** counterexample is as follow. *******************")
                            print([dtw.show() for dtw in ctx])
                        next_to_explore.append((current_table, ctx))
                    else:
                        eq_flag = True
                        target = copy.deepcopy(hypothesisOTA)
                        break
            if eq_flag:
                break

        if eq_flag:
            break
        else:
            if len(next_to_explore) <= width:
                need_to_explore = next_to_explore
            else:
                need_to_explore = random.sample(next_to_explore, width)

    if target is not None:
        target = target.build_simple_hypothesis()
    return target, system.mq_num, system.eq_num, system.test_num, system.test_num_cache, system.action_num, t_number


# Find a random successor of the current table. Here a successor means a prepared table that agrees with teacher on all existing counterexamples.
def random_steps(current_table, actions, prev_ctx, system, debug_flag):
    while True:
        # First check if the table is closed
        flag_closed, closed_move = current_table.is_closed()
        if not flag_closed:
            if debug_flag:
                print("------------------make closed--------------------------")
            temp_tables = make_closed(closed_move, actions, current_table, system)
            if len(temp_tables) > 0:
                current_table = random.sample(temp_tables, 1)[0]
                continue
            else:
                return False, None

        # If is closed, check if the table is consistent
        flag_consistent, prefix_LTWs, e_index, reset_i, reset_j, index_i, index_j = current_table.is_consistent()
        if not flag_consistent:
            if debug_flag:
                print("------------------make consistent--------------------------")
            temp_tables = make_consistent(prefix_LTWs, e_index, reset_i, reset_j, index_i, index_j, current_table, system)
            if len(temp_tables) > 0:
                current_table = random.sample(temp_tables, 1)[0]
                continue
            else:
                return False, None

        # If prepared, check conversion to FA
        discreteOTA = struct_discreteOTA(current_table, actions)
        if discreteOTA is None:
            return False, None
        if debug_flag:
            print("***************** discreteOTA_" + str(system.eq_num + 1) + " is as follow. *******************")
            discreteOTA.show_discreteOTA()

        # Convert FA to OTA
        hypothesisOTA = struct_hypothesisOTA(discreteOTA)
        if debug_flag:
            print("***************** Hypothesis_" + str(system.eq_num + 1) + " is as follow. *******************")
            hypothesisOTA.show_OTA()

        equivalent, ctx = True, None
        if prev_ctx is not None:
            for ctx in prev_ctx:
                real_outputs = system.test_DTWs_normal(ctx)
                outputs = hypothesisOTA.test_DTWs_normal(ctx)
                if real_outputs != outputs:
                    equivalent = False
                    ctx = minimize_counterexample_normal(hypothesisOTA, system, ctx)
                    break

        if not equivalent:
            if debug_flag:
                print("***************** counterexample is as follow. *******************")
                print([dtw.show() for dtw in ctx])
            temp_tables = deal_ctx(current_table, ctx, system)
            if len(temp_tables) > 0:
                current_table = random.sample(temp_tables, 1)[0]
                continue
            else:
                return False, None
        else:
            return True, current_table
