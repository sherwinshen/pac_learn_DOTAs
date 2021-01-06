import queue
import copy
from common.hypothesis import struct_discreteOTA, struct_hypothesisOTA
from normal_learning.teacher import EQs_level
from normal_learning.obsTable import init_table_normal, make_closed, make_consistent, deal_ctx
from normal_learning.pac import minimize_counterexample_normal
from normal_learning.debug.check_algorithm import check_guess_table_correct


def learnOTA_normal_level(system, actions, upper_guard, epsilon, delta, state_num, debug_flag):
    level = 0
    cur_level_neel_to_explore = queue.PriorityQueue()
    next_level_need_to_explore = queue.PriorityQueue()
    cur_level_hypothesis = queue.PriorityQueue()

    # init table
    for table in init_table_normal(actions, system):
        cur_level_neel_to_explore.put((table.table_id, table))

    prev_ctx = []  # List of existing counterexamples
    t_number = 0  # Current number of tables
    target = None
    learn_flag = False
    m_i = 0

    while not learn_flag:
        if cur_level_neel_to_explore.qsize() == 0:
            break

        while cur_level_neel_to_explore.qsize() > 0:
            flag = False
            depth, current_table = cur_level_neel_to_explore.get()
            t_number = t_number + 1

            if check_guess_table_correct(current_table, system):
                flag = True
                print('***************** Now is Correct *****************')

            print("Table %s: %s has parent-%s by %s" % (t_number, current_table.table_id, current_table.parent, current_table.reason))
            if debug_flag:
                current_table.show()
                print("--------------------------------------------------")

            # First check if the table is closed
            flag_closed, closed_move = current_table.is_closed()
            if not flag_closed:
                if debug_flag:
                    print("------------------make closed--------------------------")
                temp_tables = make_closed(closed_move, actions, current_table, system)
                temp_flag = False
                count = []
                if flag and len(temp_tables) == 0:
                    raise Exception('Attention-1.1!!!')
                if len(temp_tables) > 0:
                    for table in temp_tables:
                        if flag and check_guess_table_correct(table, system):
                            count.append(table)
                            temp_flag = True
                        cur_level_neel_to_explore.put((table.table_id, table))
                if len(count) > 1:
                    raise Exception('Attention-1.2!!!')
                if flag and not temp_flag:
                    raise Exception('Attention-1.3!!!')
                continue

            # If is closed, check if the table is consistent
            flag_consistent, prefix_LTWs, e_index, reset_i, reset_j, index_i, index_j = current_table.is_consistent()
            if not flag_consistent:
                if debug_flag:
                    print("------------------make consistent--------------------------")
                temp_tables = make_consistent(prefix_LTWs, e_index, reset_i, reset_j, index_i, index_j, current_table, system)
                temp_flag = False
                count = []
                if flag and len(temp_tables) == 0:
                    raise Exception('Attention-2.1!!!')
                if len(temp_tables) > 0:
                    for table in temp_tables:
                        if flag and check_guess_table_correct(table, system):
                            count.append(table)
                            temp_flag = True
                        cur_level_neel_to_explore.put((table.table_id, table))
                if len(count) > 1:
                    raise Exception('Attention-2.2!!!')
                if flag and not temp_flag:
                    raise Exception('Attention-2.3!!!')
                continue

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

            cur_level_hypothesis.put((len(hypothesisOTA.states), hypothesisOTA, current_table))

        if cur_level_hypothesis.qsize() == 0:
            break

        print('———————————————— 当前level完成 ————————————————')

        m_i = cur_level_hypothesis.qsize()
        while cur_level_hypothesis.qsize() > 0:
            flag = False
            hyp_state_num, cur_hypothesis, current_table = cur_level_hypothesis.get()

            if check_guess_table_correct(current_table, system):
                flag = True
                print('***************** Now hypothesis is Correct *****************')
                print("%s has parent-%s by %s" % (current_table.table_id, current_table.parent, current_table.reason))

            equivalent, ctx = True, None
            if prev_ctx is not None:
                for ctx in prev_ctx:
                    real_outputs = system.test_DTWs_normal(ctx, True)
                    outputs = cur_hypothesis.test_DTWs_normal(ctx)
                    if real_outputs != outputs:
                        equivalent = False
                        ctx = minimize_counterexample_normal(cur_hypothesis, system, ctx)
                        break

            if equivalent:
                system.eq_num += 1
                equivalent, ctx = EQs_level(cur_hypothesis, upper_guard, epsilon, delta, level, m_i, state_num, system)

            if not equivalent:
                # show ctx
                if debug_flag:
                    print("***************** counterexample is as follow. *******************")
                    print([dtw.show() for dtw in ctx])
                # deal with ctx
                if ctx not in prev_ctx:
                    prev_ctx.append(ctx)
                temp_tables = deal_ctx(current_table, ctx, system)
                temp_flag = False
                count = []
                if flag and len(temp_tables) == 0:
                    raise Exception('Attention-3.1!!!')
                if len(temp_tables) > 0:
                    for table in temp_tables:
                        if flag and check_guess_table_correct(table, system):
                            count.append(table)
                            temp_flag = True
                        next_level_need_to_explore.put((table.table_id, table))
                if len(count) > 1:
                    raise Exception('Attention-3.2!!!')
                if flag and not temp_flag:
                    raise Exception('Attention-3.3!!!')
            else:
                learn_flag = True
                target = copy.deepcopy(cur_hypothesis)
                break
        level += 1
        cur_level_neel_to_explore = next_level_need_to_explore
        next_level_need_to_explore = queue.PriorityQueue()
    if target is not None:
        target = target.build_simple_hypothesis()
    return target, system.mq_num, system.eq_num, system.test_num, system.test_num_cache, system.action_num, t_number, level, m_i
