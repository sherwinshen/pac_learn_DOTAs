import queue
import copy
from common.hypothesis import struct_discreteOTA, struct_hypothesisOTA
from normal_learning.teacher import EQs
from normal_learning.obsTable import init_table_normal, make_closed, make_consistent, deal_ctx
from normal_learning.pac import minimize_counterexample_normal


def learnOTA_normal(system, actions, upper_guard, epsilon, delta, state_num, debug_flag):
    need_to_explore = queue.PriorityQueue()

    # init table
    for table in init_table_normal(actions, system):
        # table.show()
        need_to_explore.put((table.table_id, table))  # 优先级为表格的有效长度

    # List of existing counterexamples
    prev_ctx = []

    # Current number of tables
    t_number = 0
    target = None

    while True:
        if need_to_explore.qsize() == 0:
            break
        depth, current_table = need_to_explore.get()
        t_number = t_number + 1

        print("Table %s: current %s has parent-%s by %s" % (t_number, current_table.table_id, current_table.parent, current_table.reason))
        if debug_flag:
            current_table.show()
            print("--------------------------------------------------")

        # First check if the table is closed
        flag_closed, closed_move = current_table.is_closed()
        if not flag_closed:
            if debug_flag:
                print("------------------make closed--------------------------")
            temp_tables = make_closed(closed_move, actions, current_table, system)
            if len(temp_tables) > 0:
                for table in temp_tables:
                    # table.show()
                    need_to_explore.put((table.table_id, table))
            continue

        # If is closed, check if the table is consistent
        flag_consistent, prefix_LTWs, e_index, reset_i, reset_j, index_i, index_j = current_table.is_consistent()
        if not flag_consistent:
            if debug_flag:
                print("------------------make consistent--------------------------")
            temp_tables = make_consistent(prefix_LTWs, e_index, reset_i, reset_j, index_i, index_j, current_table, system)
            if len(temp_tables) > 0:
                for table in temp_tables:
                    # table.show()
                    need_to_explore.put((table.table_id, table))
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

        equivalent, ctx = True, None
        if prev_ctx is not None:
            for ctx in prev_ctx:
                real_value = system.test_DTWs_normal(ctx)
                value = hypothesisOTA.test_DTWs_normal(ctx)
                if real_value != value:
                    equivalent = False
                    ctx = minimize_counterexample_normal(hypothesisOTA, system, ctx)
                    break

        if equivalent:
            system.eq_num += 1
            equivalent, ctx = EQs(hypothesisOTA, upper_guard, epsilon, delta, system.eq_num, state_num, system)

        if not equivalent:
            # show ctx
            if debug_flag:
                print("***************** counterexample is as follow. *******************")
                print([dtw.show() for dtw in ctx])
            # deal with ctx
            if ctx not in prev_ctx:
                prev_ctx.append(ctx)
            temp_tables = deal_ctx(current_table, ctx, system)
            if len(temp_tables) > 0:
                for table in temp_tables:
                    # table.show()
                    need_to_explore.put((table.table_id, table))
        else:
            target = copy.deepcopy(hypothesisOTA)
            break
    if target is not None:
        target = target.build_simple_hypothesis()

    return target, system.mq_num, system.eq_num, system.test_num, t_number
