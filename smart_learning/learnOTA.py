import copy
import smart_learning.obsTable as obsTable
from common.hypothesis import struct_discreteOTA, struct_hypothesisOTA
from smart_learning.teacher import EQs
from smart_learning.comparator import model_compare


def learnOTA_smart(system, actions, upper_guard, epsilon, delta, state_num, comparator_flag=True, debug_flag=False):
    ### init Table
    table = obsTable.initTable(actions, system)
    if debug_flag:
        print("***************** init-Table_1 is as follow *******************")
        table.show()

    ### learning start
    equivalent = False
    learned_system = None  # learned model
    table_num = 1  # number of table

    while not equivalent:
        ### make table prepared
        prepared = table.is_prepared()
        while not prepared:
            # make closed
            closed_flag, close_move = table.is_closed()
            if not closed_flag:
                table = obsTable.make_closed(table, actions, close_move, system)
                table_num = table_num + 1
                if debug_flag:
                    print("***************** closed-Table_" + str(table_num) + " is as follow *******************")
                    table.show()

            # make consistent
            consistent_flag, consistent_add = table.is_consistent()
            if not consistent_flag:
                consistent_flag, consistent_add = table.is_consistent()
                table = obsTable.make_consistent(table, consistent_add, system)
                table_num = table_num + 1
                if debug_flag:
                    print("***************** consistent-Table_" + str(table_num) + " is as follow *******************")
                    table.show()
            prepared = table.is_prepared()

        ### build hypothesis
        # Discrete OTA
        discreteOTA = struct_discreteOTA(table, actions)
        if discreteOTA is None:
            raise Exception('Attention!!!')
        if debug_flag:
            print("***************** discreteOTA_" + str(system.eq_num + 1) + " is as follow. *******************")
            discreteOTA.show_discreteOTA()
        # Hypothesis OTA
        hypothesisOTA = struct_hypothesisOTA(discreteOTA)
        if debug_flag:
            print("***************** Hypothesis_" + str(system.eq_num + 1) + " is as follow. *******************")
            hypothesisOTA.show_OTA()

        ### comparator + EQs
        if comparator_flag:
            ctx_flag, ctx = model_compare(learned_system, hypothesisOTA, upper_guard, system)
            if ctx_flag:
                ### EQs
                equivalent, ctx = EQs(hypothesisOTA, upper_guard, epsilon, delta, state_num, system)
                learned_system = copy.deepcopy(hypothesisOTA)
            else:
                if debug_flag:
                    print("Comparator found a counterexample!!!")
                equivalent = False
        else:
            # without comparator
            learned_system = copy.deepcopy(hypothesisOTA)
            ### EQs
            equivalent, ctx = EQs(hypothesisOTA, upper_guard, epsilon, delta, state_num, system)

        if not equivalent:
            # show ctx
            if debug_flag:
                print("***************** counterexample is as follow. *******************")
                print([dtw.show() for dtw in ctx])
            # deal with ctx
            table = obsTable.deal_ctx(table, ctx, system)
            table_num = table_num + 1
            if debug_flag:
                print("***************** New-Table" + str(table_num) + " is as follow *******************")
                table.show()

    return learned_system.build_simple_hypothesis(), system.mq_num, system.eq_num, system.test_num, system.test_num_cache, system.action_num, table_num
