# main file
import sys
import random
import json
import time
from common.system import build_system
from common.validate import validate
from common.make_pic import make_system, make_hypothesis
from normal_learning.learnOTA import learnOTA_normal
# from normal_learning.debug.learnOTA import learnOTA_normal
# from normal_learning.learnOTA_level import learnOTA_normal_level
# from normal_learning.debug.learnOTA_level import learnOTA_normal_level
# from normal_learning.learnOTA_pruning import learnOTA_normal_pruning
from smart_learning.learnOTA import learnOTA_smart


def main():
    # get model information and build target system
    with open(model_file, 'r') as json_model:
        model = json.load(json_model)
    system = build_system(model)
    make_system(system, result_path, '/model_target')

    # get prior information required for learning
    with open(precondition_file, 'r') as json_precondition:
        information = json.load(json_precondition)
        actions = information["inputs"]
        upper_guard = information["upperGuard"]  # upper bound of guard time
        epsilon = information["epsilon"]  # accuracy
        delta = information["delta"]  # confidence
        state_num = information["stateNum"]  # not necessary

    # pac learning of DOTAs
    start_time = time.time()
    print("********** learning starting *************")
    if teacher_type == "smart_teacher":
        comparator_flag = True
        learned_system, mq_num, eq_num, test_num, test_num_cache, action_num, table_num = learnOTA_smart(system, actions, upper_guard, epsilon, delta, state_num, comparator_flag, debug_flag)
    elif teacher_type == "normal_teacher":
        learned_system, mq_num, eq_num, test_num, test_num_cache, action_num, table_num = learnOTA_normal(system, actions, upper_guard, epsilon, delta, state_num, debug_flag)
        # learned_system, mq_num, eq_num, test_num, test_num_cache, action_num, table_num, level, hpy_num = learnOTA_normal_level(system, actions, upper_guard, epsilon, delta, state_num, debug_flag)
        # learned_system, mq_num, eq_num, test_num, test_num_cache, action_num, table_num = learnOTA_normal_pruning(system, actions, upper_guard, epsilon, delta, state_num, debug_flag)
    else:
        raise Exception('Teacher type only allows two options: smart_teacher and normal_teacher.')
    end_time = time.time()

    # learning result
    if learned_system is None:
        print("Error! Learning Failed.")
        print("*********** learning ending  *************")
        return {"result": "Failed"}
    else:
        # validate
        correct_flag, passing_rate = validate(learned_system, system, upper_guard)
        make_hypothesis(learned_system, result_path, '/model_hypothesis')
        print("———————————————————————————————————————————")
        print("Succeed! The result is as follows:")
        print("Total time of learning: " + str(end_time - start_time))
        print("Total number of MQs (no-cache): " + str(mq_num))
        print("Total number of EQs (no-cache): " + str(eq_num))
        print("Total number of tests (no-cache): " + str(test_num))
        print("Total number of tests (with-cache): " + str(test_num_cache))
        print("Total number of actions: " + str(action_num))
        print("Total number of tables explored: " + str(table_num))
        print("Completely correct: " + str(correct_flag) + "   Testing pass rate: " + str(passing_rate))
        # print("level:" + str(level) + "   hypothesis number of current level: " + str(hpy_num))
        print("*********** learning ending  *************")
        trans = []
        for t in learned_system.trans:
            trans.append([str(t.tran_id), str(t.source), str(t.action), t.show_guards(), str(t.reset), str(t.target)])
        result_obj = {
            "result": "Success",
            "learningTime": end_time - start_time,
            "mqNum": mq_num,
            "eqNum": eq_num,
            "testNum": test_num,
            "testNumCache": test_num_cache,
            "actionNum": action_num,
            "tableNum": table_num,
            "correct": correct_flag,
            "passingRate": passing_rate,
            # "level": level,
            # "hpyNum_of_level": hpy_num,
            "model": {
                "actions": learned_system.actions,
                "states": learned_system.states,
                "initState": learned_system.init_state,
                "acceptStates": learned_system.accept_states,
                "sinkState": learned_system.sink_state,
                "trans": trans
            }
        }
        return result_obj


if __name__ == '__main__':
    ### used to reproduce experimental results
    random.seed(3)

    ### file directory
    # file_path = sys.argv[1]
    file_path = "benchmarks/4_2_10/4_2_10-10"
    # target model file
    model_file = file_path + "/model.json"
    # prior information required for learning
    precondition_file = file_path + "/precondition.json"

    ### teacher type - smart_teacher / normal_teacher
    # teacher_type = sys.argv[2]
    teacher_type = "smart_teacher"

    # results file directory
    result_path = 'results/' + teacher_type + '/' + file_path
    # debug mode
    debug_flag = True

    ### start running experiment
    result = main()
    with open(result_path + "/result.json", 'w') as json_file:
        json_file.write(json.dumps(result, indent=2))
