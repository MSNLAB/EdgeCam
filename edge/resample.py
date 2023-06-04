import random

def history_sample(index_list, score_list):
    num_elements = min(len(index_list), len(score_list)) // 4
    index_list_c = random.sample(index_list, num_elements)
    score_list_c = []
    for item in score_list:
        if list(item.keys())[0] in index_list_c:
            score_list_c.append(item)
    return index_list_c, score_list_c

def annotion_process(annotations, index_list):
    new_annotations = []
    for sublist in annotations:
        if sublist[0] in index_list:
            new_annotations.append(sublist)
    return new_annotations
