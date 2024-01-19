def dict_add(keys, values, out_dict):
    for key, val in zip(keys, values):
        if key in out_dict.keys():
            out_dict[key] = out_dict[key] + val
        else:
            out_dict[key] = val
    return out_dict

def lists_to_dict(keys, values):
    out_dict = {}
    return dict_add(keys, values, out_dict)

def dict_to_dict(in_dict):
    out_dict = {}
    return _dict_to_dict(in_dict, out_dict)

def _dict_to_dict(in_dict, out_dict):
    return dict_add(in_dict.keys(), in_dict.values(), out_dict)

def dicts_to_dict(dict_list):
    out_dict = {}
    for in_dict in dict_list:
        out_dict = _dict_to_dict(in_dict, out_dict)
    return out_dict
