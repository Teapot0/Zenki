import pandas as pd
import numpy as np


def cha_cha(x1, x2, direction=False):
    diff = (x1 - x2) > 0
    jincha = ((diff - diff.shift(1)) > 0) * 1
    sicha = ((diff - diff.shift(1)) < 0) * -1
    if direction == 'jincha':
        return jincha
    if direction == 'sicha':
        return sicha
    else:
        return jincha + sicha  # 金叉是1，死叉-1，其他情况是0









