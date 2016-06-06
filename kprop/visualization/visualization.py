# -*- coding: utf-8 -*-

import kprop.visualization.terminal.util as tutil
import kprop.visualization.plot.util as putil


class HintonDiagram:
    def __init__(self, is_terminal=True):
        self.is_terminal = is_terminal

    def __call__(self, data):
        res = None
        if self.is_terminal is True:
            res = tutil.hinton_diagram(data)
        else:
            res = putil.hinton(data)
        return res


import numpy as np

if __name__ == '__main__':
    data = np.random.randn(50, 100)
    hd = HintonDiagram(is_terminal=False)
    print(hd(data))

