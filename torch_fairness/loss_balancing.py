
"""
These are all complicated and will take time to setup, so putting on hold for now.
"""


class ReinforcedTaskWeighting:
    """
    RMTL
    https://github.com/chao1224/Loss-Balanced-Task-Weighting

    References
    ----------

        [1] Liu, S. (2018). Exploration on deep drug discovery: Representation and learning. University of
            Wisconsin-Madison, Master’s Thesis TR1854, 2018.
    """


class LossBalancedTaskWeighting:
    """
    LBTW
    https://github.com/chao1224/Loss-Balanced-Task-Weighting/blob/master/src/pcba_run.py

    References
    ----------
        [1] Shengchao Liu, Yingyu Liang, Anthony Gitter. Loss-Balanced Task Weighting to Reduce Negative Transfer in
            Multi-Task Learning . AAAI-SA (2019).
    """


class GradNorm:
    """
    Best way to test it is working would be to:
    - simulate data for two continuous objectives, y1 and y2, and train model on them.
    - Change scale for one, y1=y1*3, then retrain w/o gradnorm and with gradnorm
    - Performance w/ gradnrom and scaled should be the same as the unscaled if it works properly
    """
    ...


class MOOMTL:
    ...


class ParetoMTL:
    ...


