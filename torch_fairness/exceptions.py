

class NotFittedError(ValueError, AttributeError):
    """Exception class to raise if estimator is used before fitting - based on SKLearn.

    Examples
    --------

    """


# TODO -Make sure sufficientl differn than LabelImbalancedBatchWarning
class MissingSensitiveGroupsError(Exception):
    """Exception class to raise if an expected sensitive group is missing.

    """


class DummyCodingError(Exception):
    """Exception class to raise if incorrectly formatted dummy coded data is encountered.

    """
