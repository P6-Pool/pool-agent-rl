

ConstantWeight = 1
NegativeConstantWeight = -ConstantWeight


def ConstantWeightMaxSteps(num_balls: int, current_step: int, max_steps: int | None) -> float:
    assert max_steps is not None, "Max steps must be defined for ConstantWeightMaxSteps"
    return ConstantWeight / max_steps


def NegativeConstantWeightMaxSteps(num_balls: int, current_step: int, max_steps: int | None) -> float:
    assert max_steps is not None, "Max steps must be defined for NegativeConstantWeightMaxSteps"
    return NegativeConstantWeight / max_steps


def ConstantWeightNumBalls(num_balls: int, current_step: int, max_steps: int | None) -> float:
    return ConstantWeight / num_balls


def NegativeConstantWeightNumBalls(num_balls: int, current_step: int, max_steps: int | None) -> float:
    return NegativeConstantWeight / num_balls


def ConstantWeightBalls(num_balls: int, current_step: int, max_steps: int | None) -> float:
    return ConstantWeight / (num_balls - 1)


def NegativeConstantWeightBalls(num_balls: int, current_step: int, max_steps: int | None) -> float:
    return NegativeConstantWeight / (num_balls - 1)


def ConstantWeightCurrentStep(num_balls: int, current_step: int, max_steps: int | None) -> float:
    return ConstantWeight / current_step


def NegativeConstantWeightCurrentStep(num_balls: int, current_step: int, max_steps: int | None) -> float:
    return NegativeConstantWeight / current_step
