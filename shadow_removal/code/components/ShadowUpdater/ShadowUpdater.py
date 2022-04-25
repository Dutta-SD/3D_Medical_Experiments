def update_shadow_params(
    reward: float,
    prev_params: dict,
    a_0: float = 0.80,
    b_0: float = 0.01,
) -> dict:

    """Expected reward to be within [0, 1]"""

    def updater_fxn(value, reward):
        new_val = (a_0 * reward + b_0) * value
        return int(new_val) if type(value) == int else new_val

    return {key: updater_fxn(value, reward) for key, value in prev_params.items()}
