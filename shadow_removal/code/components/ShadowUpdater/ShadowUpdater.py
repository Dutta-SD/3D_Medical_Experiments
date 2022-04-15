def update_shadow_params(
    reward: float,
    prev_params: dict,
    a_0: float = 0.3,
    b_0: float = 0.2,
) -> dict:

    """Expected reward to be within [0, 1]"""
    return {key: value * (a_0 * reward + b_0) for key, value in prev_params.items()}
