def soh_cost(replacement_cost: float, delta_soh: float, soh_limit: float) -> float:
    """
        Compute the cost associated to the variation of the state of health of the battery.

        Parameters:
        ----------------
        replacement_cost (float): The replacement cost of the battery.
        delta_soh (float): The variation in SoH of the battery.
        soh_limit (float): The end of life of the battery in SoH percentage.

        Reference: https://github.com/OscarPindaro/RLithium-0/tree/main
        """
    assert 0 <= delta_soh < 1, "Reward error: 'delta_soh' be within [0, 1], instead of {}".format(delta_soh)
    assert 0 <= soh_limit < 1, "Reward error: 'soh_limit' should be be within [0, 1], instead of {}".format(soh_limit)
    assert replacement_cost >= 0, "Reward error: battery replacement cost should be non-negative"

    return delta_soh * replacement_cost / (1 - soh_limit)
