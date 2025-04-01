def operational_cost(replacement_cost: float,
                     C_rated: float,
                     C: float,
                     DoD_rated: float,
                     L_rated: float,
                     v_rated: float,
                     p: float,
                     r: float,
                     K_rated: float,
                     soc: float,
                     is_discharging: bool
                     ) -> float:
    """
    Compute the operational cost of using the battery depending on the replacement cost of the battery, the power used
    to charge and discharge the system and the power losses occurring within the process.

    Parameters:
    ----------------
    replacement_cost (float): The replacement cost of the battery.
    C_rated (float): The rated capacity of the battery.
    C (float): The actual capacity of the battery.
    DoD_n (float): The rated depth of discharge (DoD) of the battery.
    L_rated (float): The rated lifetime of the battery.
    v_rated (float): The rated voltage of the battery.
    p (float): The charge/discharged power of the battery.
    r (float): The internal resistance of the battery.
    K_rated (float): The polarization constant of the battery that corresponds to the R1 of the ECM.
    soc (float): The current state of charge (SoC) of the battery.
    is_discharging (bool): Whether the battery is discharging or charging.


    Reference paper:
    ----------------
    T. A. Nguyen and M. L. Crow, "Stochastic Optimization of Renewable-Based Microgrid Operation Incorporating Battery
    Operating Cost," in IEEE Transactions on Power Systems, vol. 31, no. 3, pp. 2289-2296, May 2016,
    doi: 10.1109/TPWRS.2015.2455491.
    """
    # To prevent division by zero error
    soc = 1e-6 if soc == 0. else soc

    # Coefficient c_avai = c_bat
    c_bat = replacement_cost / (C_rated * DoD_rated * (0.9 * L_rated - 0.1))

    # P_loss depending on P charged or discharged
    if is_discharging:
        p_loss = (1 * (r + K_rated / soc) / v_rated**2 * p**2 +
                  1 * C * K_rated * (1 - soc) / (soc * v_rated**2) * p)
        h_bat = abs(p) + p_loss

    else:
        h_bat = (1 * (r + K_rated / (0.9 - soc)) / v_rated**2 * p**2 +
                 1 * C * K_rated * (1 - soc) / (soc * v_rated**2) * p)

    # Dividing by 1e3 to convert because it is in €/kWh, to get the cost in €/Wh
    return c_bat * h_bat / 1e3


# c_bat = 3000 / (60 * 0.8 * (0.9 * 3000 - 0.1)) = 0.023149005518722912

# i_max = (0.8-0.2) / 1 * 60 * 3600 
# p_loss = (1**-3 * (0.19 + 0.7 / 0.2) i_max * v_max/ 350.4**2 * i_max * v_max**2 + 1**-3 * 60 * 0.7 * (1 - 0.2) / (0.2 * 350.4**2) * max_gen)
# h_bat = abs(max_gen) + p_loss
# max_discharge = h_bat * c_bat 



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


def linearized_degradation(replacement_cost: float,
                           delta_deg: float,
                           deg_limit: float
                           ) -> float:
    """
    Compute the cost associated to a linearized degradation of the battery.

    Parameters:
    ----------------
    replacement_cost (float): The replacement cost of the battery.
    delta_deg (float): The step degradation of the battery.
    deg_limit (float): The maximum degradation of the battery.

    Reference: https://github.com/OscarPindaro/RLithium-0/tree/main
    """
    assert delta_deg >= 0, "Reward error: 'delta_deg' be non-negative"
    assert deg_limit >= 0, "Reward error: 'deg_limit' should be non-negative"
    assert replacement_cost >= 0, "Reward error: battery replacement cost should be non-negative"

    return delta_deg * replacement_cost / deg_limit
