from gymnasium.envs.registration import register

register(
    id='gym4real/micro_grid-v0',
    entry_point='gym4real.envs.microgrid.env:MicroGridEnv',
)