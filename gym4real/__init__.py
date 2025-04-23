from gymnasium.envs.registration import register

register(
    id='gym4real/micro_grid-v0',
    entry_point='gym4real.envs.microgrid.env:MicroGridEnv',
)

register(
    id='gym4real/robofeeder-v0',
    entry_point='gym4real.envs.robofeeder.Env_0:robotEnv',
)

register(
    id='gym4real/robofeeder-v1',
    entry_point='gym4real.envs.robofeeder.Env_1:robotEnv',
)

register(
    id='gym4real/robofeeder-v2',
    entry_point='gym4real.envs.robofeeder.Env_2:robotEnv',
)