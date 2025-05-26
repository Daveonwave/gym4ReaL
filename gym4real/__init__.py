from gymnasium.envs.registration import register

register(
    id='gym4real/microgrid-v0',
    entry_point='gym4real.envs.microgrid.env:MicroGridEnv',
)

register(
    id='gym4real/dam-v0',
    entry_point='gym4real.envs.dam.env:DamEnv',
)

register(
    id='gym4real/wds-v0',
    entry_point='gym4real.envs.wds.env:WaterDistributionSystemEnv',
)

#register(
#    id='gym4real/wds_cps-v0',
#    entry_point='gym4real.envs.wds.env_cps:WaterDistributionSystemEnv',
#)

register(
    id='gym4real/robofeeder-picking-v0',
    entry_point='gym4real.envs.robofeeder.rf_picking_v0:robotEnv',
)

register(
    id='gym4real/robofeeder-picking-v1',
    entry_point='gym4real.envs.robofeeder.rf_picking_v1:robotEnv',
)

register(
    id='gym4real/robofeeder-planning',
    entry_point='gym4real.envs.robofeeder.rf_planning:robotEnv',
)

register(
    id='gym4real/elevator-v0',
    entry_point='gym4real.envs.elevator.env:ElevatorEnv'
    )