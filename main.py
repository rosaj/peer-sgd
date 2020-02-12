import simulation as sim

sim.default_network(3, 0.05)
"""
sim.simulate(name='kl',
             epochs=1,
             agent_count=2,
             p=0.02,
             scenario_func=sim.interact,
             default_weights=False,
             # data_func=sim.augment_test,
             # batches_num=lambda: sim.np.random.randint(2, 16),
             save_weights=False)
"""
# ako svaki agent ima samo jenu znamenku ? not working
#  ((192000/128) * 1000) / (60000/128)
