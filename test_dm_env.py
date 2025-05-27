import envpool
env_name = "HumanoidRun-v1"
env = envpool.make_dm(env_name, num_envs=1)
a = env.reset()
print(a.observation)
# Assuming 'env_pool' is your DmcHumanoidDMEnvPool object
observation_spec = env.observation_spec

print(observation_spec)

# observation_spec is usually a dictionary or a single ArraySpec
# For a single observation, it will be an ArraySpec
# For environments with multiple observation components, it might be a dictionary of ArraySpecs

# If the environment provides a single observation array:
observation_shape = observation_spec.shape
observation_dtype = observation_spec.dtype

print(f"Observation Shape: {observation_shape}")
print(f"Observation Dtype: {observation_dtype}")

# If the environment provides a dictionary of observations (common in dm_control):
# You'll get a dictionary of specs
if isinstance(observation_spec, dict):
    print("Observation Spec is a dictionary:")
    for key, spec in observation_spec.items():
        print(f"  Key: {key}, Shape: {spec.shape}, Dtype: {spec.dtype}")
    # To get the shape of a specific component, e.g., 'observations' or 'position':
    # try:
    #     specific_shape = observation_spec['observations'].shape # Or 'position', 'velocity', etc.
    #     print(f"Shape of 'observations' component: {specific_shape}")
    # except KeyError:
    #     print("The key 'observations' was not found in the spec dictionary.")