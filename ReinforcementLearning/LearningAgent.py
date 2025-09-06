import gymnasium as gym
env = gym.make('CartPole-v1', render_mode="human")
for _ in range(20): #restarts the environment 20 times
    observation, info = env.reset() #observation = the state (cart position, velocity, pole angle, angular velocity).
    for i in range(100): #Runs up to 100 timesteps inside each episode.
        env.render() #updates the animation
        print(observation) #prints the current state
        action = env.action_space.sample() #picks a random action, it's just acting randomly
        observation, reward, terminated, truncated, info = env.step(action) #reward for this step (in CartPole usually +1 for each timestep the pole stays up). True
        if terminated:
            print("Episode finished after {} timesteps".format(i+1))
            break