import gymnasium as gym
env = gym.make("CartPole-v1", render_mode="human") #Creates an instance of the CartPole-v0 environment: a pole attached to a cart
env.reset() #Resets the environment to the starting state
for _ in range(1000): #Runs a loop for 1000 iterations (steps in the simulation).
    env.render()
    env.step(env.action_space.sample())
    #env.action_space.sample = picks a random action (move left or right).
    #env.step(action) = applies the action in the environment.