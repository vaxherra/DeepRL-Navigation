from unityagents import UnityEnvironment
from dqn_agent import Agent
from collections import deque

import yaml
import numpy as np
import random
import os
import torch
import matplotlib.pyplot as plt


def load_yaml(filename):
    """ Load a yaml file """
    with open(filename, 'r') as f:
        return yaml.load(f)


def plot_results(scores, title):
    """ Plot the results (scores) of the agent and save as a figure in PNG format """
    fig, ax = plt.subplots()
    ax.plot(scores)
    ax.set_title(title)
    ax.set_xlabel('Episodes')
    ax.set_ylabel('Score')
    ax.grid(True)
    plt.show()
    fig.savefig(title + '.png')
    plt.close()


def save_model(model, episode_num, save_path):
    """ Save the model checkpoint """
    checkpoint = {
        'episode_num': episode_num,
        'state_dict': model.state_dict()
    }
    torch.save(checkpoint, save_path)


def load_model(model, save_path):
    """ Load the model """
    checkpoint = torch.load(save_path)
    model.load_state_dict(checkpoint['state_dict'])
    return model


def dqn(agent, env, brain_name, settings, n_episodes=2000, max_t=100000, eps_start=1., eps_end=.01, eps_decay=.995,
        checkpoint_path="checkpoint.pth", model_path="model.pth"):
    """ Deep Q-Learning: a training method for Deep Reinforcement Q-Learning.

    Params:
        agent: the agent
        env: the environment
        brain_name: the name of the brain (the brain is the RL Gym environment)
        settings: the settings, loaded from the settings.yaml file
        n_episodes: the number of episodes to train the agent
        max_t: the maximum number of timesteps per episode
        eps_start: the starting value of epsilon, the epsilon-greedy parameter
        eps_end: the final value of epsilon after decay
        eps_decay: the decay rate of epsilon
        checkpoint_path: the path to the checkpoint file
        model_path: the path to the model file

    """

    scores = []
    scores_window = deque(maxlen=settings['evaluation']['window_size'])
    eps = eps_start

    for i_episode in range(n_episodes):
        env_info = env.reset(train_mode=True)[brain_name]
        state = env_info.vector_observations[0]
        score = 0
        for t in range(max_t):
            action = int(agent.act(state, eps))
            env_info = env.step(action)[brain_name]
            next_state = env_info.vector_observations[0]
            reward = env_info.rewards[0]
            done = env_info.local_done[0]
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break
        scores_window.append(score)
        scores.append(score)
        eps = max(eps_end, eps_decay * eps)

        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end='')

        if i_episode % settings['evaluation']['window_size'] == 0:
            save_model(agent.qnetwork_local, i_episode, checkpoint_path)

        if np.mean(scores_window) >= settings['evaluation']['min_score']:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(
                i_episode - settings['evaluation']['window_size'],
                np.mean(scores_window)))
            torch.save(agent.qnetwork_local.state_dict(), model_path)
            return scores
    return scores


def test_agent(env, brain_name, settings, train_params, model_path="model.pth"):
    """ Test the agent

    Load the model and test the agent on the environment, evaluating the agent's performance.
    Checks the agent's performance on the environment and evaluates the agent's performance against specific
    metric specified in the settings.yaml file.

    Params:
        env: the environment
        brain_name: the name of the brain (the brain is the RL Gym environment)
        settings: the settings, loaded from the settings.yaml file
        train_params: the parameters of the training
        model_path: the path to the model file



    """

    # Define an untrained agent
    agent_test = Agent(state_size=settings['state_size'], action_size=settings['action_size'], seed=settings['seed'],
                       train_params=train_params)

    # Load the trained model
    agent_test.qnetwork_local.load_state_dict(torch.load(model_path))

    # Reset the environment
    env_info = env.reset(train_mode=False)[brain_name]

    # Get the state, and initialize the score
    state = env_info.vector_observations[0]
    score = 0
    min_score = settings['evaluation']['min_score']

    # Loop until the agent finds the goal
    while True:
        action = agent_test.act(state)
        env_info = env.step(action)[brain_name]
        next_state = env_info.vector_observations[0]
        reward = env_info.rewards[0]
        done = env_info.local_done[0]
        score += reward
        state = next_state
        if done:
            break

    try:
        assert score >= min_score
    except AssertionError:
        print("The trained agent failed to find the goal")
    else:
        print(f"The trained agent found the goal: {min_score} average score, over"
              f" {settings['evaluation']['window_size']} episodes")


def main():
    # Load the config file
    settings = load_yaml("settings.yaml")

    # Load the environment
    env = UnityEnvironment(file_name=settings['data'])

    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    # Define the agent
    agent = Agent(state_size=settings['state_size'], action_size=settings['action_size'], seed=settings['seed'])

    # Train the agent
    scores = dqn(agent, env, brain_name, settings)

    # Plot and save the results
    plot_results(scores, 'Scores_plot')

    # Test the agent
    test_agent(env, brain_name, settings)

    # Close the environment
    env.close()


if __name__ == "__main__":
    main()
