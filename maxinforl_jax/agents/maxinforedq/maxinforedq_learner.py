"""Implementations of RedQ.
https://arxiv.org/abs/2101.05982
"""

import functools
from typing import Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax

from jaxrl.agents.sac import temperature
from maxinforl_jax.agents.maxinforedq.actor import update as update_actor
from maxinforl_jax.agents.maxinforedq.critic import target_update
from maxinforl_jax.agents.maxinforedq.critic import update as update_critic
from jaxrl.agents.sac.temperature import update as update_temp

from jaxrl.datasets import Batch
from jaxrl.networks import critic_net, policies
from jaxrl.networks.common import InfoDict, Model, PRNGKey

from maxinforl_jax.models.ensemble_model import EnsembleState, DeterministicEnsemble, ProbabilisticEnsemble


@functools.partial(jax.jit,
                   static_argnames=('ens',
                                    'backup_entropy',
                                    'n',
                                    'm',
                                    'update_target',
                                    'update_policy',
                                    'update_model',
                                    'use_log_transform',
                                    'predict_rewards',
                                    'predict_diff',
                                    ))
def _update_jit(
        rng: PRNGKey,
        actor: Model,
        critic: Model,
        target_actor: Model,
        target_critic: Model,
        temp: Model,
        dyn_entropy_temp: Model,
        ens: DeterministicEnsemble,
        ens_state: EnsembleState,
        batch: Batch, discount: float, tau: float,
        target_entropy: float, backup_entropy: bool, n: int, m: int,
        update_target: bool, update_policy: bool, update_model: bool,
        use_log_transform: bool,
        predict_rewards: bool, predict_diff: bool
) -> Tuple[PRNGKey, Model, Model, Model, Model, Model, Model, EnsembleState, InfoDict]:
    rng, key = jax.random.split(rng)
    new_critic, ens_state, critic_info = update_critic(rng=key,
                                                       actor=actor,
                                                       critic=critic,
                                                       target_critic=target_critic,
                                                       temp=temp,
                                                       dyn_entropy_temp=dyn_entropy_temp,
                                                       ens=ens,
                                                       ens_state=ens_state,
                                                       batch=batch,
                                                       discount=discount,
                                                       backup_entropy=backup_entropy,
                                                       n=n,
                                                       m=m)
    if update_target:
        new_target_critic = target_update(new_critic, target_critic, tau)
    else:
        new_target_critic = target_critic

    if update_policy:
        rng, key = jax.random.split(rng)
        new_actor, ens_state, actor_info = update_actor(
            key=key,
            actor=actor,
            target_actor=target_actor,
            critic=new_critic,
            temp=temp,
            dyn_entropy_temp=dyn_entropy_temp,
            ens=ens,
            ens_state=ens_state,
            batch=batch)
        if update_target:
            new_target_actor = target_update(new_actor, target_actor, tau)
        else:
            new_target_actor = target_actor

        new_temp, alpha_info = update_temp(temp, actor_info['entropy'],
                                           target_entropy, use_log_transform=use_log_transform)

        new_dyn_entropy_temp, dyn_ent_info = update_temp(dyn_entropy_temp, actor_info['info_gain'],
                                                         actor_info['target_info_gain'],
                                                         use_log_transform=use_log_transform)
        dyn_ent_info = {f'dyn_ent_{key}': val for key, val in dyn_ent_info.items()}

    else:
        new_actor, actor_info = actor, {}
        new_target_actor = target_actor
        new_temp, alpha_info = temp, {}
        new_dyn_entropy_temp, dyn_ent_info = dyn_entropy_temp, {}

    if update_model:
        if predict_diff:
            outputs = batch.next_observations - batch.observations
        else:
            outputs = batch.next_observations
        if predict_rewards:
            outputs = jnp.concatenate([outputs, batch.rewards.reshape(-1, 1)], axis=-1)
        new_ens_state, (loss, mse) = ens.update(
            input=jnp.concatenate([batch.observations, batch.actions], axis=-1),
            output=outputs,
            state=ens_state,
        )
        ens_info = {'ens_loss': loss, 'ens_mse': mse,
                    'ens_inp_mean': ens_state.ensemble_normalizer_state.input_normalizer_state.mean.mean(),
                    'ens_inp_std': ens_state.ensemble_normalizer_state.input_normalizer_state.std.mean(),
                    'ens_out_mean': ens_state.ensemble_normalizer_state.output_normalizer_state.mean.mean(),
                    'ens_out_std': ens_state.ensemble_normalizer_state.output_normalizer_state.std.mean(),
                    'ens_info_gain_mean': ens_state.ensemble_normalizer_state.info_gain_normalizer_state.mean.mean(),
                    'ens_info_gain_std': ens_state.ensemble_normalizer_state.info_gain_normalizer_state.std.mean(),
                    }

    else:
        new_ens_state, ens_info = ens_state, {}

    return rng, \
        new_actor, \
        new_critic, \
        new_target_critic, \
        new_target_actor, \
        new_temp, \
        new_dyn_entropy_temp, \
        new_ens_state, {
        **critic_info,
        **actor_info,
        **alpha_info,
        **dyn_ent_info,
        **ens_info,
    }


class MaxInfoREDQLearner(object):

    def __init__(
            self,
            seed: int,
            observations: jnp.ndarray,
            actions: jnp.ndarray,
            actor_lr: float = 3e-4,
            critic_lr: float = 3e-4,
            temp_lr: float = 3e-4,
            dyn_ent_lr: float = 3e-4,
            dyn_wd: float = 0.0,
            ens_lr: float = 3e-4,
            ens_wd: float = 0.0,
            n: int = 10,  # Number of critics.
            m: int = 2,  # Nets to use for critic backups.
            policy_update_delay: int = 20,  # See the original implementation.
            model_update_delay: int = 1,
            hidden_dims: Sequence[int] = (256, 256),
            model_hidden_dims: Sequence[int] = (256, 256),
            num_heads: int = 5,
            predict_reward: bool = True,
            predict_diff: bool = True,
            use_log_transform: bool = True,
            learn_std: bool = False,
            discount: float = 0.99,
            tau: float = 0.005,
            target_update_period: int = 1,
            target_entropy: Optional[float] = None,
            backup_entropy: bool = True,
            init_temperature: float = 1.0,
            init_temperature_dyn_entropy: float = 1.0,
            init_mean: Optional[np.ndarray] = None,
            policy_final_fc_init_scale: float = 1.0):
        """
        An implementation of the version of Soft-Actor-Critic described in https://arxiv.org/abs/1812.05905
        """
        self.predict_reward = predict_reward
        self.predict_diff = predict_diff
        self.num_heads = num_heads

        action_dim = actions.shape[-1]

        if target_entropy is None:
            self.target_entropy = -action_dim
        else:
            self.target_entropy = target_entropy

        self.backup_entropy = backup_entropy
        self.n = n
        self.m = m
        self.policy_update_delay = policy_update_delay

        self.tau = tau
        self.target_update_period = target_update_period
        self.discount = discount

        self.model_update_delay = model_update_delay
        self.use_log_transform = use_log_transform

        rng = jax.random.PRNGKey(seed)
        rng, actor_key, critic_key, temp_key = jax.random.split(rng, 4)
        actor_def = policies.NormalTanhPolicy(
            hidden_dims,
            action_dim,
            init_mean=init_mean,
            final_fc_init_scale=policy_final_fc_init_scale)
        actor = Model.create(actor_def,
                             inputs=[actor_key, observations],
                             tx=optax.adam(learning_rate=actor_lr))

        target_actor = Model.create(actor_def,
                                    inputs=[actor_key, observations])

        critic_def = critic_net.DoubleCritic(hidden_dims, num_qs=n)
        critic = Model.create(critic_def,
                              inputs=[critic_key, observations, actions],
                              tx=optax.adam(learning_rate=critic_lr))
        target_critic = Model.create(
            critic_def, inputs=[critic_key, observations, actions])

        temp = Model.create(temperature.Temperature(init_temperature),
                            inputs=[temp_key],
                            tx=optax.adam(learning_rate=temp_lr))

        # information gain kwargs
        dyn_ent_temp_key, rng = jax.random.split(rng, 2)
        dyn_ent_temp = Model.create(temperature.Temperature(init_temperature_dyn_entropy),
                                    inputs=[dyn_ent_temp_key],
                                    tx=optax.adamw(learning_rate=dyn_ent_lr, weight_decay=dyn_wd))

        tx = optax.adamw(learning_rate=ens_lr, weight_decay=ens_wd)
        model_key, rng = jax.random.split(rng, 2)

        output_dim = observations.shape[-1]
        if predict_reward:
            output_dim += 1

        if learn_std:
            model_type = ProbabilisticEnsemble
        else:
            model_type = DeterministicEnsemble
        ensemble = model_type(
            model_kwargs={'hidden_dims': model_hidden_dims + (output_dim,)},
            optimizer=tx,
            num_heads=self.num_heads)

        ens_state = ensemble.init(key=model_key, input=jnp.concatenate([observations, actions], axis=-1))

        self.actor = actor
        self.target_actor = target_actor
        self.critic = critic
        self.target_critic = target_critic
        self.temp = temp
        self.dyn_ent_temp = dyn_ent_temp
        self.ens_state = ens_state
        self.ensemble = ensemble
        self.rng = rng

        self.step = 0

    def sample_actions(self,
                       observations: np.ndarray,
                       temperature: float = 1.0) -> np.ndarray:
        rng, actions = policies.sample_actions(self.rng, self.actor.apply_fn,
                                               self.actor.params, observations,
                                               temperature)
        self.rng = rng

        actions = np.asarray(actions)
        return np.clip(actions, -1, 1)

    def update(self, batch: Batch) -> InfoDict:
        self.step += 1
        new_rng, new_actor, new_critic, new_target_critic, \
            new_target_actor, new_temp, \
            new_dyn_entropy_temp, new_ens_state, info = _update_jit(
                rng=self.rng,
                actor=self.actor,
                critic=self.critic,
                target_actor=self.target_actor,
                target_critic=self.target_critic,
                target_entropy=self.target_entropy,
                temp=self.temp,
                dyn_entropy_temp=self.dyn_ent_temp,
                ens=self.ensemble,
                ens_state=self.ens_state,
                batch=batch,
                discount=self.discount,
                tau=self.tau,
                backup_entropy=self.backup_entropy,
                n=self.n,
                m=self.m,
                update_target=self.step % self.target_update_period == 0,
                update_policy=self.step % self.policy_update_delay == 0,
                update_model=self.step % self.model_update_delay == 0,
                use_log_transform=self.use_log_transform,
                predict_diff=self.predict_diff,
                predict_rewards=self.predict_reward)

        self.rng = new_rng
        self.actor = new_actor
        self.critic = new_critic
        self.target_actor = new_target_actor
        self.target_critic = new_target_critic
        self.temp = new_temp
        self.dyn_ent_temp = new_dyn_entropy_temp
        self.ens_state = new_ens_state

        return info
