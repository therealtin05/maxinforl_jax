from typing import Tuple

import jax
import jax.numpy as jnp

from jaxrl.datasets import Batch
from jaxrl.networks.common import InfoDict, Model, Params, PRNGKey
from maxinforl_jax.models.ensemble_model import EnsembleState, DeterministicEnsemble


def target_update(critic: Model, target_critic: Model, tau: float) -> Model:
    new_target_params = jax.tree_util.tree_map(
        lambda x, y: x * (1 - tau) + y * tau, target_critic.params,
        critic.params)

    return target_critic.replace(params=new_target_params)


def update(rng: PRNGKey,
           actor: Model,
           critic: Model,
           target_critic: Model,
           temp: Model,
           dyn_entropy_temp: Model,
           ens: DeterministicEnsemble,
           ens_state: EnsembleState,
           batch: Batch, discount: float, backup_entropy: bool,
           n: int, m: int) -> Tuple[Model, EnsembleState, InfoDict]:
    dist = actor(batch.next_observations)
    rng, key = jax.random.split(rng)
    next_actions = dist.sample(seed=key)
    next_log_probs = dist.log_prob(next_actions)

    # get info gain
    info_gain, new_ens_state = ens.get_info_gain(
        input=jnp.concatenate([batch.next_observations, next_actions], axis=-1),
        state=ens_state, update_normalizer=False)

    all_indx = jnp.arange(0, n)
    rng, key = jax.random.split(rng)
    indx = jax.random.choice(key, a=all_indx, shape=(m,), replace=False)
    # params = jax.tree_util.tree_map(lambda param: param[indx],
    #                                target_critic.params)
    # next_qs = target_critic.apply_fn({'params': params},
    #                                 batch.next_observations, next_actions)
    # evaluate for all params
    next_qs = target_critic(batch.next_observations, next_actions)
    # sample random indices
    next_qs = jax.tree_util.tree_map(lambda x: x[indx],
                                     next_qs)
    next_q = jnp.min(next_qs, axis=0)

    target_q = batch.rewards + discount * batch.masks * next_q

    if backup_entropy:
        dyn_ent_coef, _ = dyn_entropy_temp()
        act_ent_coef, _ = temp()
        total_entropy = dyn_ent_coef * info_gain - act_ent_coef * next_log_probs
        target_q += discount * batch.masks * total_entropy

    def critic_loss_fn(critic_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        qs = critic.apply_fn({'params': critic_params}, batch.observations,
                             batch.actions)
        critic_loss = ((qs - target_q) ** 2).mean()
        return critic_loss, {'critic_loss': critic_loss, 'qs': qs.mean()}

    new_critic, info = critic.apply_gradient(critic_loss_fn)

    return new_critic, new_ens_state, info
