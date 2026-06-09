pip install --quiet jax flax chex typing


import jax
import flax
from typing import Sequence, Type
from flax import linen as nn
from flax.linen.module import Module
from functools import partial
from typing import Any
import jax.numpy as jnp
from chex import Array
from jax import random
Dtype = Any
__all__ = ["Attention", "SpatialGatingUnit", "LayerNorm","Sequential", "Residual", "PreNorm", "Identity"]
ATTN_MASK_VALUE = -1e10
Dtype = Any
LayerNorm = partial(nn.LayerNorm)


import pandas as pd
df  = pd.read_csv('../input/amex-tfrecords/minidata (1).csv')
cat_features  = ['B_30', 'B_38', 'D_63', 'D_64', 'D_66', 'D_68', 'D_114', 'D_116', 'D_117', 'D_120', 'D_126']
cont_features = [x for x in list(df.columns) if x not in cat_features]
cont_features.remove('target')
len(cont_features)


len(cat_features)


cats = []
for col in cat_features:
  cats.append(df[col].unique().shape[0])
cats


cont_arr = jnp.array(df[cont_features].to_numpy())
cat_arr = jnp.array(df[cat_features].to_numpy(),dtype=int)
y_arr = jnp.array(df['target'].to_numpy())
print(cont_arr.shape,cat_arr.shape,y_arr.shape)





import wandb
from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()
wandb_key = user_secrets.get_secret("api key")
wandb.login(key = wandb_key)


# Save training data to W&B Artifacts
run = wandb.init(project='capstone', name='processed_data') 
artifact = wandb.Artifact(name='processed_data',type='dataset')
artifact.add_file("/kaggle/input/amex-tfrecords/minidata (1).csv")
wandb.log_artifact(artifact)
wandb.finish()


class Sequential(Module):
    layers: Sequence[Type[Module]]

    @nn.compact
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class Residual(Module):
    layers: Sequence[Type[Module]]
    @nn.compact
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x) + x
        return x


class PreNorm(Module):
    layers: Sequence[Type[Module]]
    def setup(self):
        self.norm = nn.LayerNorm()
    @nn.compact
    def __call__(self, x):
        for layer in self.layers:
            x = self.norm(x)
            x = layer(x)
        return x

class Identity(Module):
    @nn.compact
    def __call__(self, x):
        return x

class Attention(nn.Module):
    dim_out: int
    dim_head: int
    dtype: Dtype = jnp.float32
    def setup(self):
        self.scale = self.dim_head ** -0.5
        self.to_qkv = nn.Dense(features=self.dim_head * 3, dtype=self.dtype)
        self.to_out = nn.Dense(features=self.dim_out, dtype=self.dtype)
    @nn.compact
    def __call__(self, x) -> Array:
        n = x.shape[0]
        qkv = self.to_qkv(x)
        q, k, v = jnp.split(qkv, 3, axis=-1)
        sim = jnp.einsum("i d, j d -> i j", q, k) * self.scale
        mask = jnp.triu(jnp.ones((n, n), dtype=bool), 1)
        sim = jnp.where(mask, ATTN_MASK_VALUE, sim)
        attn = nn.softmax(sim, axis=-1)
        out = jnp.einsum("i j, j d -> i d", attn, v)
        return self.to_out(out)

class SpatialGatingUnit(nn.Module):
    dim_out: int
    dtype: Dtype = jnp.float32
    def setup(self):
        self.norm = LayerNorm(dtype=self.dtype)
        self.proj_out = nn.Dense(features=self.dim_out, dtype=self.dtype)
    @nn.compact
    def __call__(self, x, gate_res=None) -> Array:
        x, gate = jnp.split(x, 2, axis=-1)
        gate = self.norm(gate)
        if gate_res is not None:
            gate += gate_res
        x = x * gate
        return self.proj_out(x)
  
class gMLPBlock(nn.Module):
    dim: int
    dim_ff: int
    attn_dim: Any = None
    dtype: Dtype = jnp.float32

    def setup(self):
        self.proj_in = nn.Dense(features=self.dim_ff, dtype=self.dtype)
        self.attn = (
            Attention(
                dim_head=self.attn_dim, dim_out=self.dim_ff // 2, dtype=self.dtype
            )
            if self.attn_dim is not None
            else None
        )
        self.sgu = SpatialGatingUnit(dim_out=self.dim_ff // 2, dtype=self.dtype)
        self.proj_out = nn.Dense(features=self.dim, dtype=self.dtype)

    @nn.compact
    def __call__(self, x) -> Array:
        gate_res = self.attn(x) if self.attn is not None else None

        x = self.proj_in(x)
        x = nn.gelu(x)
        x = self.sgu(x, gate_res=gate_res)
        x = self.proj_out(x)
        return x


class gMLP(nn.Module):

    dim: int
    depth: int
    num_tokens: Any = None
    ff_mult: int = 4
    attn_dim: Any = None
    dtype: Dtype = jnp.float32

    def setup(self):
        dim_ff = self.dim * self.ff_mult
        self.to_embed = (
            nn.Embed(
                num_embeddings=self.num_tokens, features=self.dim, dtype=self.dtype
            )
            if self.num_tokens is not None
            else Identity()
        )

        self.layers = [
            Residual(
                [
                    PreNorm(
                        [
                            gMLPBlock(
                                dim=self.dim,
                                dim_ff=dim_ff,
                                attn_dim=self.attn_dim,
                                dtype=self.dtype,
                            )
                        ]
                    )
                ]
            )
            for i in range(self.depth)
        ]

        self.to_logits = (
            Sequential(
                [nn.LayerNorm(), nn.Dense(features=self.num_tokens, dtype=self.dtype)]
            )
            if self.num_tokens is not None
            else Identity()
        )

    @nn.compact
    def __call__(self, x) -> Array:
        x = self.to_embed(x)
        out = Sequential(self.layers)(x)
        return self.to_logits(out)


def scaled_dot_product(q, k, v, mask=None):
    d_k = q.shape[-1]
    attn_logits = jnp.matmul(q, jnp.swapaxes(k, -2, -1))
    attn_logits = attn_logits / math.sqrt(d_k)
    # print('dot1',attn_logits[0], attn_logits.shape, jnp.amax(attn_logits),d_k)
    attention = jax.nn.softmax(attn_logits)
    # print('dot2',attention[0])
    values = jnp.matmul(attention, v)
    return values, attention

class MultiheadAttention(nn.Module):
    embed_dim : int  # Output dimension
    num_heads : int  # Number of parallel heads (h)
    
    def setup(self):
        # Stack all weight matrices 1...h and W^Q, W^K, W^V together for efficiency
        # Note that in many implementations you see "bias=False" which is optional
        self.qkv_proj = nn.Dense(3*self.embed_dim,
                                 kernel_init=nn.initializers.xavier_uniform(),  # Weights with Xavier uniform init
                                 bias_init=nn.initializers.zeros  # Bias init with zeros
                                )
        self.o_proj = nn.Dense(self.embed_dim,
                               kernel_init=nn.initializers.xavier_uniform(),
                               bias_init=nn.initializers.zeros)

    def __call__(self, x, mask=None):
        batch_size, seq_length, embed_dim = x.shape
        qkv = self.qkv_proj(x)
        # Separate Q, K, V from linear output
        qkv = qkv.reshape(batch_size, seq_length, self.num_heads, -1)
        qkv = qkv.transpose(0, 2, 1, 3) # [Batch, Head, SeqLen, Dims]
        q, k, v = jnp.array_split(qkv, 3, axis=-1)
        # print('att0',jnp.isnan(jax.device_get(x)).any()==True )
        # Determine value outputs
        values, attention = scaled_dot_product(q, k, v, mask=mask)
        # print('att1',jnp.isnan(jax.device_get(values)).any()==True )
        values = values.transpose(0, 2, 1, 3) # [Batch, SeqLen, Head, Dims]
        values = values.reshape(batch_size, seq_length, embed_dim)
        o = self.o_proj(values)
        return o, attention

class EncoderBlock(nn.Module):
    input_dim : int  # Input dimension is needed here since it is equal to the output dimension (residual connection)
    num_heads : int
    dim_feedforward : int
    dropout_prob : float
    
    def setup(self):
        # Attention layer
        self.self_attn = MultiheadAttention(embed_dim=self.input_dim, 
                                            num_heads=self.num_heads)
        # Two-layer MLP
        self.linear = [
            nn.Dense(self.dim_feedforward),
            nn.Dropout(self.dropout_prob),
            nn.relu,
            nn.Dense(self.input_dim)
        ]
        # Layers to apply in between the main layers
        self.norm1 = nn.LayerNorm()
        self.norm2 = nn.LayerNorm()
        self.dropout = nn.Dropout(self.dropout_prob)

    def __call__(self, x, mask=None, train=True):
        # Attention part
        # print('block0',jnp.isnan(jax.device_get(x)).any()==True )
        attn_out, _ = self.self_attn(x, mask=mask)
        # print('block-1',attn_out[0])
        x = x + self.dropout(attn_out, deterministic=not train)
        x = self.norm1(x)
        # MLP part
        linear_out = x
        for l in self.linear:
            linear_out = l(linear_out) if not isinstance(l, nn.Dropout) else l(linear_out, deterministic=not train)
        x = x + self.dropout(linear_out, deterministic=not train)
        x = self.norm2(x)
        return x

class TransformerEncoder(nn.Module):
    num_layers : int
    input_dim : int
    num_heads : int
    dim_feedforward : int
    dropout_prob : float
    
    def setup(self):
        self.layers = [EncoderBlock(self.input_dim, self.num_heads, self.dim_feedforward, self.dropout_prob) for _ in range(self.num_layers)]

    def __call__(self, x, mask=None, train=True):
        for l in self.layers:
            # print('transenc',x[0])
            x = l(x, mask=mask, train=train)
        # print('transenc',x[0])
        return x

class GatedTabTransformer(nn.Module):
    model_dim : int                   # Hidden dimensionality to use inside the Transformer
    num_classes : int                 # Number of classes to predict per sequence element
    num_heads : int                   # Number of heads to use in the Multi-Head Attention blocks
    num_layers : int                  # Number of encoder blocks to use
    categories : Sequence
    gmlp_blocks : int
    dropout_prob : float = 0.2        # Dropout to apply inside the model
    input_dropout_prob : float = 0.0  # Dropout to apply on the input features
    embedding_dim : int =256

    def setup(self):
        # Input dim -> Model dim
        self.input_layer = nn.Dense(self.model_dim)
        self.contnorm = nn.LayerNorm()
        # Transformer
        self.transformer = TransformerEncoder(num_layers=self.num_layers,
                                              input_dim=self.model_dim,
                                              dim_feedforward=2*self.model_dim,
                                              num_heads=self.num_heads,
                                              dropout_prob=self.dropout_prob)
        # Output classifier per sequence lement
        self.embed_layers = [nn.Embed(num_embeddings = 10, features = self.model_dim ) for number_of_classes in self.categories]
        self.embedder2 = nn.Dense(self.embedding_dim)
        # print(type(self.embed_layers))
        # print(self.categories)
        # for number_of_classes in self.categories:
        #     self.embed_layers.append(nn.Embed(num_embeddings = self.model_dim, features = number_of_classes))
        # print(22222)
        self.gmlp_layers = []
        self.gmlp_layer = gMLP(dim = self.embedding_dim , depth =self.gmlp_blocks)
        self.output_net = [
            nn.Dense(self.model_dim),
            nn.LayerNorm(),
            nn.relu,
            nn.Dropout(self.dropout_prob),
            nn.Dense(self.num_classes),
            nn.softmax
        ]

    def __call__(self, cat_x,cont_x, mask=None,train=True):
        cont_x = self.contnorm(cont_x)
        embedding_outputs = []
        for categorical_input, embedding_layer in zip(cat_x.T, self.embed_layers):
            embedding_outputs.append(jnp.expand_dims(embedding_layer(categorical_input),axis=1))
        # print(cat_x.shape)
        categorical_inputs = jnp.concatenate(embedding_outputs,axis=1)
        # print('cat',jnp.isnan(jax.device_get(cat_x)).any()==True )
        # print(categorical_inputs.shape)
        # print(1111, categorical_inputs.shape)
        # categorical_inputs = self.input_layer(categorical_inputs)
        # print(2222, categorical_inputs.shape)
        categorical_inputs = self.transformer(categorical_inputs, mask=mask, train=train)
        # print(categorical_inputs)
        x,y,z = categorical_inputs.shape
        # print(4444 , cont_x.shape)
        # print(categorical_inputs.shape)
        categorical_inputs = jnp.reshape(categorical_inputs,newshape=(x,y*z))
        gmlp_inp = jnp.concatenate([categorical_inputs,cont_x],axis=1)
        
        gmlp_inp = jnp.expand_dims(self.embedder2(gmlp_inp),axis=1)
        gmlp_inp = self.gmlp_layer(gmlp_inp)
        
        x = jnp.mean(gmlp_inp,axis=1)
        for l in self.output_net:
            x = l(x) if not isinstance(l, nn.Dropout) else l(x, deterministic=not train)
        return x



import math
main_rng = random.PRNGKey(42)
main_rng, x_rng = random.split(main_rng)


transpre = GatedTabTransformer(num_layers=5, 
                                model_dim=128,
                                num_classes=2,
                                num_heads=4,
                                dropout_prob=0.15,
                                input_dropout_prob=0.05,
                                gmlp_blocks = 6,
                                categories = cats)
# Initialize parameters of transformer predictor with random key and inputs
main_rng, init_rng, dropout_init_rng = random.split(main_rng, 3)
params = transpre.init({'params': init_rng, 'dropout': dropout_init_rng}, cat_arr,cont_arr, train=True)['params']
# Apply transformer predictor with parameters on the inputs
# Since dropout is stochastic, we need to pass a rng to the forward
main_rng, dropout_apply_rng = random.split(main_rng)
# Instead of passing params and rngs every time to a function call, we can bind them to the module
binded_mod = transpre.bind({'params': params}, rngs={'dropout': dropout_apply_rng})
out = binded_mod(cat_arr,cont_arr, train=True)
print('Out', out.shape)



y_arr1 = jax.nn.one_hot(y_arr, num_classes=2)
from flax.training import train_state  # Useful dataclass to keep train state
from tqdm import tqdm
import numpy as np                     # Ordinary NumPy
import optax  

def cross_entropy_loss(*, logits, labels):
  # labels_onehot = jax.nn.one_hot(labels, num_classes=10)
  return optax.softmax_cross_entropy(logits=logits, labels=labels).mean()

def compute_metrics(*, logits, labels):
  loss = cross_entropy_loss(logits=logits, labels=labels)
  # print(logits.shape, labels.shape)
  arr = jnp.argmax(logits, -1)
  # print(arr.shape)
  accuracy = jnp.mean(jnp.argmax(logits, -1) == jnp.argmax(labels, -1))
  metrics = {
      'loss': loss,
      'accuracy': accuracy,
  }
  return metrics
def create_train_state(init_rng,dropout_init_rng,cat,cont, learning_rate, momentum):
  """Creates initial `TrainState`."""
  model = GatedTabTransformer(num_layers=5, 
                                model_dim=128,
                                num_classes=2,
                                num_heads=4,
                                dropout_prob=0.15,
                                input_dropout_prob=0.05,
                                gmlp_blocks = 6,
                                categories = cats)
  params = model.init({'params': init_rng, 'dropout': dropout_init_rng}, cat,cont, train=True)['params']
  tx = optax.sgd(learning_rate, momentum)
  return train_state.TrainState.create(
      apply_fn=model.apply, params=params, tx=tx)

@jax.jit
def train_step(state, cat,cont,y,dropout_apply_rng):
  """Train for a single step."""
  def loss_fn(params):
    model = GatedTabTransformer(num_layers=5, 
                                model_dim=128,
                                num_classes=2,
                                num_heads=4,
                                dropout_prob=0.15,
                                input_dropout_prob=0.05,
                                gmlp_blocks = 6,
                                categories = cats)
    logits = model.apply({'params': params}, cat,cont,rngs={'dropout': dropout_apply_rng})
    loss = cross_entropy_loss(logits=logits, labels=y)
    return loss, logits
  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
  (_, logits), grads = grad_fn(state.params)
  state = state.apply_gradients(grads=grads)
  metrics = compute_metrics(logits=logits, labels=y)
  return state, metrics

# def train_epoch(state, cat,cont,y, epoch, rng):
#   main_rng, dropout_apply_rng = random.split(rng)
  
#   state, metrics = train_step(state, cat,cont,y,dropout_apply_rng)
#   print('train epoch: %d, loss: %.4f, accuracy: %.2f' % (
#       epoch, metrics['loss'], metrics['accuracy'] * 100))

#   return state

def train_epoch(state, cat,cont,y, epoch, rng):
  main_rng, dropout_apply_rng = random.split(rng)
  batch_metrics = []
  for i in tqdm(range(0,30000,32)):
    cat_x = cat[i:i+32]    
    cont_x = cont[i:i+32]
    yy = y[i:i+32]
    state, metrics = train_step(state, cat_x,cont_x,yy,dropout_apply_rng)
    batch_metrics.append(metrics)

  batch_metrics_np = jax.device_get(batch_metrics)
  epoch_metrics_np = {
      k: np.mean([metrics[k] for metrics in batch_metrics_np])
      for k in batch_metrics_np[0]}

  print('train epoch: %d, loss: %.4f, accuracy: %.2f' % (
      epoch, epoch_metrics_np['loss'], epoch_metrics_np['accuracy'] * 100))

  return state,epoch_metrics_np

learning_rate = 0.001
momentum = 0.9
main_rng = random.PRNGKey(42)
main_rng, x_rng = random.split(main_rng)
state = create_train_state(main_rng,x_rng,cat_arr,cont_arr, learning_rate, momentum)
all_mets=[]
for epoch in range(1, 10 + 1):
  # Use a separate PRNG key to permute image data during shuffling
  rng, input_rng = jax.random.split(main_rng)
  # Run an optimization step over a training batch
  state,mets = train_epoch(state, cat_arr, cont_arr,y_arr1, epoch, input_rng)
  all_mets.append(mets)


import matplotlib.pyplot as plt

# summarize history for accuracy
plt.plot([mets['accuracy'] for mets in all_mets])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# summarize history for loss
plt.plot([mets['loss'] for mets in all_mets])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

