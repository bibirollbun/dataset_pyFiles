import jax
import jax.numpy as jnp
import chex

import flax.struct

import json

import matplotlib.pyplot as plt
from matplotlib import colors


class JaxFunction:
    def __init__(self, inputs_types, output_type, name=None):
        self.name = name if name else self.__class__.__name__
        self.inputs_types = inputs_types
        self.output_type = output_type

    def _set_registry(self, registry: "DSLRegistry"):
        self.inputs_types_: tuple[int] = tuple(registry.get_type_id(t) for t in self.inputs_types)
        self.output_type_: int = registry.get_type_id(self.output_type)

    def __call__(self, *args):
        return self._call(*args)

    def call(self, *args: tuple["UnionStruct"]) -> "UnionStruct":
        # TODO check type of args
        typed_inputs = tuple(arg.values[t] for t, arg in zip(self.inputs_types_, args))
        typed_output = self._call(*typed_inputs)

        union_output = jax.tree.map(jnp.zeros_like, args[0].values)
        union_output = union_output[:self.output_type_] + (typed_output,) + union_output[self.output_type_ + 1:]

        return UnionStruct(dtype=self.output_type_, values=union_output)

    def _call(self, *inputs):
        raise NotImplementedError


class DSLRegistry:
    def __init__(self, name=None):
        self.name = name

        self.next_type_id = 0
        self.type_name_to_id = {}
        self.type_prototypes = []

        self.funcs: list[JaxFunction] = []
        self.func_names = set()

    def register_type(self, name: str, prototype=None):
        type_id = self.next_type_id
        self.next_type_id = self.next_type_id + 1
        self.type_name_to_id[name] = type_id
        self.type_prototypes.append(prototype)

    def register_func(self, func: JaxFunction):
        assert func.name not in self.func_names
        self.func_names.add(func.name)
        self.funcs.append(func)
        func._set_registry(self)
        self.partials = [f.call for f in self.funcs]

    def get_type_id(self, type_name) -> int:
        return self.type_name_to_id[type_name]

    def get_arity(self) -> int:
        return max(len(f.inputs_types_) for f in self.funcs)

    def get_info(self) -> "RegistryInfo":
        n_funcs = len(self.funcs)
        n_types = len(self.type_name_to_id)
        arity = self.get_arity()

        inputs_type = -jnp.ones((n_funcs, arity), dtype=jnp.int8)
        output_type = -jnp.ones((n_funcs,), dtype=jnp.int8)
        inputs_type_count = jnp.zeros((n_funcs, n_types), dtype=jnp.int8)
        for f_id, f in enumerate(self.funcs):
            #for i, type_id in enumerate(f.inputs_types_):
            #    inputs_type = inputs_type.at[f_id, i].set(type_id)
            output_type = output_type.at[f_id].set(f.output_type_)

            f_inputs_type = jnp.array([type_id for type_id in f.inputs_types_])
            f_inputs_type = jnp.pad(f_inputs_type, (0, arity - f_inputs_type.size), constant_values=-1)
            inputs_type = inputs_type.at[f_id].set(f_inputs_type)

            for type_name, type_id in self.type_name_to_id.items():
                inputs_type_count = inputs_type_count.at[f_id, type_id].set(jnp.sum(f_inputs_type == type_id))

        return RegistryInfo(
            n_funcs=len(self.funcs),
            n_types=len(self.type_name_to_id),
            arity=arity,
            inputs_type=inputs_type,
            output_type=output_type,
            inputs_type_count=inputs_type_count
        )

    def init_struct(self) -> "UnionStruct":
        dtype = -jnp.ones((), dtype=jnp.int8)
        values = tuple(jnp.zeros_like(proto) for proto in self.type_prototypes)
        return UnionStruct(dtype=dtype, values=values)

    def fill_struct(self, inputs) -> "UnionStruct":
        types = []
        unions = []

        proto_values = self.init_struct().values
        for input_type, input_value in inputs:
            types.append(input_type)
            # check dtype and shape
            proto_value = self.type_prototypes[input_type]
            chex.assert_trees_all_equal_structs(input_value, proto_value)
            input_value = jax.tree.map(lambda i, p: jnp.broadcast_to(i, p.shape), input_value, proto_value)
            input_value = jax.tree.map(lambda i, p: jnp.astype(i, p.dtype), input_value, proto_value)
            chex.assert_trees_all_equal_shapes_and_dtypes(input_value, proto_value)
            #
            input_values = proto_values[:input_type] + (input_value,) + proto_values[input_type + 1:]
            unions.append(input_values)
        types = jnp.stack(types, axis=0, dtype=int)
        unions = jax.tree.transpose(jax.tree.structure(['*'] * len(unions)), None, unions)
        unions = jax.tree.map(lambda t: jnp.stack(t, 0), unions, is_leaf=lambda t: isinstance(t, list))
        return UnionStruct(dtype=types, values=unions)

    def validate(self, ssa: "SSA", inputs=None):
        valid = 1
        for i in range(ssa.length):
            op = ssa.op[i]
            args = ssa.op[i]

            if ssa.type[i] != self.funcs[op].output_type_:
                valid = 0

            if any((a >= i) for a in args):
                valid = 0

            for j, a in enumerate(args):
                if a >= 0:
                    a_type = ssa.type[a]
                    if a_type != self.funcs[op].inputs_types_:
                        valid = 0

        return valid


    def run(self, ssa: "SSA", inputs):
        return run(self, ssa, inputs)


@flax.struct.dataclass
class RegistryInfo:
    n_funcs: int = flax.struct.field(pytree_node=False, metadata=dict(static=True))
    n_types: int = flax.struct.field(pytree_node=False, metadata=dict(static=True))
    arity: int = flax.struct.field(pytree_node=False, metadata=dict(static=True))
    inputs_type: jax.Array
    output_type: jax.Array
    inputs_type_count: jax.Array


@flax.struct.dataclass
class SSA:
    id: jax.Array
    op: jax.Array
    type: jax.Array
    args: jax.Array
    depth: jax.Array

    @property
    def length(self):
        return self.id.shape[-1]

    @property
    def inputs_mask(self):
        return (self.type >= 0) & (self.op < 0)

    @property
    def arity(self):
        return self.args.shape[-1]

    @property
    def output_id(self):
        var_mask = (self.type >= 0)
        return jnp.where(jnp.all(var_mask, -1), self.length - 1, jnp.argmin(var_mask, -1) - 1)

    @classmethod
    def init_empty(cls, length, arity):
        id_ = jnp.arange(length, dtype=jnp.int16)
        zeros_1d = -jnp.ones(length, dtype=jnp.int16)
        zeros_2d = -jnp.ones((length, arity), dtype=jnp.int16)
        return cls(
            id=id_,
            op=zeros_1d,
            type=zeros_1d,
            depth=zeros_1d,
            args=zeros_2d
        )

    @classmethod
    def init(cls, length, arity, input_types):
        input_types = jnp.array(input_types)
        assert input_types.ndim == 1
        ssa = cls.init_empty(length, arity)
        ssa = ssa.replace(
            type=ssa.type.at[:input_types.size].set(input_types),
            depth=ssa.depth.at[:input_types.size].set(0)
        )
        return ssa

    def prune(self, output_indexes) -> "SSA":
        return prune(self, output_indexes)

    def normalize(self) -> "SSA":
        return order_by_depth(self)


@flax.struct.dataclass
class UnionStruct:
    dtype: jax.Array
    values: tuple[jax.Array, ...]


# ----- SAMPLE -----
def sample_op(key, info, ssa):
    in_types  = jnp.arange(info.n_types).reshape(1, -1)
    ssa_types = ssa.type.reshape(-1, 1)
    input_types_count = jnp.sum(ssa_types == in_types, 0)
    
    probs = jnp.prod((info.inputs_type_count == 0) | jnp.reshape(input_types_count > 0, (1, -1)), -1)
    op_id = jax.random.choice(key, probs.shape[0], p=probs).astype(ssa.op.dtype)
    return op_id


def sample_args(key, info, ssa, node_id, op_id, beta1=0.0, beta2=0.0, n_keep=100, replace=True):
    def step_fn(i, args):
        type_id = info.inputs_type[op_id, i]

        # masking
        mask = (ssa.type == type_id) & (ssa.id < node_id)
        if not replace:
            mask = mask & jnp.all(jnp.expand_dims(ssa.id, -1) != jnp.expand_dims(args, 0), -1)
        
        # probability
        cum_sum_mask = jnp.cumsum(mask)
        x = ssa.id / mask.size
        p = mask * (1. + x * beta1 + x ** 2 * beta2) * jnp.clip(n_keep + cum_sum_mask - cum_sum_mask[-1], 0, 1)
        
        # sampling
        a = jax.random.choice(key[i], ssa.length, p=p)
        a = jnp.where(type_id < 0, -1, a)
        args = args.at[i].set(a)
        return args

    key = jax.random.split(key, ssa.arity)
    args = -jnp.ones((info.arity,), dtype=ssa.args.dtype)
    args = jax.lax.fori_loop(0, ssa.arity, step_fn, args)
    return args


def sample_inner_loop(
        key: jax.random.PRNGKey,
        info: RegistryInfo,
        ssa: SSA,
        node_id: int,
        beta1: float = 0.0,
        beta2: float = 0.0,
        n_keep: int = 100,
        op_id: int | None = None
) -> SSA:
    length = ssa.id.size

    key_args, key_op = jax.random.split(key)
    if op_id is None:
        op_id = sample_op(key_op, info, ssa)

    args = sample_args(key_args, info, ssa, node_id, op_id, beta1=beta1, beta2=beta2, n_keep=n_keep)

    depth = jnp.max((1 + ssa.depth[args]) * (args >= 0) - (args < 0)).astype(ssa.depth.dtype)

    return ssa.replace(
        op    = ssa.op.at[node_id]   .set(op_id),
        type  = ssa.type.at[node_id] .set(info.output_type[op_id].astype(ssa.type.dtype)),
        args  = ssa.args.at[node_id] .set(args),
        depth = ssa.depth.at[node_id].set(depth)
    )


def sample(
        key: jax.random.PRNGKey,
        jax_registry: RegistryInfo,
        ssa: SSA,
        beta1: float = 0.0,
        beta2: float = 0.0,
        n_keep: int = 100
) -> SSA:
    def body(node_id, state):
        key, ssa = state
        key, _key = jax.random.split(key)
        return key, sample_inner_loop(_key, jax_registry, ssa, node_id, beta1=beta1, beta2=beta2, n_keep=n_keep)

    start = jnp.argmin(ssa.type)  # first one with -1 -> first empty cell
    end = ssa.length

    key, ssa = jax.lax.fori_loop(
        start,
        end,
        body,
        (key, ssa)
    )
    return ssa


# ----- PRUNE -----
def prune(ssa: "SSA", output_id) -> "SSA":
    length = ssa.id.size
    max_depth = ssa.depth[output_id].max()
    

    def fn(rev_i, mask):
        i = length - 1 - rev_i
        args = ssa.args[i]
        cond = (args >= 0) & (mask[i] > 0)
        mask = mask.at[args].set(jnp.where(cond, 1, mask[args]))
        return mask

    mask = jnp.zeros_like(ssa.id).at[output_id].set(1)
    mask = jax.lax.fori_loop(0, length, fn, mask)
    mask = mask | ssa.inputs_mask
    mask = mask & (ssa.depth <= max_depth) & ((ssa.depth < max_depth) | jnp.isin(ssa.id, output_id))
    #
    id_to_new_id = jnp.cumsum(mask) - 1

    def fn(i, state):
        new_ssa, c = state

        args = ssa.args[i]
        new_ar = id_to_new_id[args] * (args >= 0) - (args < 0)

        new_ssa = jax.lax.cond(
            mask[i],
            lambda: new_ssa.replace(
                op=new_ssa.op.at[c].set(ssa.op[i]),
                type=new_ssa.type.at[c].set(ssa.type[i]),
                args=new_ssa.args.at[c].set(new_ar),
                depth=new_ssa.depth.at[c].set(ssa.depth[i])
            ),
            lambda: new_ssa
        )

        return new_ssa, c + mask[i]

    new_ssa = ssa.__class__.init_empty(ssa.length, ssa.arity)
    new_ssa, c = jax.lax.fori_loop(
        0, length,
        fn,
        (new_ssa, 0)
    )

    return new_ssa


# ----- NORMALIZATION -----
def order_by_depth(ssa: SSA):
    # order: depth, type, op, *args_id
    ordering = [
        (ssa.depth, 1),
        (ssa.type, ssa.type.max()),
        (ssa.op, ssa.op.max())
    ]
    for n in range(ssa.args.shape[-1]):
        ordering.append((ssa.args[:, n], ssa.length))
    v, norm = 1., 1
    for vi, n in ordering:
        norm = norm * (1 + n)
        v = v + vi / norm
    v = jnp.where(ssa.depth >= 0, v, 1 + v.max())
    order = jnp.argsort(v)
    inv_order = jnp.argsort(order)

    new_id = ssa.id
    new_op = ssa.op[order]
    new_ty = ssa.type[order]
    new_depth = ssa.depth[order]

    reordered_ar = ssa.args[order, :]
    new_ar = jnp.where(reordered_ar < 0, reordered_ar, inv_order[reordered_ar])

    new_ssa = SSA(
        id=new_id,
        op=new_op,
        type=new_ty,
        args=new_ar,
        depth=new_depth
    )
    return new_ssa


# ----- RUN -----
def run(registry: "DSLRegistry", ssa: "SSA", inputs: "UnionStruct"):
    n_inputs = inputs.dtype.shape[0]
    mem = jax.vmap(lambda _: registry.init_struct())(jnp.arange(ssa.length))
    mem = jax.tree.map(lambda m, v: m.at[:n_inputs].set(v), mem, inputs)

    def run_op(i, mem, op_id, args_id):
        var = tuple(jax.tree.map(lambda m: m[arg_id], mem) for arg_id in args_id)
        res = jax.lax.switch(op_id, registry.partials, *var)
        mem = jax.tree.map(lambda m, r: m.at[i].set(r), mem, res)
        return mem

    def skip_op(i, mem, _0, _1):
        return mem

    def step_fn(i, mem):
        cond = (mem.dtype[i] < 0) & (ssa.op[i] >= 0)
        mem = jax.lax.cond(
            cond,
            run_op,
            skip_op,
            i, mem, ssa.op[i], ssa.args[i]
        )
        return mem

    end = jnp.argmax((ssa.op < 0) & (ssa.type < 0))
    mem = jax.lax.fori_loop(
        n_inputs,
        end,
        step_fn,
        mem
    )

    return mem


def validate(info: "RegistryInfo", ssa: "SSA", *, do_print=False):
    def step_fn(i, valid):
        op = ssa.op[i]
        args = ssa.args[i]

        cond_args_back = (op < 0) | jnp.all(args < i)
        cond_args_num  = (op < 0) | (jnp.sum(args >= 0) == jnp.sum(info.inputs_type[op] >= 0))
        cond_args_type = (op < 0) | jnp.all((args < 0) | (ssa.type[args] == info.inputs_type[op]))
        cond_out_type  = (op < 0) | (ssa.type[i] == info.output_type[op])

        cond = cond_args_back & cond_args_num & cond_args_type & cond_out_type
        valid = jnp.where(cond, valid, 0)

        if do_print:
            def print_():
                jax.debug.print(
                    "id {i} - args_back {cond_args_back} & args_num {cond_args_num} & args_type {cond_args_type} & out_type {cond_out_type}",
                    i=i,
                    cond_args_back=cond_args_back,
                    cond_args_num=cond_args_num,
                    cond_args_type=cond_args_type,
                    cond_out_type=cond_out_type
                )
                return ()
            jax.lax.cond(
                cond,
                lambda: (),
                lambda: print_()
            )
        
        return valid

    
    valid = jax.lax.fori_loop(0, ssa.length, step_fn, 1)
    return valid


r = DSLRegistry()

# ----- DEFINE HERE YOUR TYPES (with prototypes for struct, shape and dtype) -----
r.register_type("GRID", prototype=jnp.zeros((8, 8), dtype=jax.numpy.int8))
r.register_type("COLOR", prototype=jnp.zeros((), dtype=jax.numpy.int8))


# ----- DEFINE HERE YOUR FUNCTIONS -----
class Rotate(JaxFunction):
    def __init__(self):
        super().__init__(("GRID",), "GRID")

    def _call(self, x):
        return jnp.rot90(x)


class SelectColor(JaxFunction):
    def __init__(self):
        super().__init__(("GRID", "COLOR"), "GRID")

    def _call(self, x, c):
        return x * (x == c)


class GridUnion(JaxFunction):
    def __init__(self):
        super().__init__(("GRID", "GRID"), "GRID")

    def _call(self, x, y):
        # apply x, then apply y where the is not x, then remove the mask where both are masked
        return x * (x > 0) + y * (x <= 0) * (y > 0) - (x < 0) * (y < 0)


class ReplaceColor(JaxFunction):
    def __init__(self):
        super().__init__(("GRID", "COLOR", "COLOR"), "GRID")

    def _call(self, x, c0, c1):
        return jnp.where(x == c0, c1, x)


# ----- REGISTER THE FUNCTIONS -----
r.register_func(Rotate())
r.register_func(GridUnion())
r.register_func(SelectColor())
r.register_func(ReplaceColor())


train = json.load(open("/kaggle/input/arc-prize-2025/arc-agi_training_challenges.json"))
task_name = list(train.keys())[9]
x = jnp.array(train[task_name]["train"][0]["input"])[:8, :8]
x = jnp.pad(x, ((0, 8 - x.shape[0]), (0, 8 - x.shape[1])), constant_values=-1)


inputs = r.fill_struct([
    # (type_id, value of type_id)
    (r.get_type_id("GRID"),  x),
] + [(r.get_type_id("COLOR"), c) for c in range(10)])
inputs.dtype


def sample_and_run(k, reg_info, ssa_init, inputs):
    k, k0 = jax.random.split(k)
    ssa = sample(
        k,
        reg_info,
        ssa_init,
        n_keep=5
    )

    # select as output of our transformation one GRID on the latest depth
    target_type = r.get_type_id("GRID")
    mask = (ssa.type == target_type) & (ssa.depth > 0)
    max_depth = (ssa.depth * mask).max()
    mask = mask & (ssa.depth == max_depth)
    p = mask
    out_id = jax.random.choice(k0, ssa.length, p=p)
    
    # prune the SSA on the selected output and normalize the order by (depth, type, op, *args)
    ssa = ssa.prune(out_id)
    ssa = ssa.normalize()
    
    return ssa, r.run(ssa, inputs)


# sample many SSA (Static-Single-Assignment) and run it on the input data
sample_and_run_many = jax.jit(jax.vmap(sample_and_run, (0, None, None, None)))

# the size will be the input size (we need one SSA line for each input) + a user defined program length
max_ssa_length = inputs.dtype.shape[0] + 16

reg_info = r.get_info()
ssa_init = SSA.init(max_ssa_length, reg_info.arity, input_types=inputs.dtype)

key = jax.random.PRNGKey(42)
ssas, memory = sample_and_run_many(jax.random.split(key, 1024), reg_info, ssa_init, inputs)

# validate all the SSAs (1 valid, 0 not valid)
jax.vmap(validate, (None, 0))(reg_info, ssas).min()


%%timeit -n 10 -r 10
ssas, memory = sample_and_run_many(jax.random.split(key, 1024), reg_info, ssa_init, inputs)
_ = memory.dtype.block_until_ready()


fig, ax = plt.subplots(2, 3, figsize=(10, 8))
plt.tight_layout()
n = 32

ax[0, 0].set_title("operation id")
ax[0, 0].matshow(ssas.op[:n])

ax[0, 1].set_title("output type")
ax[0, 1].matshow(ssas.type[:n])

for i in range(3):
    ax[1, i].set_title(f"arg[{i}] id")
    ax[1, i].matshow(ssas.args[:n, ..., i])

for axi in ax:
    for axij in axi:
        axij.axis("off")


# extract the outputs from memory
# memory.values is a tuple, we want the correct dtype (we expect always GRID, but let's check)
# memory.values[GRID] is a (batch_size, num_steps) + GRID_SIZE
# the solution is memory.values[output_dtype_id][i, output_id]

inputs_ = [memory.values[memory.dtype[i, 0]][i, 0]                 for i, output_id in enumerate(ssas.output_id)]
outputs = [memory.values[memory.dtype[i, output_id]][i, output_id] for i, output_id in enumerate(ssas.output_id)]

cmap = colors.ListedColormap(
    ['#000000', '#0074D9', '#FF4136', '#2ECC40', '#FFDC00',
     '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25']
)
norm = colors.Normalize(vmin=0, vmax=9)

_, ax = plt.subplots(10, 2, figsize=(2, 10))
for i in range(10):
    ax[i, 0].imshow(inputs_[i], cmap=cmap, norm=norm)
    ax[i, 1].imshow(outputs[i], cmap=cmap, norm=norm)
    ax[i, 0].axis("off")
    ax[i, 1].axis("off")




