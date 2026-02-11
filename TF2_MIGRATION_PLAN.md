# TensorFlow 2.x Migration Plan
**Status:** Planning Phase
**Priority:** HIGH (TF 1.x is deprecated)
**Complexity:** HIGH
**Risk:** MEDIUM-HIGH (breaks training if done incorrectly)

---

## Current State Analysis

### TensorFlow 1.x Usage Found

**Files with TF 1.x code:**
1. `policy_opt_tf.py` - Main policy optimization (Session-based)
2. `tf_policy.py` - Policy wrapper (Session-based)
3. `tf_model_example.py` - Network architecture (placeholder-based)
4. `tf_utils.py` - TensorFlow utilities

**TF 1.x APIs in use:**
- `tf.Session()` - 2 occurrences
- `tf.placeholder()` - Multiple occurrences
- `tf.Variable()` - Legacy style
- `tf.global_variables_initializer()` - Deprecated
- `tf.train.Saver()` - Legacy checkpointing
- `tf.gradients()` - Works in TF 2.x but not idiomatic

---

## Migration Strategy: Phased Approach

### Phase 1: Compatibility Mode (Week 1) ✅ RECOMMENDED FIRST
**Add TF 2.x compatibility without breaking existing code**

```python
# Add to all TF files
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
```

**Benefits:**
- Zero breaking changes
- Runs on TF 2.x with v1 compatibility
- Buys time for proper migration
- Can be deployed immediately

**Risks:** Low
- Technical debt persists
- Not using TF 2.x features
- Performance not optimal

---

### Phase 2: Eager Execution Migration (Weeks 2-4)
**Migrate from Session-based to eager execution**

#### Key Changes:

**Before (TF 1.x):**
```python
self.sess = tf.Session()
init_op = tf.global_variables_initializer()
self.sess.run(init_op)

# Training
feed_dict = {self.obs_tensor: obs, self.action_tensor: actions}
loss = self.sess.run(self.loss_scalar, feed_dict=feed_dict)
```

**After (TF 2.x):**
```python
# No session needed - eager execution by default

# Training
@tf.function  # JIT compilation
def train_step(obs, actions):
    with tf.GradientTape() as tape:
        predictions = model(obs, training=True)
        loss = loss_fn(actions, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

loss = train_step(obs, actions)
```

---

### Phase 3: Model Refactoring (Weeks 4-6)
**Migrate to tf.keras.Model**

#### Current Architecture:
- Custom network building with placeholders
- Manual variable management
- Custom training loops

#### Target Architecture:
```python
class PolicyNetwork(tf.keras.Model):
    def __init__(self, dim_input, dim_output):
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(128, activation='relu')
        self.dense2 = tf.keras.layers.Dense(128, activation='relu')
        self.output_layer = tf.keras.layers.Dense(dim_output)

    def call(self, inputs, training=False):
        x = self.dense1(inputs)
        x = self.dense2(x)
        return self.output_layer(x)
```

**Benefits:**
- Cleaner code
- Built-in serialization
- Better debugging
- TensorBoard integration
- Mixed precision training

---

### Phase 4: Optimizer Migration (Week 6)
**Update optimizers to tf.keras.optimizers**

**Before:**
```python
self.solver = TfSolver(loss_scalar=self.loss_scalar, ...)
```

**After:**
```python
self.optimizer = tf.keras.optimizers.Adam(
    learning_rate=learning_rate,
    beta_1=momentum,
    weight_decay=weight_decay
)
```

---

### Phase 5: Checkpointing Migration (Week 7)
**Update from tf.train.Saver to tf.train.Checkpoint**

**Before:**
```python
self.saver = tf.train.Saver()
self.saver.save(self.sess, checkpoint_path)
```

**After:**
```python
self.checkpoint = tf.train.Checkpoint(
    optimizer=self.optimizer,
    model=self.model
)
self.checkpoint_manager = tf.train.CheckpointManager(
    self.checkpoint, directory='./checkpoints', max_to_keep=3
)
self.checkpoint_manager.save()
```

---

## Detailed File-by-File Migration

### 1. policy_opt_tf.py

**Current Issues:**
- Line 31: `tf.set_random_seed()` → `tf.random.set_seed()`
- Line 50: `tf.Session()` → Remove (eager execution)
- Line 64: `tf.global_variables_initializer()` → Not needed in TF 2.x
- Line 65: `self.sess.run()` → Direct execution
- Line 95: `tf.train.Saver()` → `tf.train.Checkpoint()`

**Migration Complexity:** HIGH
- Core policy optimization logic
- Affects all training loops
- Requires careful validation

**Estimated Effort:** 3-4 days

---

### 2. tf_policy.py

**Current Issues:**
- Line 118: `tf.Session()` → Remove
- Line 38: `tf.placeholder()` → Function arguments
- Custom variable copying logic

**Migration Complexity:** MEDIUM
- Simpler than policy_opt_tf.py
- Mainly wrapper code

**Estimated Effort:** 1-2 days

---

### 3. tf_model_example.py

**Current Issues:**
- Lines 33-35: Multiple `tf.placeholder()` calls
- Line 280: Legacy `tf.Variable()` initialization
- Custom network building (not using Keras layers)

**Migration Complexity:** MEDIUM
- Can be rewritten with tf.keras.layers
- Cleaner in TF 2.x actually

**Estimated Effort:** 2 days

---

### 4. tf_utils.py

**Current Issues:**
- TfSolver class uses legacy optimizer APIs
- Manual gradient application

**Migration Complexity:** MEDIUM
- Mostly utility code
- Can leverage tf.keras.optimizers

**Estimated Effort:** 1 day

---

## Risk Assessment

### High-Risk Areas

**1. Training Loop Changes**
- **Risk:** Numerical differences in gradients
- **Mitigation:** Side-by-side comparison with TF 1.x
- **Validation:** Compare loss curves over 1000 iterations

**2. Gradient Computation**
- **Risk:** tf.GradientTape behaves differently
- **Mitigation:** Unit tests for gradient computation
- **Validation:** Gradient norm comparisons

**3. Checkpointing**
- **Risk:** Incompatible checkpoint formats
- **Mitigation:** Keep TF 1.x checkpoints, provide conversion script
- **Validation:** Load old checkpoints and verify weights

**4. Performance**
- **Risk:** Training might be slower/faster
- **Mitigation:** Profile both versions
- **Validation:** Benchmark on sample experiments

---

## Testing Strategy

### Unit Tests
```python
def test_policy_forward_pass():
    """Test that forward pass gives same results."""
    # TF 1.x version
    policy_v1 = PolicyOptTfV1(...)
    output_v1 = policy_v1.forward(test_input)

    # TF 2.x version
    policy_v2 = PolicyOptTfV2(...)
    output_v2 = policy_v2.forward(test_input)

    np.testing.assert_allclose(output_v1, output_v2, rtol=1e-5)
```

### Integration Tests
- Run 10 iterations of training with both versions
- Compare:
  - Loss values (should match within 1e-5)
  - Gradient magnitudes
  - Policy parameters
  - Sample trajectories

### System Tests
- Run full experiment (50 iterations)
- Compare final policy performance
- Ensure training time within 20% of baseline

---

## Backward Compatibility

### Checkpoint Conversion Tool
```python
def convert_tf1_checkpoint_to_tf2(tf1_path, tf2_path):
    """Convert TF 1.x checkpoint to TF 2.x format."""
    # Load TF 1.x checkpoint
    reader = tf.train.load_checkpoint(tf1_path)

    # Create TF 2.x model
    model = PolicyNetwork(...)

    # Map variables
    for var in model.trainable_variables:
        tf1_name = map_variable_name(var.name)
        value = reader.get_tensor(tf1_name)
        var.assign(value)

    # Save as TF 2.x checkpoint
    checkpoint = tf.train.Checkpoint(model=model)
    checkpoint.save(tf2_path)
```

---

## Performance Optimization (TF 2.x)

### 1. Use @tf.function Decorator
```python
@tf.function
def train_step(obs, actions):
    # JIT compiled, much faster
    ...
```

**Benefits:**
- 2-3x speedup on training loops
- Graph optimization
- XLA compilation option

### 2. Mixed Precision Training
```python
from tensorflow.keras import mixed_precision
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)
```

**Benefits:**
- 2x faster training on modern GPUs
- Reduced memory usage
- Same accuracy with proper scaling

### 3. Data Pipeline Optimization
```python
dataset = tf.data.Dataset.from_tensor_slices((obs, actions))
dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
```

**Benefits:**
- Overlaps data loading with training
- Better GPU utilization

---

## Timeline & Milestones

| Week | Phase | Deliverable | Risk |
|------|-------|-------------|------|
| 1 | Compatibility Mode | TF 2.x with v1 compat | LOW |
| 2-3 | Remove Sessions | Eager execution | MEDIUM |
| 4-5 | Keras Models | Model refactoring | MEDIUM |
| 6 | Optimizers | Keras optimizers | LOW |
| 7 | Checkpointing | TF 2.x checkpoints | LOW |
| 8-9 | Testing | Full validation | HIGH |
| 10 | Optimization | Performance tuning | LOW |

**Total Time:** 10 weeks
**Critical Path:** Eager execution migration (Weeks 2-3)

---

## Recommended Approach

### Option A: Full Migration (10 weeks)
**Pros:**
- Modern TF 2.x codebase
- Better performance
- Easier maintenance
- No technical debt

**Cons:**
- High effort
- Risk of bugs
- Extensive testing needed

### Option B: Compatibility Mode (1 day) ✅ RECOMMENDED
**Pros:**
- Works immediately
- Zero breaking changes
- Low risk
- Can migrate later

**Cons:**
- Technical debt
- Not using TF 2.x features
- Performance not optimal

### Option C: Hybrid (5 weeks)
**Pros:**
- Gradual migration
- Lower risk
- Can deploy incrementally

**Cons:**
- Maintain two codepaths
- Complexity during transition

---

## Decision: Start with Option B

**Immediate Action:** Add compatibility mode
**Long-term Goal:** Full migration (Option A)
**Rationale:**
- Get working code on TF 2.x NOW
- Plan full migration for dedicated sprint
- Reduce risk of breaking experiments

---

## Next Steps

1. ✅ **This Week:** Add TF 2.x compatibility imports
2. ⏳ **Next Sprint:** Begin eager execution migration
3. ⏳ **Testing:** Set up TF 1.x vs TF 2.x comparison framework
4. ⏳ **Documentation:** Update user guide for TF 2.x

---

**Author:** Claude (Senior Research Scientist Agent)
**Date:** 2026-02-10
**Status:** Ready for Implementation
**Approval Required:** Yes (before starting migration)
