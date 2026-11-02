# Algorithm specification

The public default is the research **S2** sampler, renamed SMA at project level.

## 1. TRAIN-only structural distance

For each window collect frozen visual-language features `z`, end-effector
position `x`, unit quaternion `q`, two gripper positions, and an expert action
chunk with a valid-time mask. Fit medians and IQRs on TRAIN only, floor constant
channels, and clip standardized coordinates to [-10,10]. Quaternions are not
standardized.

The four squared blocks are:

- `d_v² = mean((z_i-z_j)²)`.
- `d_g² = 0.5 * [mean((x_i-x_j)²) + (2 acos(|q_i·q_j|) / pi)²]`.
- `d_h²`: MSE of the two gripper positions, first gripper action and valid-chunk
  mean gripper action.
- `d_a²`: MSE of standardized expert actions, averaged over seven coordinates
  and only timesteps valid in both windows.

`d(i,j) = sqrt((d_v² + d_g² + d_h² + d_a²) / 4)`.

The gripper-phase code combines the first action's sign and whether the sign
changes within the valid chunk. Incompatible codes have infinite distance. This
is an action-phase prior, not a grasp-success label. Expert future actions are
used for **TRAIN structure**, never for test-time inputs.

## 2. Regions and diffusion

Within each task, take up to four candidate positions per trajectory, ensuring
all phases are represented. Seed one center per phase and use farthest-point
selection up to 32 centers. Remove empty regions and merge regions with fewer
than two source groups when a compatible center exists. An isolated region
without enough support remains uniformly accessible but priority-ineligible.

Connect up to four directed nearest neighbors before symmetrization. With
`s = median(finite candidate edge distances)`, retain distances `<=2s` and set
`W_ij=exp(-(d_ij/s)²)`. No edges cross tasks. Let

`L = diag(nonzero_degree) - D^(-1/2) W D^(-1/2)` and `K = exp(-L)`.

Isolated nodes have zero Laplacian diagonal and kernel self-value one.
The deterministic representative ordering retains the original hash salt for
reference compatibility. It has no dataset- or machine-specific information.

## 3. A feasible coverage target

Let `u` be the task-balanced, episode-balanced, time-uniform distribution over
windows. It is **not** uniform over all dataset windows. Aggregate it into
regions to obtain `u_region`. Let `q` allocate equal mass per task and then equal
mass to that task's priority-eligible regions.

Because half of every batch is mandatory uniform sampling, the feasible target is

`p* = 0.5 u_region + 0.5 q`.

Track the realized distribution `p_hat` with an exposure-based EMA and use energy
`E(p_hat) = (p_hat-p*)^T K (p_hat-p*)`.
For each region evaluate the exact marginal reduction after moving a fraction
`eta=1/2048` toward that region. Convert the marginal into a bounded coverage
coefficient in [0.5,2], scaled to have q-weighted mean one.

## 4. Residual and task allocation

Compute one initial normalized score per representative window with the frozen
starting model and fixed stochastic probe settings. Keep an observation-count
EMA of current per-window scores in each region: `alpha=1-exp(-n/16)`.

`r_c = clip(EMA_loss_c / max(initial_loss_c, epsilon), 0.5, 2)`

`epsilon=max(0.1 median(initial_loss),1e-12)`. Unseen regions or regions whose
score is older than 180 fit seconds use `r=1`.

Priority mass before task caps is `q_c * r_c * coverage_c`. Project task masses
onto a capped simplex, then preserve relative region priorities inside a task.
With ten tasks, each priority task is capped at0.3. The final 50/50 mixture gives
each task an **expected marginal** share between5% and20%; these are not hard
per-batch task-count bounds. For other task counts, the cap is
`max(0.3,1/number_of_tasks)`.

## 5. Sampling and feedback

Draw half the global batch from `u`. For each priority sample, draw a region,
then uniformly an episode represented in that region, then uniformly one of
that episode's windows in the region. Shuffle both halves. Report the shuffled
slot's mixture probability; fixed-half sampling does not make the batch IID.

Feed detached **already-normalized per-window** scores back to the sampler.
`valid_denominator` verifies a usable mask; `observe` does not divide scores a
second time and averages normalized scores within each region. Coverage updates
from actual consumed windows, not from priority probabilities or merely queued
tickets: `alpha=1-exp(-batch_size/2048)`.

SMA does not inverse-probability-weight the native loss. Such weighting would
change the intended training distribution and would be a different method.

## Complexity and costs

If a task has M candidate windows, center selection materializes an M×M distance
matrix. With R total regions, the kernel costs O(R³) setup and O(R²) storage/work
for energy operations. The default at most32 regions/task is deliberately small.
Frozen model encoding and initial score probes are additional preparation work.
The online path reuses native per-window residuals; an adapter unable to expose
them may require extra work, which must be measured separately.
