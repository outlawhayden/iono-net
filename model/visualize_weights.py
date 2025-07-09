import pickle
import jax.numpy as jnp
import matplotlib.pyplot as plt

# === Load the weights ===
with open("/home/houtlaw/iono-net/model/model_weights_color_smaller_20250609_071842.pkl", "rb") as f:
    params = pickle.load(f)

# === Flatten parameters and collect magnitudes ===
def extract_weight_magnitudes(params):
    magnitudes = []
    names = []

    def collect_fn(path, leaf):
        if 'kernel' in path[-1]:
            flat = jnp.ravel(leaf)
            magnitudes.append(jnp.abs(flat))
            names.append("/".join(path))

    def traverse(pytree, path=[]):
        if isinstance(pytree, dict):
            for k, v in pytree.items():
                yield from traverse(v, path + [k])
        else:
            yield path, pytree

    print("Params:")
    for path, val in traverse(params):
        print("/".join(path), val.shape)
        collect_fn(path, val)

    return names, magnitudes

names, mags = extract_weight_magnitudes(params)

# === Plot all in one PNG ===
fig, axes = plt.subplots(len(names), 1, figsize=(10, 3 * len(names)))
if len(names) == 1:
    axes = [axes]  # make iterable if single plot

for ax, name, mag in zip(axes, names, mags):
    ax.hist(mag, bins=100)
    ax.set_title(f"Weight Magnitudes: {name}")
    ax.set_xlabel("Magnitude")
    ax.set_ylabel("Count")
    ax.grid(True)

plt.tight_layout()
output_file = "weight_magnitude_histograms.png"
plt.savefig(output_file, dpi=300)
plt.close()
print(f"Saved weight magnitude histograms to '{output_file}'")
