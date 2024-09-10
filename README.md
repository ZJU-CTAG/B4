# B4: Towards Optimal Assessment of Plausible Code Solutions with Plausible Tests

We provide the replication package for our paper: "B4: Towards Optimal Assessment of Plausible Code Solutions with Plausible Tests".

### Run simulated experiments

```bash
python simulated.py
```

- You can change the arguments in this file.

### Run real-world experiments

First, please download `data.zip` from https://zenodo.org/records/13737381 and unzip it to `data/` folder. All the valid config names can be found in `data/config.json`. 

Second, run the following command:

```bash
python main.py --config_name codegen_humaneval --max_px 1.0 --beta_0_range 1e4,1e5,1e6 --alpha_xy_range 1e3
```

This will run CodeGen on HumanEval. Feel free to add new configuration in `data/config.json`.

### Citation

TODO

