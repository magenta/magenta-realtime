# Installing live coding extensions

## MaxMSP

### Install

Copy `mrt2~.mxo` into a directory Max searches. The simplest place is
`~/Documents/Max 9/Library/` (or `Max 8` if you're on the older version):

```sh
ditto max/mrt2~.mxo "$HOME/Documents/Max 9/Library/mrt2~.mxo"
```

Open `max/mrt2~.maxhelp` for example usage.


## PureData

### Install

Pd looks for externals in its search paths (Preferences → Path). The simplest
install puts the external + its metallib + help patch into a `mrt2~/`
subdirectory under your user externals folder:

```sh
ditto pd "$HOME/Documents/Pd/externals/mrt2~"
cp examples/pd/mrt2~-help.pd "$HOME/Documents/Pd/externals/mrt2~/mrt2~-help.pd"
```

Open `pd/mrt2~-help.pd` for example usage

## SuperCollider

### Install

SuperCollider scans `~/Library/Application Support/SuperCollider/Extensions`
recursively at server boot. The simplest install drops the binary + metallib +
sclang class + example into a `MRT2/` subdirectory there:

```sh
ditto sc "$HOME/Library/Application Support/SuperCollider/Extensions/MRT2"
cp examples/sc/MRT2.sc examples/sc/example.scd "$HOME/Library/Application Support/SuperCollider/Extensions/MRT2/"
```

Open `sc/example.scd` for example usage.

After installing, recompile the sclang class library (Language → Recompile
Class Library, or `Cmd-Shift-L`). `mlx.metallib` must remain next to
`MRT2.scx` — MLX uses `dladdr` to locate its kernels at runtime.
