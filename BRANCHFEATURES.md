# New features

This document describes the new features in this Caffe branch.

## New layers

### `WEIGHTED_SOFTMAX_LOSS`

Similar to `SOFTMAX_LOSS`, except it takes a third bottom input specifying the
importance of each sample, e.g.:

```
layers {
  name: "loss"
  type: WEIGHTED_SOFTMAX_LOSS
  bottom: "fc"
  bottom: "label"
  bottom: "sample_weight"
}  
```

The shape of `sample_weight` should be `(N, 1, 1, 1)` or simply `(N,)`, where
`N` is the number of samples.

Note that the HDF5 loader, unlike in earlier releases of Caffe, can now load
any number of inputs with any key. That way, you can add `sample_weight` (or
whatever you wish to name it) to your data file:

```
layers {
  name: "main"
  type: HDF5_DATA
  top: "data"
  top: "label"
  top: "sample_weight"
  hdf5_data_param {
    source: "/path/to/data.txt"  # File should contain an absolute path to h5 file
    batch_size: 100
  }
}
```
This assumes that the HDF5 file has an entry at `/sample_weight`. You can also
load it separately from its own HDF5 file. I have not tested it it with lmdb,
but I think it will work analogously.
