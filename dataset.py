  def as_numpy_iterator(self):
    """Returns an iterator which converts all elements of the dataset to numpy.

    Use `as_numpy_iterator` to inspect the content of your dataset. To see
    element shapes and types, print dataset elements directly instead of using
    `as_numpy_iterator`.

    >>> dataset = tf.data.Dataset.from_tensor_slices([1, 2, 3])
    >>> for element in dataset:
    ...   print(element)
    tf.Tensor(1, shape=(), dtype=int32)
    tf.Tensor(2, shape=(), dtype=int32)
    tf.Tensor(3, shape=(), dtype=int32)

    This method requires that you are running in eager mode and the dataset's
    element_spec contains only `TensorSpec` components.

    >>> dataset = tf.data.Dataset.from_tensor_slices([1, 2, 3])
    >>> for element in dataset.as_numpy_iterator():
    ...   print(element)
    1
    2
    3

    >>> dataset = tf.data.Dataset.from_tensor_slices([1, 2, 3])
    >>> print(list(dataset.as_numpy_iterator()))
    [1, 2, 3]

    `as_numpy_iterator()` will preserve the nested structure of dataset
    elements.

    >>> dataset = tf.data.Dataset.from_tensor_slices({'a': ([1, 2], [3, 4]),
    ...                                               'b': [5, 6]})
    >>> list(dataset.as_numpy_iterator()) == [{'a': (1, 3), 'b': 5},
    ...                                       {'a': (2, 4), 'b': 6}]
    True

    Returns:
      An iterable over the elements of the dataset, with their tensors converted
      to numpy arrays.

    Raises:
      TypeError: if an element contains a non-`Tensor` value.
      RuntimeError: if eager execution is not enabled.
    """
    if not context.executing_eagerly():
      raise RuntimeError("`tf.data.Dataset.as_numpy_iterator()` is only "
                         "supported in eager mode.")
    for component_spec in nest.flatten(self.element_spec):
      if not isinstance(
          component_spec,
          (tensor_spec.TensorSpec, ragged_tensor.RaggedTensorSpec,
           sparse_tensor_lib.SparseTensorSpec, structure.NoneTensorSpec)):
        raise TypeError(
            f"`tf.data.Dataset.as_numpy_iterator()` is not supported for "
            f"datasets that produce values of type {component_spec.value_type}")

    return _NumpyIterator(self)