from numpy import array, ndarray
from qutip import Qobj


def np_to_Qobj(
    obj: ndarray | Qobj, hilbert_dim: int, isket: bool = False
) -> Qobj | list[Qobj]:
    """Convert a numpy array to a (nested) list of Qobjs."""
    # If already a Qobj, return as is
    if isinstance(obj, Qobj):
        return obj

    if isket:
        dims = [[hilbert_dim], [1]]
        min_dim = 1
    else:
        dims = [[hilbert_dim], [hilbert_dim]]
        min_dim = 2

    # A single object
    if len(obj.shape) == 1:
        if isket:
            return Qobj(obj, dims=dims).unit().tidyup()
        return Qobj(obj, dims=dims).tidyup()

    # Recursively return a list (of lists) of Qobjs
    if len(obj.shape) > 1:
        return [np_to_Qobj(sub_obj, hilbert_dim, isket) for sub_obj in obj]

    # If we reach here, the input is invalid
    raise ValueError(f"When isket=={isket}, obj can only have dimension >= {min_dim}.")


def Qobj_to_np(
    obj: Qobj | list[Qobj] | ndarray, hilbert_dim: int, isket: bool = False
) -> ndarray:
    """Convert a (nested) list of Qobjs to a numpy array."""
    # If already a numpy array, return as is
    # (unless dtype=object, i.e. numpy array of Qobjs)
    if isinstance(obj, ndarray):
        if obj.dtype == object:
            obj = obj.tolist()
        else:
            return obj

    # A single object
    if isinstance(obj, Qobj):
        if isket:
            return array(obj.unit().tidyup().full(squeeze=True), dtype=complex)
        return array(obj.tidyup().full(squeeze=True), dtype=complex)

    # Recursively return a numpy array with multiple axes
    if isinstance(obj, list):
        return array([Qobj_to_np(sub_obj, hilbert_dim, isket) for sub_obj in obj])

    # If we reach here, the input is invalid
    raise TypeError("obj can only be an instance of qutip.Qobj or a list of Qobj.")
