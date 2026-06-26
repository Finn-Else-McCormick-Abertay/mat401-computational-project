import re
from typing import Dict, List, Optional
from warnings import warn

import matplotlib.pyplot as plt
from numpy import number
from numpy.typing import NDArray

from type_checking import type_matches


class DataSet[NumT = number, KeyT = str]:
    type ValT = NumT | NDArray[NumT] # type: ignore

    type DataPoint = Dict[KeyT, ValT]
    type DataList = List[DataPoint]
    type DataFlat = Dict[KeyT, List[ValT | None]]

    type Data = DataList | DataFlat | DataPoint

    _data_raw: DataList
    _data_flat: DataFlat
    
    _default_x_key: Optional[KeyT]

    @property
    def default_x_key(self) -> KeyT:
        if self._default_x_key is not None:
            #if self._default_x_key not in self._data_flat:
            #    warn("Default x key '%s' does not exist within the data." % self._default_x_key)
            return self._default_x_key
        else:
            first_key = next(iter(self._data_flat))
            warn("Graph has no default x key. Assuming it to be '%s' as it was the first key added." % first_key)
            return first_key

    @default_x_key.setter
    def default_x_key(self, x_key: Optional[KeyT]):
        self._default_x_key = x_key
        #if self._default_x_key is not None and self._default_x_key not in self._data_flat:
        #    warn("Default x key set to '%s', which does not exist within the data." % self._default_x_key)

    def __init__(self, data: Optional[Data] = None, x_key: Optional[KeyT] = None):
        self._data_raw = []
        self._data_flat = {}

        if data is not None:
            self.add_data(data)

        if x_key:
            self.default_x_key = x_key

    def add_data(self, data: Data, regenerate: bool = True):
        if type_matches(data, self.DataPoint):
            self._data_raw.append(data) # type: ignore
            self._data_raw.sort(key = lambda obj: obj[self.default_x_key]) # type: ignore
        elif type_matches(data, self.DataList):
            for data_point in data:
                self.add_data(data_point, False) # type: ignore
        elif type_matches(data, self.DataFlat):
            length = -1
            for key in data:
                new_length = len(data[key]) # type: ignore
                if length > 0 and length != new_length:
                    raise ValueError("Flat data does not contain equal length arrays.")
                length = new_length

            for i in range(length):
                data_point = {}
                for key in data:
                    data_point[key] = data[key][i] # type: ignore
                self.add_data(data_point, False)
        else:
            raise ValueError("add_data: data was unsupported type %s" % type(data).__name__)

        if regenerate:
            self._regenerate_flat_form()

    def _regenerate_flat_form(self):
        self._data_flat.clear()
        keys: Dict[KeyT, List[KeyT] | None] = {}
        for data_point in self._data_raw:
            for key in data_point:
                if type_matches(data_point[key], NDArray):
                    prefix: KeyT = "%s." % key if key != "__unnamed__" else ""
                    subkeys: List[KeyT] = []

                    named_indices = ["x", "y", "z", "w"]

                    if len(data_point[key]) <= len(named_indices): # type: ignore
                        for i in range(len(data_point[key])): # type: ignore
                            subkeys.append("{0}{1}".format(prefix, named_indices[i]))
                    else:
                        for i in range(len(data_point[key])): # type: ignore
                            subkeys.append("{0}{1}".format(prefix, i))
                    
                    keys[key] = subkeys
                    for subkey in subkeys:
                        self._data_flat[subkey] = []
                else:
                    keys[key] = None
                    self._data_flat[key] = []
        
        for data_index in range(len(self._data_raw)):
            for key, subkeys in keys.items(): # type: ignore
                raw_value = None
                if key in self._data_raw[data_index]:
                    raw_value = self._data_raw[data_index][key]
                
                if subkeys is None:
                    self._data_flat[key].append(raw_value)
                else:
                    for i in range(len(subkeys)):
                        self._data_flat[subkeys[i]].append(raw_value[i] if raw_value is not None else None) # type: ignore
    
    def clear_data(self):
        self._data_raw.clear()
        self._data_flat.clear()


def multiplot(
    data: DataSet, x_key=None, y_keys: List | None=None, colors=None,
):
    if x_key is None:
        x_key = data.default_x_key

    if y_keys is None:
        y_keys = []
        for key in data._data_flat:
            if key != x_key:
                y_keys.append(key)

    colors_final = {
        "default": "black",
        "(\\w+(_|\\s|\\.)?)?x(?!\\w)": "red",
        "(\\w+(_|\\s|\\.)?)?y(?!\\w)": "green",
        "(\\w+(_|\\s|\\.)?)?z(?!\\w)": "blue",
        "(\\w+(_|\\s|\\.)?)?(speed|spd|mag|magnitude)(?!\\w)": "purple"
    }
    if colors is not None:
        for key, value in colors.items():
            colors_final[key] = value

    fig, axes = plt.subplots(len(y_keys), 1, constrained_layout=True)

    for i in range(len(y_keys)):
        axis = axes[i] if len(y_keys) > 1 else axes
        y_key = y_keys[i]
        
        if y_key not in data._data_flat:
            raise Exception("y_key {0} is not contained in {1}".format(y_key, data._data_flat.keys()))

        color = colors_final["default"]
        for color_key in colors_final:
            if re.match(color_key, y_key) is not None:
                color = colors_final[color_key]

        axis.set_xlabel(x_key)
        axis.set_ylabel(y_key)
        axis.plot(data._data_flat[x_key], data._data_flat[y_key], color=color)

    return fig
