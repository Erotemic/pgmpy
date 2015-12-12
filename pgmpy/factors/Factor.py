from __future__ import division

from itertools import product

from collections import namedtuple

import numpy as np

from pgmpy.extern import tabulate
from pgmpy.extern import six
from pgmpy.extern.six.moves import map, range, reduce, zip


State = namedtuple('State', ['var', 'state'])

#PY_TABLEFMT = 'fancy_grid' if six.PY3 else 'psql'
PY_TABLEFMT = 'psql'


class Factor(object):
    """
    Base class for Factor.

    Public Methods
    --------------
    assignment(index)
    get_cardinality(variable)
    marginalize([variable_list])
    normalize()
    product(*Factor)
    reduce([variable_values_list])
    """

    def __init__(self, variables, cardinality, values, statename_dict=None):
        """
        Initialize a factor class.

        Defined above, we have the following mapping from variable
        assignments to the index of the row vector in the value field:

        +-----+-----+-----+-------------------+
        |  x1 |  x2 |  x3 |    phi(x1, x2, x2)|
        +-----+-----+-----+-------------------+
        | x1_0| x2_0| x3_0|     phi.value(0)  |
        +-----+-----+-----+-------------------+
        | x1_0| x2_0| x3_1|     phi.value(1)  |
        +-----+-----+-----+-------------------+
        | x1_0| x2_1| x3_0|     phi.value(2)  |
        +-----+-----+-----+-------------------+
        | x1_0| x2_1| x3_1|     phi.value(3)  |
        +-----+-----+-----+-------------------+
        | x1_1| x2_0| x3_0|     phi.value(4)  |
        +-----+-----+-----+-------------------+
        | x1_1| x2_0| x3_1|     phi.value(5)  |
        +-----+-----+-----+-------------------+
        | x1_1| x2_1| x3_0|     phi.value(6)  |
        +-----+-----+-----+-------------------+
        | x1_1| x2_1| x3_1|     phi.value(7)  |
        +-----+-----+-----+-------------------+

        Parameters
        ----------
        variables: list, array-like
            List of variables in the scope of the factor.

        cardinality: list, array_like
            List of cardinalities of each variable. `cardinality` array must have a value
            corresponding to each variable in `variables`.

        values: list, array_like
            List of values of factor.
            A Factor's values are stored in a row vector in the value
            using an ordering such that the left-most variables as defined in
            `variables` cycle through their values the fastest.

        statename_dict: dict
            dictionary that maps from variable names to the list of potential state names

        Examples
        --------
        >>> from pgmpy.factors import Factor
        >>> phi = Factor(['x1', 'x2', 'x3'], [2, 2, 2], np.ones(8))
        <Factor representing phi(x1:2, x2:2, x3:2) at 0x7f8188fcaa90>
        >>> from pgmpy.factors import Factor
        >>> statename_dict = {
        >>>     'x2': ['b1', 'b2', 'b3'],
        >>>     'x3': ['c1', 'c2'],
        >>>     'x4': ['d1', 'd2'],
        >>> }
        >>> statename_dict = statename_dict.copy()
        >>> #phi2 = Factor(['x3', 'x4', 'x1'], , range(8))
        >>> variables = ['x3', 'x4', 'x1']
        >>> cardinality = [2, 2, 2]
        >>> #cardinality = None
        >>> values = range(8)
        >>> phi2 = Factor(variables, cardinality, values, statename_dict)
        """
        self.cardinality = None
        self.statename_dict = None
        self.values = None
        self.variables = None

        values = np.array(values)

        if values.dtype != int and values.dtype != float:
            raise TypeError("Values: Expected type int or type float, got ", values.dtype)

        self.variables = list(variables)
        self.statename_dict = statename_dict
        self.cardinality = cardinality

        if self.statename_dict is not None:
            try:
                _implicitcard = list(map(len, self.statenames))
            except KeyError as ex:
                raise ValueError(
                     'Variable %s does not have cardinality or a value in statename_dict' % (ex,))
            if self.cardinality is None:
                self.cardinality = _implicitcard

            if self.cardinality is not None:
                if not np.all(_implicitcard == self.cardinality):
                    raise ValueError('cardinality does not agree with state names')

        if self.cardinality is None:
            assert self.cardinality is not None, 'cannot infer cardinality'

        self.cardinality = np.array(self.cardinality, dtype=int)
        self.values = values.reshape(self.cardinality)

        if len(self.cardinality) != len(self.variables):
            raise ValueError("Number of elements in cardinality must be equal to number of variables")

        if self.values.size != np.product(self.cardinality):
            raise ValueError("Values array must be of size: {size}".format(size=np.product(self.cardinality)))

    @property
    def statenames(self):
        """
        property alias
        """
        return self._statenames()

    def _internal_varindex(self, variable, statename):
        """
        maps from an external statename of a variable to its internal index
        """
        # TODO: build mapping instead
        var_idx = self.statename_dict[variable].index(statename)
        return var_idx

    def __getitem__(self, index_or_statename):
        """
        Returns the value from the first variable of this factor with the given
        statename.
        """
        if isinstance(index_or_statename, int):
            var_idx = index_or_statename
        else:
            var_idx = self._internal_varindex(self.variables[0], index_or_statename)
        return self.values[var_idx]

    def _row_labels(self, asindex=False):
        """
        Returns a list of statments corresponding to each value (in values.ravel())
        """
        row_labels = list(product(*self._statenames(asindex=asindex)))
        return row_labels

    def _statenames(self, variables=None, cardinality=None, asindex=False):
        """
        Returns the list of statenames associated with each row in this factor

        Parameters
        ----------
        variables: defaults to all variables
        cardinality: needs to be specified only if statename_dict has not been assigned

        Examples
        --------
        >>> from pgmpy.factors import Factor
        >>> statename_dict = {'grade': ['A', 'F'],
        >>>                   'intel': ['smart', 'dumb'],}
        >>> phi = Factor(['grade', 'intel'], None, [.9, .1, .1, .9], statename_dict)
        >>> phi._statenames()
        [['A', 'F'], ['smart', 'dumb']]
        """
        if variables is None and cardinality is None:
            variables = self.variables
            cardinality = self.cardinality
        if asindex:
            return [list(range(card))
                    for var, card in zip(variables, cardinality)]

        if self.statename_dict is None:
            # Old approach
            return [['{var}_{i}'.format(var=var, i=i) for i in range(card)]
                    for var, card in zip(variables, cardinality)]
        elif self.cardinality is None:
            # New approach
            return [self.statename_dict[var] for var in variables]
        else:
            # Hybrid aproach
            statename_lists = [
                (
                    self.statename_dict[var]
                    if self.statename_dict.get(var, None) is not None else
                    ['{var}_{i}'.format(var=var, i=i) for i in range(card)]
                )
                for var, card in zip(variables, cardinality)
            ]
            return statename_lists

    def scope(self):
        """
        Returns the scope of the factor.

        Returns
        -------
        list: List of variable names in the scope of the factor.

        Examples
        --------
        >>> from pgmpy.factors import Factor
        >>> phi = Factor(['x1', 'x2', 'x3'], [2, 3, 2], np.ones(12))
        >>> phi.scope()
        ['x1', 'x2', 'x3']
        """
        return self.variables

    def get_cardinality(self, variables):
        """
        Returns cardinality of a given variable

        Parameters
        ----------
        variables: list, array-like
                A list of variable names.

        Returns
        -------
        dict: Dictionary of the form {variable: variable_cardinality}

        Examples
        --------
        >>> from pgmpy.factors import Factor
        >>> phi = Factor(['x1', 'x2', 'x3'], [2, 3, 2], range(12))
        >>> phi.get_cardinality(['x1'])
        {'x1': 2}
        >>> phi.get_cardinality(['x1', 'x2'])
        {'x1': 2, 'x2': 3}
        """
        if isinstance(variables, six.string_types):
            raise TypeError("variables: Expected type list or array-like, got type str")

        if not all([var in self.variables for var in variables]):
            raise ValueError("Variable not in scope")

        return {var: self.cardinality[self.variables.index(var)] for var in variables}

    def assignment(self, index):
        """
        Returns a list of variable assignments for the corresponding row index.

        Parameters
        ----------
        index: list, array-like
            List of indices whose assignment is to be computed

        Returns
        -------
        list: Returns a list of full assignments of all the variables of the factor.

        Examples
        --------
        >>> import numpy as np
        >>> from pgmpy.factors import Factor
        >>> phi = Factor(['diff', 'intel'], [2, 2], np.ones(4))
        >>> phi.assignment([1, 2])
        [[('diff', 0), ('intel', 1)], [('diff', 1), ('intel', 0)]]
        """
        index = np.array(index)

        max_possible_index = np.prod(self.cardinality) - 1
        if not all(i <= max_possible_index for i in index):
            raise IndexError("Index greater than max possible index")

        assignments = np.zeros((len(index), len(self.scope())), dtype=np.int)
        rev_card = self.cardinality[::-1]
        for i, card in enumerate(rev_card):
            assignments[:, i] = index % card
            index = index // card

        assignments = assignments[:, ::-1]

        return [[(key, val) for key, val in zip(self.variables, values)] for values in assignments]

    def assignment_statenames(self, index):
        """
        Returns a list of variable statename assignments for the corresponding row index.

        Parameters
        ----------
        index: list, array-like
            List of indices whose assignment is to be computed

        Returns
        -------
        list: Returns a list of full statename assignments of all the variables of the factor.

        Examples
        --------
        >>> from pgmpy.factors import Factor
        >>> statename_dict = {'grade': ['A', 'F'],
        >>>                   'intel': ['smart', 'dumb'],}
        >>> phi = Factor(['grade', 'intel'], None, [.9, .1, .1, .9], statename_dict)
        >>> phi.assignment([1, 2])
        [[('grade', 0), ('intel', 1)], [('grade', 1), ('intel', 0)]]
        """
        assert self.statename_dict is not None, 'must have assigned statements'
        idx_assign_lists = self.assignment(index)
        state_assign_lists = [
            [(key, self.statename_dict[key][val])
             for key, val in idx_assigns]
            for idx_assigns in idx_assign_lists
        ]
        return state_assign_lists

    def identity_factor(self):
        """
        Returns the identity factor.

        Def: The identity factor of a factor has the same scope and cardinality as the original factor,
             but the values for all the assignments is 1. When the identity factor is multiplied with
             the factor it returns the factor itself.

        Returns
        -------
        Factor: The identity factor.

        Examples
        --------
        >>> from pgmpy.factors import Factor
        >>> phi = Factor(['x1', 'x2', 'x3'], [2, 3, 2], range(12))
        >>> phi_identity = phi.identity_factor()
        >>> phi_identity.variables
        ['x1', 'x2', 'x3']
        >>> phi_identity.values
        array([[[ 1.,  1.],
                [ 1.,  1.],
                [ 1.,  1.]],

               [[ 1.,  1.],
                [ 1.,  1.],
                [ 1.,  1.]]])
        """
        return Factor(self.variables, self.cardinality,
                      np.ones(self.values.size), self.statename_dict)

    def marginalize(self, variables, inplace=True):
        """
        Modifies the factor with marginalized values.

        Parameters
        ----------
        variables: list, array-like
            List of variables over which to marginalize.

        inplace: boolean
            If inplace=True it will modify the factor itself, else would return
            a new factor.

        Returns
        -------
        Factor or None: if inplace=True (default) returns None
                        if inplace=False returns a new `Factor` instance.

        Examples
        --------
        >>> from pgmpy.factors import Factor
        >>> phi = Factor(['x1', 'x2', 'x3'], [2, 3, 2], range(12))
        >>> phi.marginalize(['x1', 'x3'])
        >>> phi.values
        array([ 14.,  22.,  30.])
        >>> phi.variables
        ['x2']
        """

        if isinstance(variables, six.string_types):
            raise TypeError("variables: Expected type list or array-like, got type str")

        phi = self if inplace else self.copy()

        for var in variables:
            if var not in phi.variables:
                raise ValueError("{var} not in scope.".format(var=var))

        var_indexes = [phi.variables.index(var) for var in variables]

        index_to_keep = list(set(range(len(self.variables))) - set(var_indexes))
        phi.variables = [phi.variables[index] for index in index_to_keep]
        phi.cardinality = phi.cardinality[index_to_keep]

        phi.values = np.sum(phi.values, axis=tuple(var_indexes))

        if not inplace:
            return phi

    def maximize(self, variables, inplace=True):
        """
        Maximizes the factor with respect to `variables`.

        Parameters
        ----------
        variable: list, array-like
            List of variables with respect to which factor is to be maximized

        inplace: boolean
            If inplace=True it will modify the factor itself, else would return
            a new factor.

        Returns
        -------
        Factor or None: if inplace=True (default) returns None
                        if inplace=False returns a new `Factor` instance.

        Examples
        --------
        >>> from pgmpy.factors import Factor
        >>> phi = Factor(['x1', 'x2', 'x3'], [3, 2, 2], [0.25, 0.35, 0.08, 0.16, 0.05, 0.07,
        ...                                              0.00, 0.00, 0.15, 0.21, 0.09, 0.18])
        >>> phi.maximize(['x2'])
        >>> phi.variables
        ['x1', 'x3']
        >>> phi.cardinality
        array([3, 2])
        >>> phi.values
        array([[ 0.25,  0.35],
               [ 0.05,  0.07],
               [ 0.15,  0.21]])
        """
        if isinstance(variables, six.string_types):
            raise TypeError("variables: Expected type list or array-like, got type str")

        phi = self if inplace else self.copy()

        for var in variables:
            if var not in phi.variables:
                raise ValueError("{var} not in scope.".format(var=var))

        var_indexes = [phi.variables.index(var) for var in variables]

        index_to_keep = list(set(range(len(self.variables))) - set(var_indexes))
        phi.variables = [phi.variables[index] for index in index_to_keep]
        phi.cardinality = phi.cardinality[index_to_keep]

        phi.values = np.max(phi.values, axis=tuple(var_indexes))

        if not inplace:
            return phi

    def normalize(self, inplace=True):
        """
        Normalizes the values of factor so that they sum to 1.

        Parameters
        ----------
        inplace: boolean
            If inplace=True it will modify the factor itself, else would return
            a new factor

        Returns
        -------
        Factor or None: if inplace=True (default) returns None
                        if inplace=False returns a new `Factor` instance.

        Examples
        --------
        >>> from pgmpy.factors import Factor
        >>> phi = Factor(['x1', 'x2', 'x3'], [2, 3, 2], range(12))
        >>> phi.normalize()
        >>> phi.variables
        ['x1', 'x2', 'x3']
        >>> phi.cardinality
        array([2, 3, 2])
        >>> phi.values
        array([[[ 0.        ,  0.01515152],
                [ 0.03030303,  0.04545455],
                [ 0.06060606,  0.07575758]],

               [[ 0.09090909,  0.10606061],
                [ 0.12121212,  0.13636364],
                [ 0.15151515,  0.16666667]]])

        """
        phi = self if inplace else self.copy()

        phi.values = phi.values / phi.values.sum()

        if not inplace:
            return phi

    def reduce(self, values, inplace=True):
        """
        Reduces the factor to the context of given variable values.

        Parameters
        ----------
        values: list, array-like
            A list of tuples of the form (variable_name, variable_state).

        inplace: boolean
            If inplace=True it will modify the factor itself, else would return
            a new factor.

        Returns
        -------
        Factor or None: if inplace=True (default) returns None
                        if inplace=False returns a new `Factor` instance.

        Examples
        --------
        >>> from pgmpy.factors import Factor
        >>> phi = Factor(['x1', 'x2', 'x3'], [2, 3, 2], range(12))
        >>> phi.reduce([('x1', 0), ('x2', 0)])
        >>> phi.variables
        ['x3']
        >>> phi.cardinality
        array([2])
        >>> phi.values
        array([0., 1.])

        >>> # example with string statenames
        >>> statename_dict = {'grade': ['A', 'F'],
        >>>                   'intel': ['smart', 'dumb'],}
        >>> phi2 = Factor(['grade', 'intel'], None, [.9, .1, .1, .9], statename_dict)
        >>> values = [('grade', 'A')]
        >>> phi3 = phi2.reduce(values, inplace=False)
        >>> phi3.values
        array([ 0.9,  0.1])
        """
        if isinstance(values, six.string_types):
            raise TypeError("values: Expected type list or array-like, got type str")

        if isinstance(values, dict):
            # allow for input as a dictionary
            values = list(values.items())

        if all(isinstance(state, six.string_types) for var, state in values):
            # rectify semantic statenames
            values = [(var, self._internal_varindex(var, state)) for var, state in values]

        if (any(isinstance(value, six.string_types) for value in values) or
                not all(isinstance(state, (int, np.integer)) for var, state in values)):
            raise TypeError("values: must contain tuples or array-like elements of the form (hashable object, type int)")

        phi = self if inplace else self.copy()

        var_index_to_del = []
        slice_ = [slice(None)] * len(phi.variables)

        for var, state in values:
            var_index = phi.variables.index(var)
            slice_[var_index] = state
            var_index_to_del.append(var_index)

        var_index_to_keep = sorted(list(set(range(len(phi.variables))) - set(var_index_to_del)))
        new_values = phi.values[tuple(slice_)]
        new_vars = [phi.variables[index] for index in var_index_to_keep]
        new_card = phi.cardinality[var_index_to_keep]

        phi.variables = new_vars
        phi.cardinality = new_card
        phi.values = new_values

        if not inplace:
            return phi

    def sum(self, phi1, inplace=True):
        """
        Factor sum with `phi1`.

        Parameters
        ----------
        phi1: `Factor`
            Factor to be added.

        Returns
        -------
        Factor or None: if inplace=True (default) returns None
                        if inplace=False returns a new `Factor` instance.

        Example
        -------
        >>> from pgmpy.factors import Factor
        >>> phi1 = Factor(['x1', 'x2', 'x3'], [2, 3, 2], range(12))
        >>> phi2 = Factor(['x3', 'x4', 'x1'], [2, 2, 2], range(8))
        >>> phi1.sum(phi2, inplace=True)
        >>> phi1.variables
        ['x1', 'x2', 'x3', 'x4']
        >>> phi1.cardinality
        array([2, 3, 2, 2])
        >>> phi1.values
        array([[[[ 0,  0],
                 [ 4,  6]],

                [[ 0,  4],
                 [12, 18]],

                [[ 0,  8],
                 [20, 30]]],


               [[[ 6, 18],
                 [35, 49]],

                [[ 8, 24],
                 [45, 63]],

                [[10, 30],
                 [55, 77]]]])
        """
        phi = self if inplace else self.copy()
        if isinstance(phi1, (int, float)):
            phi.values += phi1
        else:
            phi1 = phi1.copy()

            if phi1.statename_dict is not None:
                if phi.statename_dict is None:
                    phi.statename_dict = phi1.statename_dict.copy()
                else:
                    phi.statename_dict.update(phi1.statename_dict)

            # modifying phi to add new variables
            extra_vars = set(phi1.variables) - set(phi.variables)
            if extra_vars:
                slice_ = [slice(None)] * len(phi.variables)
                slice_.extend([np.newaxis] * len(extra_vars))
                phi.values = phi.values[slice_]

                phi.variables.extend(extra_vars)

                new_var_card = phi1.get_cardinality(extra_vars)
                phi.cardinality = np.append(phi.cardinality, [new_var_card[var] for var in extra_vars])

            # modifying phi1 to add new variables
            extra_vars = set(phi.variables) - set(phi1.variables)
            if extra_vars:
                slice_ = [slice(None)] * len(phi1.variables)
                slice_.extend([np.newaxis] * len(extra_vars))
                phi1.values = phi1.values[slice_]

                phi1.variables.extend(extra_vars)
                # No need to modify cardinality as we don't need it.

            # rearranging the axes of phi1 to match phi
            for axis in range(phi.values.ndim):
                exchange_index = phi1.variables.index(phi.variables[axis])
                phi1.variables[axis], phi1.variables[exchange_index] = phi1.variables[exchange_index], \
                                                                       phi1.variables[axis]
                phi1.values = phi1.values.swapaxes(axis, exchange_index)

            phi.values = phi.values + phi1.values

        if not inplace:
            return phi

    def product(self, phi1, inplace=True):
        """
        Factor product with `phi1`.

        Parameters
        ----------
        phi1: `Factor`
            Factor to be multiplied.

        Returns
        -------
        Factor or None: if inplace=True (default) returns None
                        if inplace=False returns a new `Factor` instance.

        Example
        -------
        >>> from pgmpy.factors import Factor
        >>> statename_dict = {
        >>>     'x1': ['a1', 'a2'],
        >>>     'x2': ['b1', 'b2', 'b3'],
        >>>     'x3': ['c1', 'c2'],
        >>>     'x4': ['d1', 'd2'],
        >>> }
        >>> statename_dict1 = statename_dict.copy()
        >>> del statename_dict1['x4']
        >>> statename_dict2 = statename_dict.copy()
        >>> del statename_dict2['x2']
        >>> #phi1 = Factor(['x1', 'x2', 'x3'], [2, 3, 2], range(12))
        >>> #phi2 = Factor(['x3', 'x4', 'x1'], [2, 2, 2], range(8))
        >>> phi1 = Factor(['x1', 'x2', 'x3'], None, range(12), statename_dict1)
        >>> phi2 = Factor(['x3', 'x4', 'x1'], None, range(8), statename_dict2)
        >>> phi3 = phi1.product(phi2, inplace=False)
        >>> phi3.variables
        ['x1', 'x2', 'x3', 'x4']
        >>> phi3.cardinality
        array([2, 3, 2, 2])
        >>> phi3.values
        array([[[[ 0,  0],
                 [ 4,  6]],

                [[ 0,  4],
                 [12, 18]],

                [[ 0,  8],
                 [20, 30]]],

               [[[ 6, 18],
                 [35, 49]],

                [[ 8, 24],
                 [45, 63]],

                [[10, 30],
                 [55, 77]]]])
        >>> print(phi3._str(tablefmt='fancy_grid'))
        """
        phi = self if inplace else self.copy()

        if isinstance(phi1, (int, float)):
            phi.values *= phi1
        else:
            phi1 = phi1.copy()

            if phi1.statename_dict is not None:
                if phi.statename_dict is None:
                    phi.statename_dict = phi1.statename_dict.copy()
                else:
                    phi.statename_dict.update(phi1.statename_dict)

            # modifying phi to add new variables
            extra_vars = set(phi1.variables) - set(phi.variables)
            if extra_vars:
                slice_ = [slice(None)] * len(phi.variables)
                slice_.extend([np.newaxis] * len(extra_vars))
                phi.values = phi.values[slice_]

                phi.variables.extend(extra_vars)

                new_var_card = phi1.get_cardinality(extra_vars)
                phi.cardinality = np.append(phi.cardinality, [new_var_card[var] for var in extra_vars])

            # modifying phi1 to add new variables
            extra_vars = set(phi.variables) - set(phi1.variables)
            if extra_vars:
                slice_ = [slice(None)] * len(phi1.variables)
                slice_.extend([np.newaxis] * len(extra_vars))
                phi1.values = phi1.values[slice_]

                phi1.variables.extend(extra_vars)
                # No need to modify cardinality as we don't need it.

            # rearranging the axes of phi1 to match phi
            for axis in range(phi.values.ndim):
                exchange_index = phi1.variables.index(phi.variables[axis])
                phi1.variables[axis], phi1.variables[exchange_index] = phi1.variables[exchange_index], \
                                                                       phi1.variables[axis]
                phi1.values = phi1.values.swapaxes(axis, exchange_index)

            phi.values = phi.values * phi1.values

        if not inplace:
            return phi

    def divide(self, phi1, inplace=True):
        """
        Factor division by `phi1`.

        Parameters
        ----------
        phi1 : Factor
            The denominator for division.

        Returns
        -------
        Factor or None: if inplace=True (default) returns None
                        if inplace=False returns a new `Factor` instance.

        Examples
        --------
        >>> from pgmpy.factors import Factor
        >>> phi1 = Factor(['x1', 'x2', 'x3'], [2, 3, 2], range(12))
        >>> phi2 = Factor(['x3', 'x1'], [2, 2], range(1, 5)])
        >>> phi1.divide(phi2)
        >>> phi1.variables
        ['x1', 'x2', 'x3']
        >>> phi1.cardinality
        array([2, 3, 2])
        >>> phi1.values
        array([[[ 0.        ,  0.33333333],
                [ 2.        ,  1.        ],
                [ 4.        ,  1.66666667]],

               [[ 3.        ,  1.75      ],
                [ 4.        ,  2.25      ],
                [ 5.        ,  2.75      ]]])
        """
        phi = self if inplace else self.copy()
        phi1 = phi1.copy()

        if set(phi1.variables) - set(phi.variables):
            raise ValueError("Scope of divisor should be a subset of dividend")

        # TODO: update phi.statename_dict if needed

        # Adding extra variables in phi1.
        extra_vars = set(phi.variables) - set(phi1.variables)
        if extra_vars:
            slice_ = [slice(None)] * len(phi1.variables)
            slice_.extend([np.newaxis] * len(extra_vars))
            phi1.values = phi1.values[slice_]

            phi1.variables.extend(extra_vars)

        # Rearranging the axes of phi1 to match phi
        for axis in range(phi.values.ndim):
            exchange_index = phi1.variables.index(phi.variables[axis])
            phi1.variables[axis], phi1.variables[exchange_index] = phi1.variables[exchange_index], phi1.variables[axis]
            phi1.values = phi1.values.swapaxes(axis, exchange_index)

        phi.values = phi.values / phi1.values

        # If factor division 0/0 = 0 but is undefined for x/0. In pgmpy we are using
        # np.inf to represent x/0 cases.
        phi.values[np.isnan(phi.values)] = 0

        if not inplace:
            return phi

    def copy(self):
        """
        Returns a copy of the factor.

        Returns
        -------
        Factor: copy of the factor

        Examples
        --------
        >>> import numpy as np
        >>> from pgmpy.factors import Factor
        >>> phi = Factor(['x1', 'x2', 'x3'], [2, 3, 3], np.arange(18))
        >>> phi_copy = phi.copy()
        >>> phi_copy.variables
        ['x1', 'x2', 'x3']
        >>> phi_copy.cardinality
        array([2, 3, 3])
        >>> phi_copy.values
        array([[[ 0,  1,  2],
                [ 3,  4,  5],
                [ 6,  7,  8]],
               [[ 9, 10, 11],
                [12, 13, 14],
                [15, 16, 17]]])
        """
        # not creating a new copy of self.values and self.cardinality
        # because __init__ methods does that.
        statename_dict = (self.statename_dict.copy()
                          if self.statename_dict is not None else None)
        return Factor(self.scope(), self.cardinality, self.values,
                      statename_dict)

    def __str__(self):
        return self._str(phi_or_p='phi')

    def _str(self, phi_or_p="phi", tablefmt=None, sort=False, maxrows=None):
        """
        Generate the string from `__str__` method.

        Parameters
        ----------
        phi_or_p: 'phi' | 'p'
                'phi': When used for Factors.
                  'p': When used for CPDs.
        tablefmt: argument to tabulate
        sort: indicates if rows should be sorted by value
            if 1 or True value are sorted in ascending order
            if -1 values are sorted in descending order
        """
        if tablefmt is None:
            tablefmt = PY_TABLEFMT
        string_header = list(self.scope())
        string_header.append('{phi_or_p}({variables})'.format(
            phi_or_p=phi_or_p, variables=','.join(string_header)))

        factor_table = []

        row_values = self.values.ravel()
        row_labels = list(product(*self.statenames))
        factor_table = [list(lbls) + [val]
                        for lbls, val in zip(row_labels, row_values)]

        if sort:
            sortx = row_values.argsort()[::sort]
            factor_table = [factor_table[row] for row in sortx]

        if maxrows is not None and maxrows < len(factor_table):
            factor_table = factor_table[:maxrows]
            factor_table.append(['...'] * len(string_header))

        return tabulate(factor_table, headers=string_header, tablefmt=tablefmt,
                        floatfmt=".4f")

    def map_bruteforce(self, query_variables, evidence={}):
        """ if this is a joint distribution compute MAP """
        joint = self.evidence_based_reduction(
            query_variables, evidence, inplace=False)
        new_rows = joint._row_labels()
        new_vals = joint.values.ravel()
        map_vals = new_rows[new_vals.argmax()]
        map_assign = dict(zip(joint.variables, map_vals))
        return map_assign

    def evidence_based_reduction(self, query_variables=None,
                                 evidence={}, inplace=False):
        """
        conditions this distribution on evidence
        """
        joint = self if inplace else self.copy()
        if query_variables is None:
            query_variables = self.variables
        self.reduce(evidence)
        joint.normalize()
        # Marginalize over non-query, non-evidence
        irrelevant_vars = (
            set(joint.variables) -
            (set(evidence.keys()) | set(query_variables))
        )
        joint.marginalize(irrelevant_vars)
        joint.normalize()
        if not inplace:
            return joint

    def __repr__(self):
        var_card = ", ".join([
            '{var}:{card}'.format(var=var, card=card)
            for var, card in zip(self.variables, self.cardinality)])
        return "<Factor representing phi({var_card}) at {address}>".format(
            address=hex(id(self)), var_card=var_card)

    def __mul__(self, other):
        return self.product(other, inplace=False)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __add__(self, other):
        return self.sum(other, inplace=False)

    def __radd__(self, other):
        return self.__add__(other)

    def __truediv__(self, other):
        return self.divide(other, inplace=False)

    __div__ = __truediv__

    def __eq__(self, other):
        if not (isinstance(self, Factor) and isinstance(other, Factor)):
            return False

        elif set(self.scope()) != set(other.scope()):
            return False

        else:
            for axis in range(self.values.ndim):
                exchange_index = other.variables.index(self.variables[axis])
                other.variables[axis], other.variables[exchange_index] = (other.variables[exchange_index],
                                                                          other.variables[axis])
                other.cardinality[axis], other.cardinality[exchange_index] = (other.cardinality[exchange_index],
                                                                              other.cardinality[axis])
                other.values = other.values.swapaxes(axis, exchange_index)

            if other.values.shape != self.values.shape:
                return False
            elif not np.allclose(other.values, self.values):
                return False
            elif not all(self.cardinality == other.cardinality):
                return False
            else:
                return True

    def __hash__(self):
        return hash(str(self.variables) + str(self.cardinality.tolist()) +
                    str(self.values.astype('float').tolist()))


def factor_product(*args):
    """
    Returns factor product over `args`.

    Parameters
    ----------
    args: `Factor` instances.
        factors to be multiplied

    Returns
    -------
    Factor: `Factor` representing factor product over all the `Factor` instances in args.

    Examples
    --------
    >>> from pgmpy.factors import Factor, factor_product
    >>> phi1 = Factor(['x1', 'x2', 'x3'], [2, 3, 2], range(12))
    >>> phi2 = Factor(['x3', 'x4', 'x1'], [2, 2, 2], range(8))
    >>> phi = factor_product(phi1, phi2)
    >>> phi.variables
    ['x1', 'x2', 'x3', 'x4']
    >>> phi.cardinality
    array([2, 3, 2, 2])
    >>> phi.values
    array([[[[ 0,  0],
             [ 4,  6]],

            [[ 0,  4],
             [12, 18]],

            [[ 0,  8],
             [20, 30]]],


           [[[ 6, 18],
             [35, 49]],

            [[ 8, 24],
             [45, 63]],

            [[10, 30],
             [55, 77]]]])
    """
    if not all(isinstance(phi, Factor) for phi in args):
        raise TypeError("Arguments must be factors")
    assert len(args) > 0, 'not enough arguments given'
    return reduce(lambda phi1, phi2: phi1 * phi2, args)


def factor_divide(phi1, phi2):
    """
    Returns `Factor` representing `phi1 / phi2`.

    Parameters
    ----------
    phi1: Factor
        The Dividend.

    phi2: Factor
        The Divisor.

    Returns
    -------
    Factor: `Factor` representing factor division `phi1 / phi2`.

    Examples
    --------
    >>> from pgmpy.factors import Factor, factor_divide
    >>> phi1 = Factor(['x1', 'x2', 'x3'], [2, 3, 2], range(12))
    >>> phi2 = Factor(['x3', 'x1'], [2, 2], range(1, 5))
    >>> phi = factor_divide(phi1, phi2)
    >>> phi.variables
    ['x1', 'x2', 'x3']
    >>> phi.cardinality
    array([2, 3, 2])
    >>> phi.values
    array([[[ 0.        ,  0.33333333],
            [ 2.        ,  1.        ],
            [ 4.        ,  1.66666667]],
           [[ 3.        ,  1.75      ],
            [ 4.        ,  2.25      ],
            [ 5.        ,  2.75      ]]])
    """
    if not isinstance(phi1, Factor) or not isinstance(phi2, Factor):
        raise TypeError("phi1 and phi2 should be factors instances")
    return phi1.divide(phi2, inplace=False)


# Search for constructors that need modification
# grep -ER "Factor\(" | grep -v '>>>' | grep -v 'test' | grep -E "Factor\("
