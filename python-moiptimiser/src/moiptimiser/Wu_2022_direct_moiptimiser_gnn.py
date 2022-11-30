from moiptimiser.Wu_2022_moiptimiser_gnn import *

class Wu2022DirectMOIPtimiser_gnn(Wu2022MOIPtimiser_gnn):

    def _delta_for_direct(self, k, u):
        delta = 1
        for i in range(self._num_obj):
            if i != k:
                delta = delta + u[i] - self._ideal_point[i]
        return delta

    def _construct_subproblem(self, k, u):
        # Direct Approach
        #model = self._new_empty_objective_model()
        self.fixed_part.update()
        model = self.fixed_part.copy()
        self._set_other_objectives_as_constraints_simple(model, k, u)
        weights = [1] * self._num_obj
        weights[k] = self._delta_for_direct(k, u)
        summed_expression = self._summed_expression_from_objectives(model, weights)
        model.setObjective(summed_expression)
        model.update()
        self._find_and_set_start_values(model, k, u)
        return model


    def _construct_subproblem_warm(self, k, u, sol):
        # Direct Approach
        #model = self._new_empty_objective_model()
        self.fixed_part.update()
        model = self.fixed_part.copy()
        self._set_other_objectives_as_constraints(model, k, u)
        weights = [1] * self._num_obj
        weights[k] = self._delta_for_direct(k, u)
        summed_expression = self._summed_expression_from_objectives(model, weights)
        model.setObjective(summed_expression)
        model.update()
        self._find_and_set_start_values_warm(model, k, u, sol)
        return model
