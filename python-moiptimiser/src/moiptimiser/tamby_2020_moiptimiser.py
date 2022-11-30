from moiptimiser.base import *
import itertools
import random
import time
import pickle
import gzip

#open_file = open("solution.pkl", "rb")
#lis_solution = pickle.load(open_file)
#open_file.close()


class Tamby2020MOIPtimiser(MOIPtimiser):

    def __init__(self, model):
        super().__init__(model)
        self._convert_to_min_problem()
        self._defining_points = dict()
        self._decision_variable_map = dict()
        self._init_ideal_point()
        self._once_for_one_instance()
        self._counter = 0
        self._counter_objs = [0,0,0,0,0,0]


    def _kth_projection(self, point, k):
        return tuple(list(point[0:k]) + list(point[k+1:len(point)]))

    # Algorithm 1
    def _update_search_region(self, new_point, search_region):
        # Output
        new_search_region = search_region.copy()
        # Line 2
        for u in search_region:
            # Line 3
            if self.strictly_dominates(new_point,u):
                # Line 4
                new_search_region.remove(u)
                # Line 5
                for l in range(self._num_obj):
                    # Line 6
                    ul = list(u)
                    ul[l] = new_point[l]
                    ul = tuple(ul)
                    # Line 7
                    for k in range(self._num_obj):
                        new_defining_points = set()
                        if k != l:
                            # Line 8
                            if (k,u) in self._defining_points:
                                for defining_point in self._defining_points[(k, u)]:
                                    if defining_point[l] < new_point[l]:
                                        new_defining_points.add(defining_point)
                        self._defining_points[(k, ul)] = new_defining_points
                    # Line 9
                    self._defining_points[(l,ul)] = set([new_point])
                    # Line 10
                    valid_defining_point_exists = True
                    for k in range(self._num_obj):
                        if k != l:
                            if ul[k] != self._M:
                                if len(self._defining_points[(k, ul)]) == 0:
                                    valid_defining_point_exists = False
                                    break
                    if valid_defining_point_exists:
                        # Line 11
                        new_search_region.add(ul)
            # Line 12
            else:
                for k in range(self._num_obj):
                    if new_point[k] == u[k]:
                        kth_new_point_projection = self._kth_projection(new_point, k)
                        kth_u_projection = self._kth_projection(u, k)
                        if self.strictly_dominates(kth_new_point_projection, kth_u_projection):
                            self._defining_points[(k,u)].add(new_point)
        return new_search_region, new_search_region-search_region

    def _set_start_values(self, model, values):
        for varname in values:
            model.getVarByName(varname).setAttr(GRB.Attr.Start, values[varname])

    def _kth_obj_model(self, k):
        new_model = self._new_empty_objective_model()
        self._copy_objective_to(self._model, new_model, k, 0)
        return new_model

    def _init_ideal_point(self):
        point = []
        for k in range(self._num_obj):
            kth_model = self._kth_obj_model(k)
            self._call_solver(kth_model)
            point.append(round(kth_model.ObjNVal))
        self._ideal_point = tuple(point)

    def _once_for_one_instance(self):
        self.var_feats, self.con_feats, self.obj_feats = self._extract_model()

    def _hypervolume_of_projection(self, k, u):
        h = 1
        for i in range(self._num_obj):
            if i != k:
                h = h * (u[i] - self._ideal_point[i])
        return h

    def _next_k_u_(self, U):
        ku_pairs = list(itertools.product(range(self._num_obj), U))
        h_values = [ self._hypervolume_of_projection(k,u) for k,u in ku_pairs ]
        max_h = max(h_values)
              
        #inds = sorted(range(len(h_values)), key=lambda k: h_values[k], reverse=True)
        #max_h = random.randint(0, 2)
        #max_h = h_values[inds[max_h]]

        print(ku_pairs[h_values.index(max_h)])
        return ku_pairs[h_values.index(max_h)]


    #def _next_k_u(self, U):
    #    arr_U = np.array(list(U)).astype(float)
    #    arr_ideal = arr_U-np.array([self._ideal_point]).astype(float)
    #    st = time.time()
    #    ku_pairs = list(itertools.product(range(self._num_obj), U))
    #    produc_all = np.prod(arr_ideal, axis=1)
    #    produc_all_objs = np.tile(produc_all, self._num_obj)
    #    flatten_U = arr_ideal.flatten('F')
    #    #print(111111111111111111111)
    #    #print(time.time()-st)

    #    h_values = (produc_all_objs/flatten_U).tolist()
    #    max_h = max(h_values)  
    #    #print(ku_pairs[h_values.index(max_h)])          
    #    return ku_pairs[h_values.index(max_h)]


    def _next_k_u(self, U):
        arr_U = np.array(list(U)).astype(float)
        arr_U = arr_U[:10000]
        arr_ideal = arr_U-np.array([self._ideal_point]).astype(float)
        st = time.time()
        #ku_pairs = list(itertools.product(range(self._num_obj), U))
        ku_pairs = list(itertools.product(range(self._num_obj), list(U)[:10000]))
        produc_all = np.prod(arr_ideal, axis=1)
        produc_all_objs = np.tile(produc_all, self._num_obj)
        flatten_U = arr_ideal.flatten('F')
        #print(111111111111111111111)
        #print(time.time()-st)

        h_values = (produc_all_objs/flatten_U).tolist()
        max_h = max(h_values)  
        #print(ku_pairs[h_values.index(max_h)])          
        return ku_pairs[h_values.index(max_h)]



    def _return_counter(self):
        return self._counter, self._counter_objs


    def _find_and_set_start_values(self, model, k, u):
        #model.NumStart = 2
        if (k,u) in self._defining_points:
            N_ku = self._defining_points[(k,u)]
            if len(N_ku) > 0:
                feasible_nd = list(N_ku)[0]
                if feasible_nd in self._decision_variable_map:
                    feasible_variables = self._decision_variable_map[feasible_nd]
                    #model.params.StartNumber = 0
                    self._set_start_values(model, feasible_variables)

        #if (k,u) in lis_solution:
        #    #print('have')
        #    #model.params.StartNumber = 1
        #    for varname in lis_solution[(k,u)]:
        #        model.getVarByName(varname).setAttr(GRB.Attr.Start, lis_solution[(k,u)][varname])

    def _find_nd(self, k, u):
        if (k,u) in self._defining_points:
            N_ku = self._defining_points[(k,u)]
            if len(N_ku) > 0:
                feasible_nd = list(N_ku)[0]
        return feasible_nd


    def _summed_expression_from_objectives(self, model, weights):
        coefficient_dict = {}
        for i in range(self._num_obj):
            objective = self._model.getObjective(i)
            for j in range(objective.size()):
                var = objective.getVar(j)
                coeff = objective.getCoeff(j) * weights[i]
                if var.VarName not in coefficient_dict:
                    coefficient_dict[var.VarName] = 0
                coefficient_dict[var.VarName] = coefficient_dict[var.VarName] + coeff
        summed_expression = gp.LinExpr()
        for varname in coefficient_dict:
            new_var = model.getVarByName(varname)
            summed_expression.add(new_var, coefficient_dict[varname])
        return summed_expression

    def _upper_bounds_from_solved_model(self, model):
        upper_bounds = []
        for i in range(self._num_obj):
            upper_bounds.append(self._eval_objective_given_model(model, self._model.getObjective(i)))
        return tuple(upper_bounds)

    def _find_point(self, k, u):
        subproblem = self._construct_subproblem(k, u)
        st = time.time()
        self._call_solver(subproblem)
        #print(time.time()-st)
        new_point = tuple(
            [self._eval_objective_given_model(subproblem, self._model.getObjective(i))
             for i in range(self._num_obj)]
        )
        decision_variables = self._var_values_by_name_dict(subproblem)
        return (new_point, decision_variables)

    def _remove_dominated(self, nds):
        filtered = set()
        for nd in nds:
            if not any( (self.dominates(other, nd) for other in nds) ):
                filtered.add(nd)
        return filtered

    # Algorithm 2
    def find_non_dominated_objective_vectors(self, runtime, tag):
        # add dict to extract features
        F_dict = {}
        F_dict['ind_select'] = {}
        F_dict['defining_point'] = {}
        F_dict['pre_defining_point'] = {}
        # F1: ideal points
        #F_dict['ideal'] = self._ideal_point
        # record the precedor of a defining point
        M = {}

        # Line 1
        N = set()
        U = set()
        U.add( tuple([self._M] * self._num_obj) )
        V = {}
        for k in range(self._num_obj):
            V[k] = set()

        # Line 2
        # Ideal point computed once already in the class constructor

        # Line 3
        #lis_defining = {}
        lis_solution = {}
        open_file = open("infeasible.pkl", "rb")
        lis_defining = pickle.load(open_file)
        open_file.close()

        ite = 0
        st_run = time.time()
        while len(U) > 0:
            ite+=1
            # Line 4
            k, u = self._next_k_u(U)
            #complete the current single-objective IP
            #con_mat = np.concatenate([np.delete(self.var_feats, k, 0), np.delete(np.asarray(u)[:,np.newaxis]-0.5, k, 0)],axis=-1)
            #con_mat = np.concatenate([self.con_feats, con_mat],axis=0)
            # F2: selected single-objective
            #F_dict['ind_select'][ite] = (k, u)
            # F3: defining point if exsiting, or give all 0s
            #if (k,u) in self._defining_points:
            #    dpoint = self._defining_points[(k,u)]
            #    # F4: find the precedor of defining
            #    pbound_k, pbound_u = M[list(self._defining_points[(k,u)])[0]]
            #    obj_mat = np.concatenate([self.var_feats, np.asarray(self._ideal_point)[:,np.newaxis], np.asarray(list(dpoint)[0])[:,np.newaxis], np.asarray(list(pbound_u))[:,np.newaxis], np.asarray(list(U)).mean(0)[:,np.newaxis], np.asarray(list(U)).max(0)[:,np.newaxis], np.asarray(list(U)).min(0)[:,np.newaxis]], axis=-1)
            #else:
            #    dpoint = {tuple([0] * self._num_obj)}
            #    # F4: find the precedor of defining
            #    pbound_k, pbound_u = (k, tuple([0] * self._num_obj))
            #    obj_mat = np.concatenate([self.var_feats, np.asarray(self._ideal_point)[:,np.newaxis], np.asarray(list(dpoint)[0])[:,np.newaxis], np.asarray(list(pbound_u))[:,np.newaxis], np.asarray(list(U)).mean(0)[:,np.newaxis], np.asarray(list(U)).max(0)[:,np.newaxis], np.asarray(list(U)).min(0)[:,np.newaxis]], axis=-1)
            #if len(self._decision_variable_map)>0:
            #    cur_solutions = np.asarray([list(self._decision_variable_map[i].values()) for i in self._decision_variable_map])
            #    var_mat = np.concatenate([self.var_feats[k][:,np.newaxis], cur_solutions.mean(0)[:,np.newaxis], cur_solutions.std(0)[:,np.newaxis]],axis=-1)
            #else:
            #    var_mat = np.concatenate([self.var_feats[k][:,np.newaxis], np.zeros_like(self.var_feats[k])[:,np.newaxis], np.zeros_like(self.var_feats[k])[:,np.newaxis]],axis=-1)

            if (k,u) in lis_defining:
                new_point = lis_defining[(k,u)]
                if lis_defining[(k,u)] in self._defining_points[(k,u)]:
                    U.remove(u)
            else:
            # Line 5
                new_point, decision_variables = self._find_point(k, u)
                self._decision_variable_map[new_point] = decision_variables
                # Line 6
                V[k].add( (u, new_point[k]) )
                u_, new_, k_ = u, new_point[k], k
                # record the precedor of a defining point
                M[new_point] = (k, u)

            # Line 7
            if (k,u) not in self._defining_points:
                self._defining_points[(k,u)] = set()

            #lis_solution[(k,u)] = decision_variables
            sol = np.fromiter(decision_variables.values(), dtype=float)
            if new_point in self._defining_points[(k,u)]:
                #print('true')
                label = 1
                lis_defining[(k,u)] = new_point
            else:
                label = 0

            #data = (con_mat, obj_mat, var_mat, label, k, sol)
            #filename = f"{'valid_data_oc_sol/AP_6_40'}/sample_{ite}_{round(time.time())}_{tag}.pkl"
            #with gzip.open(filename, "wb") as out_file:
            #    pickle.dump(data, out_file)


            if new_point not in self._defining_points[(k,u)]:
                # Line 8
                U, U_ = self._update_search_region(new_point, U)
                # Line 9
                N.add(new_point)

            ## Line 10
            #for u_dash in U.copy():
            #    # Line 11
            #    for k in range(self._num_obj):
            #        # Line 12
            #        if u_dash[k] == self._ideal_point[k]:
            #            # Line 13
            #            U.remove(u_dash)
            #        # Line 14
            #        else:
            #            # Line 15
            #            for u, y_k in V[k]:
            #                # Line 16
            #                if y_k == u_dash[k] and u_dash in U:
            #                    kth_u_dash_projection = self._kth_projection(u_dash, k)
            #                    kth_u_projection = self._kth_projection(u, k)
            #                    weakly_dominated = self.weakly_dominates(kth_u_dash_projection, kth_u_projection)
            #                    if weakly_dominated:
            #                        #print('false')
            #                        # Line 17
            #                        U.remove(u_dash)
            #                        #print(u_dash)
            #                        #print('-------------')


            st =time.time()
            # Line 10
            for u_dash in U_.copy():
                # Line 11
                for k in range(self._num_obj):
                    # Line 12
                    if u_dash[k] == self._ideal_point[k]:
                        # Line 13
                        U.remove(u_dash)
                        U_.remove(u_dash)
                    # Line 14
                    else:
                        # Line 15
                        for u, y_k in V[k]:
                            # Line 16
                            if y_k == u_dash[k] and u_dash in U_:
                                kth_u_dash_projection = self._kth_projection(u_dash, k)
                                kth_u_projection = self._kth_projection(u, k)
                                weakly_dominated = self.weakly_dominates(kth_u_dash_projection, kth_u_projection)
                                if weakly_dominated:
                                    # Line 17
                                    U.remove(u_dash)
                                    U_.remove(u_dash)
                                    #print(u_dash)

            UC = U-U_
            for u_dash in UC.copy():
                u, y_k = u_, new_
                # Line 16
                if y_k == u_dash[k_] and u_dash in UC:
                    kth_u_dash_projection = self._kth_projection(u_dash, k_)
                    kth_u_projection = self._kth_projection(u, k_)
                    weakly_dominated = self.weakly_dominates(kth_u_dash_projection, kth_u_projection)
                    if weakly_dominated:
                        # Line 17
                        U.remove(u_dash)
                        UC.remove(u_dash)
                        #print(u_dash)
            #print(time.time()-st)


        # Line 18
        #file_name = "infeasible.pkl"
        #open_file = open(file_name, "wb")
        #pickle.dump(lis_defining, open_file)
        #open_file.close()

        #file_name = "solution.pkl"
        #open_file = open(file_name, "wb")
        #pickle.dump(lis_solution, open_file)
        #open_file.close()

            #if time.time()-st_run>runtime:
            #    break


        return self._correct_sign_for_solutions(self._remove_dominated(N))
