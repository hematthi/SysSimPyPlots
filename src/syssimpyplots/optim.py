# To import required modules:
import numpy as np





# Functions to analyze the results of optimizing our models:

#file_names = 'Clustered_P_R_optimization_random%s_targs86760_evals5000.txt'
file_names = 'Hybrid1_targs86760_evals5000_run%s.txt'

def analyze_bboptimize_runs(loadfiles_directory, file_names=file_names, run_numbers=range(1,51)):
    active_params_names_all = [] # list to be filled with arrays of the names of the active parameters for each run (should be the same for all runs)
    active_params_bounds_all = [] # list to be filled with arrays of the search bounds of the active parameters for each run (should be the same for all runs)
    active_params_start_all = []
    steps_best_weighted_all = [] # list to be filled with the number of model iterations to find the best active parameter values (lowest total weighted distance) for each run
    steps_tot_all = [] # list to be filled with the number of total model iterations in the optimization procedure, for each run
    time_optimization_all = [] # list to be filled with the elapsed times (s) for the full optimization procedure, for each run

    active_params_runs = [] # list to be filled with 2D array of the values of all the active parameters at every step for each run
    active_params_all = [] # list to be filled with arrays of the values of all the active parameters at every step (excluding starting values but including best values) for all the runs
    d_used_keys_runs = []
    d_used_vals_runs = []
    d_used_vals_all = []
    d_used_vals_w_runs = []
    d_used_vals_w_all = []

    runs_started = 0
    runs_finished = 0
    for i in run_numbers:
        with open(loadfiles_directory + file_names % i, 'r') as file:

            optim_lines = False # set to true once we start reading lines in the file that are the outputs of the optimization
            active_params_start = [] # will be replaced by the actual active parameter values if the file is not empty

            active_params_run = [] # will be filled with all the active params at each step of the optimization in this run
            d_used_keys_run = []
            d_used_vals_run = [] # will be filled with all the distances at each step of the optimization in this run
            d_used_vals_w_run = [] # will be filled with all the weighted distances at each step of the optimization in this run

            best_fitness = np.inf # will be replaced with the best total weighted distance if the optimization progressed
            steps = 0 # will be a running count of the number of model iterations
            steps_best_weighted = steps # will be replaced by the number of the model iteration at which the best total weighted distance was found
            for line in file:
                # For recording the preliminary runs of the model before optimizations:
                if line[0:19] == '# Active parameters':
                    active_params_names = line[23:-3].split('", "')
                    active_params_names_all.append(active_params_names)

                # For recording the results of the optimizations:
                elif line[0:7] == '# Start':
                    active_params_start = [float(x) for x in line[37:-2].split(', ')]
                elif line[0:7] == '# Optim':
                    runs_started += 1
                    active_params_bounds = [(float(x.split(', ')[0]), float(x.split(', ')[1])) for x in line[50:-3].split('), (')]
                    optim_lines = True
                elif line[0:13] == 'Active_params' and optim_lines:
                    steps += 1
                    active_params = [float(x) for x in line[16:-2].split(', ')]
                    active_params_run.append(active_params)
                    active_params_all.append(active_params)

                if optim_lines:
                    if line[0:6] == 'Counts':
                        Nmult_str, counts_str = line[9:-2].split('][')
                        Nmult = [int(x) for x in Nmult_str.split(', ')]

                    elif line[0:12] == 'd_used_keys:':
                        d_used_keys = line[15:-3].split('", "')
                        d_used_keys_run.append(d_used_keys)

                    elif line[0:12] == 'd_used_vals:':
                        d_used_vals_str, d_used_vals_tot_str = line[14:-2].split('][')
                        d_used_vals = [float(x) for x in d_used_vals_str.split(', ')]
                        d_used_vals_run.append(d_used_vals)
                        d_used_vals_all.append(tuple(d_used_vals))

                    elif line[0:13] == 'd_used_vals_w':
                        d_used_vals_w_str, d_used_vals_tot_w_str = line[16:-2].split('][')
                        d_used_vals_w = [float(x) for x in d_used_vals_w_str.split(', ')]
                        d_used_vals_tot_w = float(d_used_vals_tot_w_str)
                        d_used_vals_w_run.append(d_used_vals_w)
                        d_used_vals_w_all.append(tuple(d_used_vals_w))
                        if d_used_vals_tot_w < best_fitness:
                            best_fitness = d_used_vals_tot_w
                            steps_best_weighted = steps

                if line[0:14] == '# best_fitness':
                    runs_finished += 1
                    best_fitness_end = float(line[16:-1])
                elif line[0:9] == '# elapsed' and optim_lines:
                    time_optimization_all.append(float(line[16:-8]))

            print(i, best_fitness, len(active_params_all), len(d_used_vals_w_all))

            active_params_bounds_all.append(active_params_bounds)
            active_params_start_all.append(active_params_start)

            active_params_runs.append(active_params_run)
            d_used_keys_runs.append(d_used_keys_run)
            d_used_vals_runs.append(d_used_vals_run)
            d_used_vals_w_runs.append(d_used_vals_w_run)

            steps_best_weighted_all.append(steps_best_weighted)
            steps_tot_all.append(steps)

    print('Runs successfully started (and not killed): ', runs_started) # runs killed because of the wall time are not counted here because they have their output files emptied
    print('Runs successfully finished (reached max iterations or target fitness): ', runs_finished) # runs not counted here are ones killed either because of the wall time, or because of bus error

    active_params_names_all = np.array(active_params_names_all)
    active_params_bounds_all = np.array(active_params_bounds_all)
    active_params_start_all = np.array(active_params_start_all)

    steps_best_weighted_all = np.array(steps_best_weighted_all)
    steps_tot_all = np.array(steps_tot_all)
    time_optimization_all = np.array(time_optimization_all)

    if np.min(steps_tot_all) != np.max(steps_tot_all):
        print('Warning: not all runs have the same number of iterations.')
        active_params_runs = [np.array(x) for x in active_params_runs] # list of 2d arrays
    else:
        active_params_runs = np.array(active_params_runs) # 3d array
    active_params_all = np.array(active_params_all)

    d_used_keys_runs = np.array(d_used_keys_runs)
    d_used_vals_runs = np.array(d_used_vals_runs)
    d_used_vals_all = np.array(d_used_vals_all, dtype=[(d_key, 'f8') for d_key in d_used_keys_runs[0][0]])
    d_used_vals_w_runs = np.array(d_used_vals_w_runs)
    d_used_vals_w_all = np.array(d_used_vals_w_all, dtype=[(d_key, 'f8') for d_key in d_used_keys_runs[0][0]])

    # To compute the sums of weighted distances per iteration, for each sample:
    dtot_runs = np.array([np.sum(run_2d, axis=1) for run_2d in d_used_vals_runs]) # will be a 2D array of size (runs, evals)
    dtot_all = np.array([sum(x) for x in d_used_vals_all]) # will be a 1D array of length runs*evals
    dtot_w_runs = np.array([np.sum(run_2d, axis=1) for run_2d in d_used_vals_w_runs]) # will be a 2D array of size (runs, evals)
    dtot_w_all = np.array([sum(x) for x in d_used_vals_w_all]) #np.sum(d_used_vals_w_all[sample], axis=1) # will be a 1D array of length runs*evals

    active_params_best_all = np.array([active_params_runs[n][np.argsort(dtot_w_runs[n])[0]] for n in range(len(run_numbers))])
    dtot_best_all = np.array([dtot_w_runs[n][np.argsort(dtot_w_runs[n])[0]] for n in range(len(run_numbers))])

    results = {}
    #results['d_used_keys_runs'] = d_used_keys_runs
    results['t_optim_all'] = time_optimization_all
    results['d_used_vals_all'] = d_used_vals_all
    results['d_used_vals_w_all'] = d_used_vals_w_all
    results['dtot_all'] = dtot_all
    results['dtot_w_all'] = dtot_w_all # 1D array (flattened)
    results['dtot_w_runs'] = dtot_w_runs # 2D array of size (runs, evals)
    results['dtot_best_all'] = dtot_best_all
    results['i_sort_dtot_w_all'] = np.argsort(dtot_w_all) # indices that would sort all the model evaluations by the total weighted distance
    results['active_params_names_all'] = active_params_names_all
    results['active_params_bounds_all'] = active_params_bounds_all
    results['active_params_all'] = active_params_all # 2D array of size (runs*evals, params)
    results['active_params_runs'] = active_params_runs # 3D array of size (runs, evals, params)
    results['active_params_best_all'] = active_params_best_all
    return results



def analyze_bboptimize_split_stars_runs(loadfiles_directory, file_names=file_names, run_numbers=range(1,51)):
    sample_names = ['all', 'bluer', 'redder']

    active_params_names_all = [] # list to be filled with arrays of the names of the active parameters for each run (should be the same for all runs)
    active_params_bounds_all = [] # list to be filled with arrays of the search bounds of the active parameters for each run (should be the same for all runs)
    active_params_start_all = []
    steps_best_weighted_all = [] # list to be filled with the number of model iterations to find the best active parameter values (lowest total weighted distance) for each run
    steps_tot_all = [] # list to be filled with the number of total model iterations in the optimization procedure, for each run
    time_optimization_all = [] # list to be filled with the elapsed times (s) for the full optimization procedure, for each run

    active_params_runs = [] # list to be filled with 2D array of the values of all the active parameters at every step for each run
    active_params_all = [] # list to be filled with arrays of the values of all the active parameters at every step (excluding starting values but including best values) for all the runs
    d_used_keys_runs = {key: [] for key in sample_names}
    d_used_vals_runs = {key: [] for key in sample_names}
    d_used_vals_all = {key: [] for key in sample_names}
    d_used_vals_w_runs = {key: [] for key in sample_names}
    d_used_vals_w_all = {key: [] for key in sample_names}

    runs_started = 0
    runs_finished = 0
    for i in run_numbers:
        with open(loadfiles_directory + file_names % i, 'r') as file:

            optim_lines = False # set to true once we start reading lines in the file that are the outputs of the optimization
            active_params_start = [] # will be replaced by the actual active parameter values if the file is not empty

            active_params_run = [] # will be filled with all the active params at each step of the optimization in this run
            d_used_keys_run = {key: [] for key in sample_names}
            d_used_vals_run = {key: [] for key in sample_names} # will be filled with all the distances at each step of the optimization in this run
            d_used_vals_w_run = {key: [] for key in sample_names} # will be filled with all the weighted distances at each step of the optimization in this run

            best_fitness = np.inf # will be replaced with the best total weighted distance if the optimization progressed
            steps = 0 # will be a running count of the number of model iterations
            steps_best_weighted = steps # will be replaced by the number of the model iteration at which the best total weighted distance was found
            for line in file:
                # For recording the preliminary runs of the model before optimizations:
                if line[0:19] == '# Active parameters':
                    active_params_names = line[23:-3].split('", "')
                    active_params_names_all.append(active_params_names)

                # For recording the results of the optimizations:
                elif line[0:7] == '# Start':
                    active_params_start = [float(x) for x in line[37:-2].split(', ')]
                elif line[0:7] == '# Optim':
                    runs_started += 1
                    active_params_bounds = [(float(x.split(', ')[0]), float(x.split(', ')[1])) for x in line[50:-3].split('), (')] # NOTE: was line[72:-3] before
                    optim_lines = True
                elif line[0:13] == 'Active_params' and optim_lines:
                    steps += 1
                    active_params = [float(x) for x in line[16:-2].split(', ')]
                    active_params_run.append(active_params)
                    active_params_all.append(active_params)
                elif line[0:12] == 'Total_dist_w' and optim_lines:
                    total_dist_w = float(line[15:-2])
                    if total_dist_w < best_fitness:
                        best_fitness = total_dist_w
                        steps_best_weighted = steps

                for sample in sample_names:
                    n = len(sample)
                    if line[0:n+2] == '[%s]' % sample and optim_lines:
                        if line[n+3:n+3+6] == 'Counts':
                            Nmult_str, counts_str = line[n+3+9:-2].split('][')
                            Nmult = [int(x) for x in Nmult_str.split(', ')]

                        elif line[n+3:n+3+12] == 'd_used_keys:':
                            d_used_keys = line[n+3+15:-3].split('", "')
                            d_used_keys_run[sample].append(d_used_keys)

                        elif line[n+3:n+3+12] == 'd_used_vals:':
                            d_used_vals_str, d_used_vals_tot_str = line[n+3+14:-2].split('][')
                            d_used_vals = [float(x) for x in d_used_vals_str.split(', ')]
                            d_used_vals_run[sample].append(d_used_vals)
                            d_used_vals_all[sample].append(tuple(d_used_vals))

                        elif line[n+3:n+3+13] == 'd_used_vals_w':
                            d_used_vals_w_str, d_used_vals_tot_w_str = line[n+3+16:-2].split('][')
                            d_used_vals_w = [float(x) for x in d_used_vals_w_str.split(', ')]
                            d_used_vals_tot_w = float(d_used_vals_tot_w_str)
                            d_used_vals_w_run[sample].append(d_used_vals_w)
                            d_used_vals_w_all[sample].append(tuple(d_used_vals_w))

                if line[0:14] == '# best_fitness':
                    runs_finished += 1
                    best_fitness_end = float(line[16:-1])
                elif line[0:9] == '# elapsed' and optim_lines:
                    time_optimization_all.append(float(line[16:-8]))

            print(i, best_fitness, len(active_params_all), [len(d_used_vals_w_all[key]) for key in d_used_vals_w_all])

            active_params_bounds_all.append(active_params_bounds)
            active_params_start_all.append(active_params_start)

            active_params_runs.append(active_params_run)
            for sample in sample_names:
                d_used_keys_runs[sample].append(d_used_keys_run[sample])
                d_used_vals_runs[sample].append(d_used_vals_run[sample])
                d_used_vals_w_runs[sample].append(d_used_vals_w_run[sample])

            steps_best_weighted_all.append(steps_best_weighted)
            steps_tot_all.append(steps)

    print('Runs successfully started (and not killed): ', runs_started) # runs killed because of the wall time are not counted here because they have their output files emptied
    print('Runs successfully finished (reached max iterations or target fitness): ', runs_finished) # runs not counted here are ones killed either because of the wall time, or because of bus error

    active_params_names_all = np.array(active_params_names_all)
    active_params_bounds_all = np.array(active_params_bounds_all)
    active_params_start_all = np.array(active_params_start_all)

    steps_best_weighted_all = np.array(steps_best_weighted_all)
    steps_tot_all = np.array(steps_tot_all)
    time_optimization_all = np.array(time_optimization_all)

    active_params_runs = np.array(active_params_runs)
    active_params_all = np.array(active_params_all)

    for sample in sample_names:
        d_used_keys_runs[sample] = np.array(d_used_keys_runs[sample])
        d_used_vals_runs[sample] = np.array(d_used_vals_runs[sample])
        d_used_vals_all[sample] = np.array(d_used_vals_all[sample], dtype=[(d_key, 'f8') for d_key in d_used_keys_runs[sample][0][0]])
        d_used_vals_w_runs[sample] = np.array(d_used_vals_w_runs[sample])
        d_used_vals_w_all[sample] = np.array(d_used_vals_w_all[sample], dtype=[(d_key, 'f8') for d_key in d_used_keys_runs[sample][0][0]])

    # To compute the sums of weighted distances per iteration, for each sample:
    dtot_samples_runs = {}
    dtot_samples_all = {}
    dtot_w_samples_runs = {}
    dtot_w_samples_all = {}
    for sample in sample_names:
        dtot_samples_runs[sample] = np.array([np.sum(run_2d, axis=1) for run_2d in d_used_vals_runs[sample]]) # will be a 2D array of size (runs, steps)
        dtot_samples_all[sample] = np.array([sum(x) for x in d_used_vals_all[sample]]) # will be a 1D array of length runs*steps
        dtot_w_samples_runs[sample] = np.array([np.sum(run_2d, axis=1) for run_2d in d_used_vals_w_runs[sample]]) # will be a 2D array of size (runs, steps)
        dtot_w_samples_all[sample] = np.array([sum(x) for x in d_used_vals_w_all[sample]]) #np.sum(d_used_vals_w_all[sample], axis=1) # will be a 1D array of length runs*steps

    dtot_runs = sum(dtot_samples_runs[sample] for sample in sample_names) # 2D
    dtot_all = sum(dtot_samples_all[sample] for sample in sample_names) # 1D
    dtot_w_runs = sum(dtot_w_samples_runs[sample] for sample in sample_names) # 2D
    dtot_w_all = sum(dtot_w_samples_all[sample] for sample in sample_names) # 1D

    active_params_best_all = np.array([active_params_runs[n][np.argsort(dtot_w_runs[n])[0]] for n in range(len(run_numbers))])
    dtot_best_all = np.array([dtot_w_runs[n][np.argsort(dtot_w_runs[n])[0]] for n in range(len(run_numbers))])

    results = {}
    #results['d_used_keys_runs'] = d_used_keys_runs
    results['t_optim_all'] = time_optimization_all
    results['d_used_vals_all'] = d_used_vals_all # dict of 2D structured arrays
    results['d_used_vals_w_all'] = d_used_vals_w_all # dict of 2D structured arrays
    results['dtot_samples_all'] = dtot_samples_all # dict of 1D arrays
    results['dtot_w_samples_all'] = dtot_w_samples_all # dict of 1D arrays
    results['dtot_all'] = dtot_all # 1D
    results['dtot_w_all'] = dtot_w_all # 1D
    results['dtot_best_all'] = dtot_best_all # 1D
    results['i_sort_dtot_w_all'] = np.argsort(dtot_w_all) # 1D indices that would sort all the model evaluations by the total weighted distance
    results['active_params_names_all'] = active_params_names_all # 2D of size (runs, params)
    results['active_params_bounds_all'] = active_params_bounds_all # 3D of size (runs, params, 2)
    results['active_params_all'] = active_params_all # 2D of size (runs*steps, params)
    results['active_params_best_all'] = active_params_best_all # 2D of size (runs, params)
    return results





# Functions to read the file of recomputed distances:

def load_recomputed_distances_file(file_name):
    active_params_evals = []
    d_used_keys_evals = []
    d_used_vals_evals = []
    d_used_vals_w_evals = []

    with open(file_name, 'r') as file:
        for line in file:
            if line[0:19] == '# Active parameters':
                active_params_names = line[23:-3].split('", "')
            elif line[0:13] == 'Active_params':
                active_params = [float(x) for x in line[16:-2].split(', ')]
                active_params_evals.append(active_params)

            # For recording the results of the re-simulations:
            if line[0:12] == 'd_used_keys:':
                d_used_keys = line[15:-3].split('", "')
                d_used_keys_evals.append(d_used_keys)

            elif line[0:12] == 'd_used_vals:':
                d_used_vals_str, d_used_vals_tot_str = line[14:-2].split('][')
                d_used_vals = [float(x) for x in d_used_vals_str.split(', ')]
                d_used_vals_evals.append(tuple(d_used_vals))

            elif line[0:13] == 'd_used_vals_w':
                d_used_vals_w_str, d_used_vals_tot_w_str = line[16:-2].split('][')
                d_used_vals_w = [float(x) for x in d_used_vals_w_str.split(', ')]
                d_used_vals_tot_w = float(d_used_vals_tot_w_str)
                d_used_vals_w_evals.append(tuple(d_used_vals_w))

    active_params_evals = np.array(active_params_evals)
    d_used_keys_evals = np.array(d_used_keys_evals)
    d_used_vals_evals = np.array(d_used_vals_evals, dtype=[(d_key, 'f8') for d_key in d_used_keys_evals[0]])
    d_used_vals_w_evals = np.array(d_used_vals_w_evals, dtype=[(d_key, 'f8') for d_key in d_used_keys_evals[0]])
    dtot_w_evals = np.array([sum(x) for x in d_used_vals_w_evals])

    results = {}
    results['active_params_names'] = active_params_names
    results['active_params_evals'] = active_params_evals
    results['d_used_vals_w_evals'] = d_used_vals_w_evals
    results['dtot_w_evals'] = dtot_w_evals
    return results



def load_recomputed_distances_split_stars_file(file_name):
    sample_names = ['all', 'bluer', 'redder']

    active_params_evals = []
    d_used_keys_evals = {key: [] for key in sample_names}
    d_used_vals_evals = {key: [] for key in sample_names}
    d_used_vals_w_evals = {key: [] for key in sample_names}
    total_dist_w_evals = []

    with open(file_name, 'r') as file:
        for line in file:
            if line[0:19] == '# Active parameters':
                active_params_names = line[23:-3].split('", "')
            elif line[0:13] == 'Active_params':
                active_params = [float(x) for x in line[16:-2].split(', ')]
                active_params_evals.append(active_params)
            elif line[0:12] == 'Total_dist_w':
                total_dist_w = float(line[15:-2])
                total_dist_w_evals.append(total_dist_w)

            for key in sample_names:
                n = len(key)
                if line[0:n+2] == '[%s]' % key:

                    if line[n+3:n+3+12] == 'd_used_keys:':
                        d_used_keys = line[n+3+15:-3].split('", "')
                        d_used_keys_evals[key].append(d_used_keys)

                    elif line[n+3:n+3+12] == 'd_used_vals:':
                        d_used_vals_str, d_used_vals_tot_str = line[n+3+14:-2].split('][')
                        d_used_vals = [float(x) for x in d_used_vals_str.split(', ')]
                        d_used_vals_evals[key].append(tuple(d_used_vals))

                    elif line[n+3:n+3+13] == 'd_used_vals_w':
                        d_used_vals_w_str, d_used_vals_tot_w_str = line[n+3+16:-2].split('][')
                        d_used_vals_w = [float(x) for x in d_used_vals_w_str.split(', ')]
                        d_used_vals_tot_w = float(d_used_vals_tot_w_str)
                        d_used_vals_w_evals[key].append(tuple(d_used_vals_w))

    active_params_evals = np.array(active_params_evals)
    total_dist_w_evals = np.array(total_dist_w_evals)

    for sample in sample_names:
        d_used_keys_evals[sample] = np.array(d_used_keys_evals[sample])
        d_used_vals_evals[sample] = np.array(d_used_vals_evals[sample], dtype=[(d_key, 'f8') for d_key in d_used_keys_evals[sample][0]])
        d_used_vals_w_evals[sample] = np.array(d_used_vals_w_evals[sample], dtype=[(d_key, 'f8') for d_key in d_used_keys_evals[sample][0]])

    # To compute the sums of weighted distances per iteration, for each sample:
    dtot_samples_evals = {}
    dtot_w_samples_evals = {}
    for sample in sample_names:
        dtot_samples_evals[sample] = np.array([sum(x) for x in d_used_vals_evals[sample]])
        dtot_w_samples_evals[sample] = np.array([sum(x) for x in d_used_vals_w_evals[sample]])
    dtot_w_evals = sum(dtot_w_samples_evals[sample] for sample in sample_names)

    for i in range(len(dtot_w_evals)):
        a, b = dtot_w_evals[i], total_dist_w_evals[i]
        #if np.abs(a - b) > 1e-4:
            #print('{:<5}: {:<8}, {:<8}'.format(i, np.round(a,4), np.round(b,4)))

    results = {}
    results['active_params_names'] = active_params_names
    results['active_params_evals'] = active_params_evals
    results['d_used_vals_w_evals'] = d_used_vals_w_evals
    results['dtot_w_samples_evals'] = dtot_w_samples_evals
    results['dtot_w_evals'] = dtot_w_evals
    return results



# Functions to save the best parameter values and their recomputed distances for training a GP emulator:

def savetxt_active_params_recomputed_distances_table(results, save_path_name):
    active_params_distances_table = np.concatenate((results['active_params_evals'], np.array([results['dtot_w_evals']]).transpose()), axis=1)
    table_header = ' '.join(results['active_params_names']) + ' dist_tot_weighted'
    fields_formats = ['%1.6f']*len(results['active_params_names']) + ['%1.6f']
    
    np.savetxt(save_path_name, active_params_distances_table, fmt=fields_formats, header=table_header, comments='')



def savetxt_active_params_recomputed_distances_table_split_stars(results, save_path_name):
    active_params_distances_table = np.concatenate((results['active_params_evals'], np.array([results['dtot_w_samples_evals']['all'], results['dtot_w_samples_evals']['bluer'], results['dtot_w_samples_evals']['redder'], results['dtot_w_evals']]).transpose()), axis=1)
    table_header = ' '.join(results['active_params_names']) + ' dtot_w_all dtot_w_bluer dtot_w_redder dist_tot_weighted'
    fields_formats = ['%1.6f']*len(results['active_params_names']) + ['%1.6f']*4
    
    np.savetxt(save_path_name, active_params_distances_table, fmt=fields_formats, header=table_header, comments='')
