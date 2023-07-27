import numpy as np

def identify_neurons_according_test_err(test_errors_file, num_neurons, method = 'most significant', focus = "all"):
    '''
    test_errors_file: the file contains test errors from leision one neurons, leisioned by zeroing out W_in, W_rc, W_fb, NOT W_out (to be tested with W_out)
    num_lesion_neurons: the number of neurons to be lesioned
    method: one of "most significant" / "least significant" / "random"
    '''
    error_lesions = np.load(test_errors_file, allow_pickle=True)
    single_neuron_significance_metrics = {}
    error_0s = []
    error_1s = []
    error_2s = []
    error_wholes = []

    for d in error_lesions.item(): 
        #print(error_lesions.item()[d]['error0'])
        error_0s.append(error_lesions.item()[d]['error0'])
        error_1s.append(error_lesions.item()[d]['error1'])
        error_2s.append(error_lesions.item()[d]['error2'])
        error_wholes.append(error_lesions.item()[d]['error_whole'])

    single_neuron_significance_metrics['error_0s'] = np.array(error_0s)
    single_neuron_significance_metrics['error_1s'] = np.array(error_1s)
    single_neuron_significance_metrics['error_2s'] = np.array(error_2s)
    single_neuron_significance_metrics['error_wholes'] = np.array(error_wholes)

    # sort the neurons by its effect on different tasks
    # Note: the effect/significance were evaluated as the TEST ERROR on the model when lesion that single neuron
    # sorted by most --> least significant
    single_neuron_significance_metrics["output0_sorted_neurons"] = np.argsort(single_neuron_significance_metrics['error_0s'])[::-1]
    single_neuron_significance_metrics["output1_sorted_neurons"] = np.argsort(single_neuron_significance_metrics['error_1s'])[::-1]
    single_neuron_significance_metrics["output2_sorted_neurons"] = np.argsort(single_neuron_significance_metrics['error_2s'])[::-1]
    single_neuron_significance_metrics["output_all_sorted_neurons"] = np.argsort(single_neuron_significance_metrics['error_wholes'])[::-1]

    if focus == 'output0': 
        if method == 'least significant': 
            lesion_neurons =  single_neuron_significance_metrics["output0_sorted_neurons"][-num_neurons:]
        elif method == 'random': 
            lesion_neurons =  np.random.choice(single_neuron_significance_metrics["output0_sorted_neurons"], size=num_neurons, replace=False)
        else: 
            lesion_neurons =  single_neuron_significance_metrics["output0_sorted_neurons"][:num_neurons]

    elif focus == 'output1':
        if method == 'least significant': 
            lesion_neurons =  single_neuron_significance_metrics["output1_sorted_neurons"][-num_neurons:]
        elif method == 'random': 
            lesion_neurons =  np.random.choice(single_neuron_significance_metrics["output1_sorted_neurons"], size=num_neurons, replace=False)
        else: 
            lesion_neurons =  single_neuron_significance_metrics["output1_sorted_neurons"][:num_neurons]

    elif focus == 'ouput2':
        if method == 'least significant': 
            lesion_neurons =  single_neuron_significance_metrics["output2_sorted_neurons"][-num_neurons:]
        elif method == 'random': 
            lesion_neurons =  np.random.choice(single_neuron_significance_metrics["output2_sorted_neurons"], size=num_neurons, replace=False)
        else: 
            lesion_neurons =  single_neuron_significance_metrics["output2_sorted_neurons"][:num_neurons]


    else:
        if method == 'least significant': 
            lesion_neurons =  single_neuron_significance_metrics["output_all_sorted_neurons"][-num_neurons:]
        elif method == 'random': 
            lesion_neurons =  np.random.choice(single_neuron_significance_metrics["output_all_sorted_neurons"], size=num_neurons, replace=False)
        else: 
            lesion_neurons =  single_neuron_significance_metrics["output_all_sorted_neurons"][:num_neurons]

    return single_neuron_significance_metrics, lesion_neurons