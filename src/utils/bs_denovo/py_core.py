from multiprocessing import Pool
import itertools

def multiproc_task_on_list(task, list_input, n_jobs):
    """
    Apply a function to each element in a list using multiprocessing.

    :param task: A callable that is applied to each element of the input list.
    :param list_input: A list of input elements to be processed.
    :param n_jobs: Number of subprocesses to use.
    :return: A list of outputs where each element is the result of applying `task` to an element of `list_input`.
    """
    proc_pool = Pool(n_jobs)
    list_output = proc_pool.map(task, list_input)
    proc_pool.close()
    return list_output

def pairwise_tupled_ops(task, list1, list2, n_jobs):
    """
    Apply a function to all pairwise combinations of two lists using multiprocessing.
    Each input pair is formed as a tuple from the Cartesian product of `list1` and `list2`.

    :param task: A callable that takes a tuple (elem_from_list1, elem_from_list2) and returns a value.
    :param list1: First list to pair from (used as row elements).
    :param list2: Second list to pair from (used as column elements).
    :param n_jobs: Number of subprocesses to use.
    :return: A matrix (list of lists) where each row corresponds to an element in `list1` and each column to `list2`.
    """
    rs, cs = len(list1), len(list2)  # row size, column size
    tup_list = list(itertools.product(list1, list2))
    flat_paired = multiproc_task_on_list(task, tup_list, n_jobs)
    re_matrix = []
    for i in range(rs):
        row_start = cs*i
        re_matrix.append(flat_paired[row_start:(row_start+cs)])
    return re_matrix
    