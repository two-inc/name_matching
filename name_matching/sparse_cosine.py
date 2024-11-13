import numpy as np
from scipy.sparse import csr_matrix, csc_matrix
from typing import Union, Tuple, Optional
from numba import jit
import warnings
import logging

logger = logging.getLogger(__name__)

@jit(nopython=True)
def _compute_cosine_similarity_numba(
    indptr_a: np.ndarray,
    indices_a: np.ndarray,
    data_a: np.ndarray,
    indptr_b: np.ndarray,
    indices_b: np.ndarray,
    data_b: np.ndarray,
    top_n: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Optimized sparse cosine similarity with Numba
    """
    n_rows_a = len(indptr_a) - 1
    n_rows_b = len(indptr_b) - 1
    
    scores = np.zeros((n_rows_b, n_rows_a), dtype=np.float32)
    result_indices = np.zeros((n_rows_b, top_n), dtype=np.int32)
    result_scores = np.zeros((n_rows_b, top_n), dtype=np.float32)
    
    # Pre-compute norms for matrix_a
    norms_a = np.zeros(n_rows_a, dtype=np.float32)
    for i in range(n_rows_a):
        start_a = indptr_a[i]
        end_a = indptr_a[i + 1]
        for k in range(start_a, end_a):
            norms_a[i] += data_a[k] * data_a[k]
        norms_a[i] = np.sqrt(norms_a[i])
    
    # Pre-compute norms for matrix_b
    norms_b = np.zeros(n_rows_b, dtype=np.float32)
    for i in range(n_rows_b):
        start_b = indptr_b[i]
        end_b = indptr_b[i + 1]
        for k in range(start_b, end_b):
            norms_b[i] += data_b[k] * data_b[k]
        norms_b[i] = np.sqrt(norms_b[i])
    
    # Compute similarities
    for i in range(n_rows_b):
        start_b = indptr_b[i]
        end_b = indptr_b[i + 1]
        
        if norms_b[i] == 0:
            continue
            
        for j in range(n_rows_a):
            if norms_a[j] == 0:
                continue
                
            start_a = indptr_a[j]
            end_a = indptr_a[j + 1]
            
            # Compute dot product
            dot_product = 0.0
            pa = start_a
            pb = start_b
            
            while pa < end_a and pb < end_b:
                if indices_a[pa] == indices_b[pb]:
                    dot_product += data_a[pa] * data_b[pb]
                    pa += 1
                    pb += 1
                elif indices_a[pa] < indices_b[pb]:
                    pa += 1
                else:
                    pb += 1
            
            scores[i, j] = dot_product / (norms_a[j] * norms_b[i])
        
        # Get top_n indices for this row
        top_indices = np.argsort(scores[i])[-top_n:][::-1]
        result_indices[i] = top_indices
        result_scores[i] = scores[i, top_indices]
    
    return result_indices, result_scores

def _estimate_memory_requirement(matrix_a: csr_matrix, matrix_b: csr_matrix) -> float:
    """Estimate memory requirement in GB for different methods"""
    dense_size = (matrix_a.shape[0] * matrix_a.shape[1] + 
                 matrix_b.shape[0] * matrix_b.shape[1]) * 4 / (1024**3)  # 4 bytes per float
    sparse_size = (matrix_a.data.nbytes + matrix_b.data.nbytes) / (1024**3)
    return dense_size, sparse_size

def select_best_method(
    matrix_a: csr_matrix,
    matrix_b: csr_matrix,
    available_memory_gb: Optional[float] = None
) -> str:
    """
    Intelligently select the best method based on data characteristics
    """
    if available_memory_gb is None:
        import psutil
        available_memory_gb = psutil.virtual_memory().available / (1024**3)

    density = (matrix_a.nnz / (matrix_a.shape[0] * matrix_a.shape[1]) +
              matrix_b.nnz / (matrix_b.shape[0] * matrix_b.shape[1])) / 2
    dense_mem_required, sparse_mem_required = _estimate_memory_requirement(matrix_a, matrix_b)
    
    logger.info(f"Matrix density: {density:.3f}")
    logger.info(f"Available memory: {available_memory_gb:.2f} GB")
    logger.info(f"Dense memory required: {dense_mem_required:.2f} GB")
    logger.info(f"Sparse memory required: {sparse_mem_required:.2f} GB")

    # Decision tree for method selection
    try:
        import faiss
        has_faiss = True
    except ImportError:
        has_faiss = False
        logger.warning("Faiss not available")

    try:
        from annoy import AnnoyIndex
        has_annoy = True
    except ImportError:
        has_annoy = False
        logger.warning("Annoy not available")

    # Decision logic
    if density > 0.1 and dense_mem_required < available_memory_gb * 0.8 and has_faiss:
        logger.info("Selected method: faiss (dense matrix, sufficient memory)")
        return 'faiss'
    
    if matrix_a.shape[0] > 1e6 or matrix_b.shape[0] > 1e6:
        if has_annoy:
            logger.info("Selected method: annoy (large dataset)")
            return 'annoy'
        logger.info("Large dataset but Annoy not available")

    if sparse_mem_required < available_memory_gb * 0.5:
        logger.info("Selected method: sparse_numba (sparse matrix, sufficient memory)")
        return 'sparse_numba'
    
    if has_annoy:
        logger.info("Selected method: annoy (limited memory)")
        return 'annoy'
    
    logger.warning("Falling back to sparse_numba implementation with potential memory issues")
    return 'sparse_numba'

def sparse_cosine_top_n(
    matrix_a: Union[csr_matrix, csc_matrix],
    matrix_b: Union[csr_matrix, csc_matrix],
    top_n: int,
    method: str = 'auto',
    available_memory_gb: Optional[float] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Improved sparse cosine similarity with automatic method selection
    
    Parameters
    ----------
    matrix_a : sparse matrix
        First sparse matrix
    matrix_b : sparse matrix
        Second sparse matrix
    top_n : int
        Number of top matches to return
    method : str
        One of 'auto', 'faiss', 'annoy', 'sparse_numba'
    available_memory_gb : float, optional
        Available memory in GB. If None, will be detected automatically.
    
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (indices, scores) of top matches
    """
    # Input validation
    if not isinstance(matrix_a, (csr_matrix, csc_matrix)):
        raise TypeError("matrix_a must be sparse matrix")
    if not isinstance(matrix_b, (csr_matrix, csc_matrix)):
        raise TypeError("matrix_b must be sparse matrix")
    if matrix_a.shape[1] != matrix_b.shape[1]:
        raise ValueError("Matrices must have same number of columns")
    if top_n > matrix_a.shape[0]:
        raise ValueError("top_n cannot be larger than number of rows in matrix_a")

    # Method selection
    if method == 'auto':
        method = select_best_method(matrix_a, matrix_b, available_memory_gb)

    # Method implementation
    if method == 'faiss':
        try:
            return faiss_cosine_similarity(matrix_a, matrix_b, top_n)
        except Exception as e:
            logger.error(f"Faiss implementation failed: {str(e)}")
            logger.warning("Falling back to sparse_numba implementation")
            method = 'sparse_numba'
    
    if method == 'annoy':
        try:
            return annoy_cosine_similarity(matrix_a, matrix_b, top_n)
        except Exception as e:
            logger.error(f"Annoy implementation failed: {str(e)}")
            logger.warning("Falling back to sparse_numba implementation")
            method = 'sparse_numba'
    
    if method == 'sparse_numba':
        # Ensure CSR format for efficient row access
        matrix_a = matrix_a.tocsr()
        matrix_b = matrix_b.tocsr()
        return _compute_cosine_similarity_numba(
            matrix_a.indptr, matrix_a.indices, matrix_a.data,
            matrix_b.indptr, matrix_b.indices, matrix_b.data,
            top_n
        )
    
    raise ValueError(f"Unknown method: {method}")

def annoy_cosine_similarity(matrix_a, matrix_b, top_n):
    """
    Memory efficient implementation using Annoy
    """
    dim = matrix_a.shape[1]
    index = AnnoyIndex(dim, 'angular')  # Angular distance = cosine distance
    
    # Add items to index
    for i in range(matrix_a.shape[0]):
        index.add_item(i, matrix_a[i])
    
    index.build(10)  # 10 trees - more trees = better accuracy but slower
    
    # Search
    results = []
    for i in range(matrix_b.shape[0]):
        results.append(index.get_nns_by_vector(matrix_b[i], top_n))
    
    return np.array(results)

def faiss_cosine_similarity(matrix_a, matrix_b, top_n):
    """
    Much faster implementation using Faiss
    """
    # Convert sparse to dense if needed
    if isinstance(matrix_a, csr_matrix):
        matrix_a = matrix_a.toarray()
    if isinstance(matrix_b, csr_matrix):
        matrix_b = matrix_b.toarray()
    
    # Normalize vectors for cosine similarity
    faiss.normalize_L2(matrix_a)
    faiss.normalize_L2(matrix_b)
    
    # Build index
    index = faiss.IndexFlatIP(matrix_a.shape[1])  # Inner product = cosine similarity for normalized vectors
    index.add(matrix_a)
    
    # Search
    distances, indices = index.search(matrix_b, top_n)
    return indices, distances
