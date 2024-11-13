import modin.pandas as pd
import numpy as np
from tqdm.auto import tqdm
import psutil
import logging
from pathlib import Path
from datetime import datetime
import os
from typing import Union, Tuple
from functools import reduce
from unicodedata import normalize
from re import escape, sub
from itertools import compress
from sklearn.feature_extraction.text import TfidfVectorizer
from name_matching.distance_metrics import make_distance_metrics
from cleanco.termdata import terms_by_type, terms_by_country
from name_matching.sparse_cosine import sparse_cosine_top_n
from memory_profiler import profile

# Configure logging
def setup_logging(log_dir: str = "logs") -> None:
    """Setup logging configuration"""
    # Create logs directory if it doesn't exist
    Path(log_dir).mkdir(exist_ok=True)
    
    # Create logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    
    # Create handlers
    # File handler with rotation
    log_file = Path(log_dir) / f"name_matcher_{datetime.now():%Y%m%d}.log"
    file_handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    file_handler.setLevel(logging.INFO)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)  # Only warnings and errors to console
    
    # Create formatters and add it to handlers
    file_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_format = logging.Formatter('%(levelname)s - %(message)s')
    
    file_handler.setFormatter(file_format)
    console_handler.setFormatter(console_format)
    
    # Add handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

class NameMatcher:
    """
    A class for the name matching of data based on the strings in a single column. The 
    NameMatcher first applies a cosine similarity on the ngrams of the strings to get 
    an approximate match followed by a fuzzy matching based on a number of different 
    algorithms.

    Parameters
    ----------
    ngrams : tuple of integers
        The length of the ngrams which should be used for the generation of ngrams for 
        the cosine similarity comparison of the possible matches
        default=(2, 3)
    top_n : integer
        The number of possible matches that should be included in the group which will 
        be analysed with the fuzzy matching algorithms
        default=50
    low_memory : bool
        Bool indicating if the a low memory approach should be taken in the sparse 
        cosine similarity step.
        default=False
    number_of_rows : integer
        Determines how many rows should be calculated at once with the sparse cosine 
        similarity step. If the low_memory bool is True this number is unused.
        default=5000
    number_of_matches : int
        The number of matches which should be returned by the matching algorithm. If a 
        number higher than 1 is given, a number of alternative matches are also returned.
        If the number is equal to the number of algorithms used, the best match for each 
        algorithm is returned. If the number is equal to the number of algorithm groups 
        which are included the best match for each group is returned.
        default=1
    legal_suffixes : bool
        Boolean indicating whether the most common company legal terms should be excluded 
        when calculating the final score. The terms are still included in determining the
        best match.
        default=False
    common_words : bool or list
        Boolean indicating whether the most common words from the matching data should be
        excluded when calculating the final score. The terms are still included in 
        determining the best match. If common_words is given as a list, the words in the
        list are excluded from the calculation of the final score, downgrading matches 
        that predominatly rely on these words.
        default=False
    cut_off_no_scoring_words: float
        the cut off percentage of the occurrence of the most occurring word for which words
        are still included in the no_scoring_words set
        default=0.01
    lowercase : bool
        A boolean indicating whether during the preprocessing all characters should be 
        converted to lowercase, to generate case insensitive matching
        default=True
    punctuations : bool
        A boolean indicating whether during the preprocessing all punctuations should be 
        ignored
        default=True
    remove_ascii : bool
        A boolean indicating whether during the preprocessing all characters should be 
        converted to ascii characters
        default=True : bool
    preprocess_split
        Indicating whether during the preprocessing an additional step should be taken in 
        which only the most common words out of a name are isolated and used in the 
        matching process. The removing of the common words is only done for the n-grams 
        cosine matching part.
        default=False
    verbose : bool
        A boolean indicating whether progress printing should be done
        default=True
    distance_metrics: list
        A list of The distance metrics to be used during the fuzzy matching. For a list of 
        possible distance metrics see the distance_metrics.py file. By default the 
        following metrics are used: overlap, weighted_jaccard, ratcliff_obershelp, 
        fuzzy_wuzzy_token_sort and editex.
    row_numbers : bool
        Bool indicating whether the row number should be used as match_index rather than 
        the original index as was the default case before version 0.8.8
        default=False
    return_algorithms_score : bool
        Bool indicating whether the scores of all the algorithms should be returned instead
        of a combined score
        default=False
    memory_threshold : float
        The threshold percentage of memory usage above which warnings are logged
        default=0.85
    log_dir : str
        The directory where the log files should be stored
        default="logs"
    """

    def __init__(
        self,
        number_of_rows: int = 100,
        number_of_matches: int = 1,
        top_n: int = 100,
        ngrams: tuple = (2, 3),
        low_memory: bool = False,
        lowercase: bool = True,
        punctuations: bool = True,
        remove_ascii: bool = True,
        legal_suffixes: bool = False,
        common_words: Union[bool, list] = False,
        cut_off_no_scoring_words: float = 0.01,
        preprocess_split: bool = False,
        verbose: bool = True,
        distance_metrics: Union[list, tuple] = [
            "overlap",
            "weighted_jaccard",
            "ratcliff_obershelp",
            "fuzzy_wuzzy_token_sort",
            "editex",
        ],
        row_numbers: bool = False,
        return_algorithms_score: bool = False,
        memory_threshold: float = 0.85,
        log_dir: str = "logs",
        modin_engine: str = "ray"
    ):
        """Initialize with Modin engine selection"""
        # Set Modin engine if not already set
        if not os.environ.get("MODIN_ENGINE"):
            os.environ["MODIN_ENGINE"] = modin_engine

        self._possible_matches = None
        self._preprocessed = False
        self._df_matching_data = pd.DataFrame()

        self._number_of_rows = number_of_rows
        self._low_memory = low_memory

        self._column = ""
        self._column_matching = ""

        self._verbose = verbose
        self._number_of_matches = number_of_matches
        self._top_n = top_n
        self._return_algorithms_score = return_algorithms_score

        self._preprocess_lowercase = lowercase
        self._preprocess_punctuations = punctuations
        self._preprocess_ascii = remove_ascii
        self._postprocess_company_legal_id = legal_suffixes

        if isinstance(common_words, bool):
            self._postprocess_common_words = common_words
            self._word_set = set()
        elif isinstance(common_words, (list, tuple, set)):
            self._postprocess_common_words = False
            self._word_set = set(common_words)
        else:
            raise TypeError("Please provide common_words as a list or a bool")

        self._preprocess_split = preprocess_split
        self._cut_off = cut_off_no_scoring_words

        if self._postprocess_company_legal_id:
            self._word_set = self._make_no_scoring_words(
                "legal", self._word_set, self._cut_off
            )

        self._original_indexes = not row_numbers
        self._original_index = None

        self.set_distance_metrics(distance_metrics)

        self._vec = TfidfVectorizer(
            lowercase=False, analyzer="char", ngram_range=(ngrams)
        )
        self._n_grams_matching = None

        self._memory_threshold = memory_threshold
        self._logger = setup_logging(log_dir)

    def set_distance_metrics(self, metrics: list) -> None:
        """
        A method to set which of the distance metrics should be employed during the
        fuzzy matching. For very short explanations of most of the name matching
        algorithms please see the make_distance_metrics function in distance_matrics.py

        Parameters
        ----------
        metrics: list
            The list with the distance metrics to be used during the name matching. The
            distance metrics can be chosen from the list below:
                indel
                discounted_levenshtein
                tichy
                cormodeL_z
                iterative_sub_string
                baulieu_xiii
                clement
                dice_asymmetricI
                kuhns_iii
                overlap
                pearson_ii
                weighted_jaccard
                warrens_iv
                bag
                rouge_l
                ratcliff_obershelp
                ncd_bz2
                fuzzy_wuzzy_partial_string
                fuzzy_wuzzy_token_sort
                fuzzy_wuzzy_token_set
                editex
                typo
                lig_3
                ssk
                refined_soundex
                double_metaphone
        """

        input_metrics = {str(metric).lower(): True for metric in metrics}
        try:
            self._distance_metrics = make_distance_metrics(**input_metrics)
        except TypeError:
            raise TypeError(
                "Not all of the supplied distance metrics are available. Please check the"
                + "list of options in the make_distance_metrics function and adjust" 
                + " your list accordingly"
            )
        self._num_distance_metrics = sum(
            [len(x) for x in self._distance_metrics.values()]
        )

    def _select_top_words(
        self, word: str, word_counts: pd.Series, occurrence_count: int
    ) -> str:
        """Vectorized version of word selection"""
        # Convert word list to series for vectorized operations
        word_series = pd.Series(word)
        mask = word_counts[word_series] < occurrence_count * word_counts[word_series].min()
        return " ".join(word_series[mask])

    def _preprocess_reduce(
        self, to_be_matched: pd.DataFrame, occurrence_count: int = 3
    ) -> pd.DataFrame:
        """Vectorized preprocessing with reduced string operations"""
        # Split once and reuse
        split_words = to_be_matched[self._column_matching].str.split()
        
        # Vectorized word counting
        individual_words = split_words.explode()
        word_counts = individual_words.value_counts()
        
        # Create new dataframe without copy
        to_be_matched_new = pd.DataFrame(index=to_be_matched.index)
        to_be_matched_new[self._column_matching] = split_words.apply(
            lambda word: self._select_top_words(word, word_counts, occurrence_count)
        )
        return to_be_matched_new

    def load_and_process_master_data(
        self,
        column: str,
        df_matching_data: pd.DataFrame,
        start_processing: bool = True,
        transform: bool = True,
    ) -> None:
        """Ensure Modin compatibility"""
        self._column = column
        # Avoid copy if possible
        self._df_matching_data = df_matching_data
        self._original_index = df_matching_data.index
        if start_processing:
            self._process_matching_data(transform)

    def _process_matching_data(self, transform: bool = True) -> None:
        """Function to process the matching data. First the matching data is preprocessed 
        and assigned to a variable within the NameMatcher. Next the data is used to 
        initialise the TfidfVectorizer.

        Parameters
        ----------
        transform : bool
            A boolean indicating whether or not the data should be transformed after the 
            vectoriser is initialised
            default: True
        """
        self._df_matching_data = self.preprocess(self._df_matching_data, self._column)
        if self._postprocess_common_words:
            self._word_set = self._make_no_scoring_words(
                "common", self._word_set, self._cut_off
            )
        self._vectorise_data(transform)
        self._preprocessed = True

    def match_names(
        self, to_be_matched: Union[pd.Series, pd.DataFrame], column_matching: str
    ) -> Union[pd.Series, pd.DataFrame]:
        """Process data in chunks while keeping data in Modin DataFrames"""
        if isinstance(to_be_matched, pd.Series):
            is_dataframe = False
            to_be_matched = pd.DataFrame({column_matching: to_be_matched})

        total_rows = len(to_be_matched)
        chunk_size = self._get_optimal_chunk_size(total_rows)
        num_chunks = (total_rows + chunk_size - 1) // chunk_size

        # Create progress bars
        outer_pbar = tqdm(total=num_chunks, desc="Processing chunks", 
                         disable=not self._verbose)
        results = []

        try:
            for chunk_idx in range(num_chunks):
                # Monitor memory
                self._check_memory_usage()
                
                # Get chunk using Modin's partition capabilities
                chunk_start = chunk_idx * chunk_size
                chunk_end = min((chunk_idx + 1) * chunk_size, total_rows)
                
                # Use Modin's query capabilities instead of iloc
                chunk = to_be_matched.loc[chunk_start:chunk_end-1]
                
                # Process chunk
                chunk = self.preprocess(chunk, self._column_matching)
                possible_matches = self._search_for_possible_matches(chunk)
                
                if self._preprocess_split:
                    reduced_matches = self._search_for_possible_matches(
                        self._preprocess_reduce(chunk)
                    )
                    # Convert to Modin DataFrame instead of numpy array
                    possible_matches = pd.concat([
                        pd.DataFrame(reduced_matches),
                        pd.DataFrame(possible_matches)
                    ], axis=1)

                # Process matches using Modin's apply
                chunk_matches = chunk.apply(
                    lambda row: self.fuzzy_matches(
                        possible_matches.loc[row.name],
                        row
                    ),
                    axis=1,
                    result_type='expand'
                )
                
                results.append(chunk_matches)
                outer_pbar.update(1)
                
                # Log progress
                self._logger.info(
                    f"Processed chunk {chunk_idx + 1}/{num_chunks} "
                    f"({(chunk_idx + 1)/num_chunks*100:.1f}%)"
                )

        finally:
            outer_pbar.close()

        # Combine results using Modin's concat
        data_matches = pd.concat(results, axis=0)
        
        return self._finalize_results(data_matches, is_dataframe)

    def fuzzy_matches(
        self, possible_matches: np.array, to_be_matched: pd.Series
    ) -> pd.Series:
        """A method which performs the fuzzy matching between the data in the 
        to_be_matched series as well as the indicated indexes of the matching_data points
        which are possible matching candidates.

        Parameters
        ----------
        possible_matches : np.array
            An array containing the indexes of the matching data with potential matches
        to_be_matched : pd.Series
            The data which should be matched

        Returns
        -------
        pd.Series
            A series containing the match index from the matching_data dataframe. the name 
            in the to_be_matched data, the name to which the datapoint was matched and a 
            score between 0 (no match) and 100(perfect match) to indicate the quality of 
            the matches.
        """
        if len(possible_matches.shape) > 1:
            possible_matches = possible_matches[0]

        indexes = np.array(
            [
                [f"match_name_{num}", f"score_{num}", f"match_index_{num}"]
                for num in range(self._number_of_matches)
            ]
        ).flatten()
        match = pd.Series(index=np.append("original_name", indexes), dtype=object)
        match["original_name"] = to_be_matched[self._column_matching]
        list_possible_matches = self._df_matching_data.iloc[
            possible_matches.flatten(), :
        ][self._column].values

        match_score = self._score_matches(
            to_be_matched[self._column_matching], list_possible_matches
        )
        if self._return_algorithms_score:
            return match_score
        ind = self._rate_matches(match_score)

        for num, col_num in enumerate(ind):
            match[f"match_name_{num}"] = list_possible_matches[col_num]
            match[f"match_index_{num}"] = possible_matches[col_num]

        match = self._adjust_scores(match_score[ind, :], match)

        if len(self._word_set):
            match = self.postprocess(match)

        return match

    def _score_matches(
        self, to_be_matched_instance: str, possible_matches: list
    ) -> np.array:
        """A method to score a name to_be_matched_instance to a list of possible matches. 
        The scoring is done based on all the metrics which are enabled.

        Parameters
        ----------
        to_be_matched_instance : str
            The name which should match one of the possible matches
        possible_matches : list
            list of the names of the possible matches

        Returns
        -------
        np.array
            The score of each of the matches with respect to the different metrics which 
            are assessed.
        """
        match_score = np.zeros((len(possible_matches), self._num_distance_metrics))
        idx = 0
        for method_list in self._distance_metrics.values():
            for method in method_list:
                match_score[:, idx] = np.array(
                    [method.sim(to_be_matched_instance, s) for s in possible_matches]
                )
                idx = idx + 1

        return match_score

    def _rate_matches(self, match_score: np.array) -> np.array:
        """Converts the match scores from the score_matches method to a list of indexes of 
        the best scoring matches limited to the _number_of_matches.

        Parameters
        ----------
        match_score : np.array
            An array containing the scores of each of the possible alternatives for each
            of the different methods used

        Returns
        -------
        np.array
            The indexes of the best rated matches
        """
        if self._number_of_matches == 1:
            ind = [np.argmax(np.mean(match_score, axis=1))]
        elif self._number_of_matches == len(self._distance_metrics):
            ind = np.zeros(len(self._distance_metrics))
            idx = 0
            for num, method_list in enumerate(self._distance_metrics.values()):
                method_grouped_results = np.reshape(
                    match_score[:, idx : idx + len(method_list)], (-1, len(method_list))
                )
                ind[num] = np.argmax(np.mean(method_grouped_results, axis=1))
                idx = idx + len(method_list)
        elif self._number_of_matches == self._num_distance_metrics:
            ind = np.argmax(match_score, axis=0).reshape(-1)
        else:
            ind = np.argsort(np.mean(match_score, axis=1))[-self._number_of_matches :][
                ::-1
            ]

        return np.array(ind, dtype=int)

    def _get_alternative_names(self, match: pd.Series) -> list:
        """Gets all the possible match names from the match.

        Parameters
        ----------
        match : pd.Series
            The series with the possible matches and original scores

        Returns
        -------
        list
            A list with the alternative names for this match
        """
        alt_names = []

        for num in range(self._number_of_matches):
            alt_names.append(str(match[f"match_name_{num}"]))

        return alt_names

    def _process_words(self, org_name: str, alt_names: list) -> Tuple[str, list]:
        """Removes the words from the word list from the org_name and all the names in 
        alt_names .

        Parameters
        ----------
        org_name : str
            The original name for the matching data
        alt_names : list
            A list of names from which the words should be removed

        Returns
        -------
        Tuple[str, list]
            The processed version of the org_name and the alt_names, with the words
            removed
        """
        len_atl_names = len(alt_names)
        for word in self._word_set:
            org_name = " ".join(sub(rf"\b{escape(word)}\b", "", org_name).split())
            for num in range(len_atl_names):
                alt_names[num] = " ".join(
                    sub(rf"\b{escape(word)}\b", "", alt_names[num]).split()
                )

        return org_name, alt_names

    def _adjust_scores(self, match_score: np.array, match: pd.Series) -> pd.Series:
        """Adjust the scores to be between 0 and 100

        Parameters
        ----------
        match_score : np.array
            An array with the scores for each of the options
        match : pd.Series
            The series with the possible matches and original scores

        Returns
        -------
        pd.Series
            The series with the possible matches and adjusted scores
        """
        for num in range(self._number_of_matches):
            match[f"score_{num}"] = 100 * np.mean(match_score[num, :])

        return match

    def postprocess(self, match: pd.Series) -> pd.Series:
        """Postprocesses the scores to exclude certain specific company words or the 
        most common words. In this method only the scores are adjusted, the matches 
        still stand.

        Parameters
        ----------
        match : pd.Series
            The series with the possible matches and original scores

        Returns
        -------
        pd.Series
            A new version of the input series with updated scores
        """
        alt_names = self._get_alternative_names(match)
        org_name = str(match["original_name"])

        org_name, alt_names = self._process_words(org_name, alt_names)

        match_score = self._score_matches(org_name, alt_names)
        ind = self._rate_matches(match_score)

        match = self._adjust_scores(match_score[ind, :], match)

        return match

    def _vectorise_data(self, transform: bool = True):
        """Initialises the TfidfVectorizer, which generates ngrams and weights them 
        based on the occurrance. Subsequently the matching data will be used to fit 
        the vectoriser and the matching data might also be send to the transform_data 
        function depending on the transform boolean.

        Parameters
        ----------
        transform : bool
            A boolean indicating whether or not the data should be transformed after the 
            vectoriser is initialised
            default: True
        """
        self._vec.fit(self._df_matching_data[self._column].values.flatten())
        if transform:
            self.transform_data()

    def transform_data(self):
        """A method which transforms the matching data based on the ngrams transformer.
        After the transformation (the generation of the ngrams), the data is normalised 
        by dividing each row by the sum of the row. Subsequently the data is changed to 
        a coo sparse matrix format with the column indices in ascending order.
        """
        ngrams = self._vec.transform(self._df_matching_data[self._column].astype(str))
        for i, j in zip(ngrams.indptr[:-1], ngrams.indptr[1:]):
            ngrams.data[i:j] = ngrams.data[i:j] / np.sum(ngrams.data[i:j])
        self._n_grams_matching = ngrams.tocsc()
        if self._low_memory:
            self._n_grams_matching = self._n_grams_matching.tocoo()

    def _search_for_possible_matches(self, to_be_matched: pd.DataFrame) -> pd.DataFrame:
        """
        Use improved sparse cosine implementation with automatic method selection
        """
        if self._n_grams_matching is None:
            raise RuntimeError(
                "First transform the data using transform_data or "
                "load_and_process_master_data with transform=True"
            )

        # Transform data
        match_ngrams = self._vec.transform(
            to_be_matched[self._column_matching].to_numpy()
        )

        # Use low_memory setting to influence available memory
        available_memory_gb = None
        if self._low_memory:
            available_memory_gb = psutil.virtual_memory().available / (1024**3) * 0.5  # Use only 50% of available memory
        
        indices, scores = sparse_cosine_top_n(
            matrix_a=self._n_grams_matching,
            matrix_b=match_ngrams,
            top_n=self._top_n,
            method='auto',
            available_memory_gb=available_memory_gb
        )

        # Convert to DataFrame
        return pd.DataFrame(
            indices,
            index=to_be_matched.index
        )

    def _finalize_results(self, data_matches: pd.DataFrame, 
                         is_dataframe: bool) -> Union[pd.Series, pd.DataFrame]:
        """Finalize and format results"""
        if self._return_algorithms_score:
            return data_matches

        if self._number_of_matches == 1:
            data_matches = data_matches.rename(columns={
                "match_name_0": "match_name",
                "score_0": "score",
                "match_index_0": "match_index",
            })

        if is_dataframe and self._original_indexes:
            index_cols = data_matches.columns[
                data_matches.columns.str.contains("match_index")
            ]
            for col in index_cols:
                data_matches[col] = self._original_index[
                    data_matches[col].astype(int).fillna(0)
                ]

        return data_matches

    def _check_memory_usage(self) -> float:
        """Monitor memory usage and log warnings"""
        memory_used = psutil.Process().memory_percent()
        if memory_used > self._memory_threshold:
            self._logger.warning(
                f"High memory usage detected: {memory_used:.1f}% "
                f"(threshold: {self._memory_threshold*100}%)"
            )
        return memory_used

    def _get_optimal_chunk_size(self, total_size: int) -> int:
        """Dynamically determine chunk size based on available memory"""
        available_memory = psutil.virtual_memory().available
        memory_per_row = psutil.Process().memory_info().rss / total_size
        
        # Target using 20% of available memory per chunk
        target_chunk_size = int(0.2 * available_memory / memory_per_row)
        return min(max(1000, target_chunk_size), total_size)

    def preprocess(self, df: pd.DataFrame, column_name: str) -> pd.DataFrame:
        """Vectorized preprocessing"""
        # Avoid copy by creating new dataframe
        processed = pd.DataFrame(index=df.index)
        series = df[column_name].astype(str)
        
        if self._preprocess_lowercase:
            series = series.str.lower()
            
        if self._preprocess_punctuations:
            # Combine regex operations
            series = (series.str.replace(r"[^\w\s]", "", regex=True)
                           .str.replace(r"\s+", " ")
                           .str.strip())
            
        if self._preprocess_ascii:
            # Vectorized normalization
            series = series.apply(
                lambda x: normalize("NFKD", str(x))
                .encode("ASCII", "ignore")
                .decode()
            )
            
        processed[column_name] = series
        return processed

    def _preprocess_word_list(self, terms: dict) -> list:
        """Preprocess legal words to remove punctuations and trailing leading space

        Parameters
        -------
        terms: dict
            a dictionary of legal words

        Returns
        -------
        list
            A list of preprocessed legal words
        """
        if self._preprocess_punctuations:
            return [
                sub(r"[^\w\s]", "", s).strip()
                for s in reduce(iconcat, terms.values(), [])
            ]
        else:
            return [s.strip() for s in reduce(iconcat, terms.values(), [])]

    def _process_legal_words(self, word_set: set) -> set:
        """Preprocess legal words and add them to the word_set

        Parameters
        -------
        word_set: str
            the current word list which should be extended with additional words

        Returns
        -------
        Set
            The original word_set with the legal words added
        """
        terms_type = self._preprocess_word_list(terms_by_type)
        terms_country = self._preprocess_word_list(terms_by_country)
        word_set = word_set.union(set(terms_country + terms_type))

        return word_set

    def _process_common_words(self, word_set: set, cut_off: float) -> set:
        """Vectorized word counting"""
        words = (self._df_matching_data[self._column]
                .str.split()
                .explode())
        word_counts = words.value_counts()
        threshold = word_counts.max() * cut_off
        common_words = word_counts[word_counts > threshold].index
        return word_set.union(set(common_words))

    def _make_no_scoring_words(
        self, indicator: str, word_set: set, cut_off: float
    ) -> set:
        """A method to make a set of words which are not taken into account when 
        scoring matches.

        Parameters
        -------
        indicator: str
            indicator for which types of words should be excluded can be legal for
            legal suffixes or common for the most common words
        word_set: str
            the current word list which should be extended with additional words
        cut_off: float
            the cut_off percentage of the occurrence of the most occurring word for 
            which words are still included in the no_soring_words set

        Returns
        -------
        Set
            The set of no scoring words
        """
        if indicator == "legal":
            word_set = self._process_legal_words(word_set)
        if indicator == "common":
            word_set = self._process_common_words(word_set, cut_off)

        return word_set
