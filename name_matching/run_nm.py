import modin.pandas as pd
from name_matching.name_matcher import NameMatcher
from typing import Union, Tuple
from unicodedata import normalize
import warnings
import os

os.environ["MODIN_ENGINE"] = "ray"
warnings.filterwarnings("ignore", category=UserWarning)


def _match_names_check_data(
    data: Union[pd.Series, pd.DataFrame], column: str, group_column: str
) -> pd.DataFrame:
    """
    Checks the input data of the name matching function to see whether the defined columns can
    be found and makes a new column which will be used for the name matching
    ----------
    data: Union[pd.DataFrame, pd.Series]
        The first dataframe or series used for the name matching
    column: str
        The column in which the name that should be matched can be found for data
    group_column_first: str
        The name of the column that should be used to generate groups within the data.

    Returns
    -------
    pd.DataFrame
        A dataframe containing data and an additional column 'name_matching_data' containing the
        names which should be matched
    """

    if isinstance(data, pd.DataFrame):
        if column == "":
            raise ValueError(
                "For one of the dataframes no column is given to perform the name matching on"
            )
        if column not in data.columns:
            raise ValueError("Could not find one of the columns in the dataframe")
        if (group_column != "") & (group_column not in data.columns):
            raise ValueError("Could not find one of the group_columns in the dataframe")
        return pd.DataFrame({
            "name_matching_data": data[column],
            **{col: data[col] for col in data.columns if col != column}
        })
    else:
        if group_column != "":
            raise ValueError(
                "Grouping is only possible when a dataframe is used for both inputs"
            )
        return pd.DataFrame(data, columns=["name_matching_data"])


def _match_names_preprocess_data(
    column: str,
    data_first: pd.DataFrame,
    data_second: pd.DataFrame,
    case_sensitive: bool,
    punctuation_sensitive: bool,
    special_character_sensitive: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Preprocess the data by making the names lower case, removing punctuations and special characters.
    And convert the indexes of the second dataframe to a column.
    ----------
    data: Union[pd.DataFrame, pd.Series]
        The first dataframe or series used for the name matching
    column: str
        The column in which the name that should be matched can be found for data
    group_column_first: str
        The name of the column that should be used to generate groups within the data.
    case_sensitive: bool
        Boolean value indicating whether the names should be converted to lower case names
        before the name matching starts. If False all the characters are converted to lowercase
    punctuation_sensitive: bool
        Boolean value indicating whether punctuations should be removed from the original names
        before the name matching starts. If False the punctuations are removed
    special_character_sensitive: bool
        Boolean value indicating whether special characters should be converted to unicode
        before the name matching starts. If False the special characters are replaced

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        A dataframe containing data and an additional column 'name_matching_data' containing the
        names which should be matched
    """

    processed_first = data_first.copy()
    processed_second = data_second.copy()

    if not case_sensitive:
        processed_first[column] = (
            processed_first[column].str.lower().str.strip()
        )
        processed_second[column] = (
            processed_second[column].str.lower().str.strip()
        )

    if not punctuation_sensitive:
        pattern = r"[^\w\s]"
        processed_first[column] = (
            processed_first[column].str.replace(pattern, "", regex=True)
        )
        processed_second[column] = (
            processed_second[column].str.replace(pattern, "", regex=True)
        )

    if not special_character_sensitive:
        normalize_func = lambda x: normalize("NFKD", x).encode("ASCII", "ignore").decode()
        processed_first[column] = processed_first[column].map(normalize_func)
        processed_second[column] = processed_second[column].map(normalize_func)

    processed_second = processed_second.reset_index(drop=False)

    return processed_first, processed_second


def _match_names_combine_data(
    data_first: pd.DataFrame,
    data_second: pd.DataFrame,
    left_cols: list,
    right_cols: list,
) -> pd.DataFrame:
    """
    Perform a merge to match data based on whether the names are equal
    ----------
    data_first: pd.DataFrame
        The first dataframe or series used for the name matching
    data_second: pd.DataFrame
        The second dataframe or series used for the name matching
    left_cols: list
        A list of columns on which the first dataframe should be merged
    right_cols: list
        A list of columns on which the first dataframe should be merged

    Returns
    -------
    pd.DataFrame
        A dataframe containing the original name, matched name, match score and match index. The index of the
        dataframe is equal to the original index of data_first, the match index is the index in data_second
        for the matched name.
    """
    matches = pd.merge(
        data_first,
        data_second,
        how="left",
        left_on=left_cols,
        right_on=right_cols,
        suffixes=["", "_matched"],
    )
    return (matches
            .assign(score=100)
            .dropna(subset=["index"])
            .rename(columns={"index": "match_index"})
            [["match_index", "score"]])


def _match_names_match_single(
    matcher: NameMatcher,
    data_first: pd.DataFrame,
    data_second: pd.DataFrame,
    name_column: str,
) -> pd.DataFrame:
    """
    Perform the name matching. First by doing a perfect string match with a merge statement, followed
    by the fuzzy matching approach as done in NameMatcher.
    ----------
    matcher: NameMatcher
        The NameMatcher to be used for the name matching part
    data_first: pd.DataFrame
        The first dataframe or series used for the name matching
    data_second: pd.DataFrame
        The second dataframe or series used for the name matching
    name_column: str
        The column in which the name that should be matched can be found for both dataframes

    Returns
    -------
    pd.DataFrame
        A dataframe containing the original name, matched name, match score and match index. The index of the
        dataframe is equal to the original index of data_first, the match index is the index in data_second
        for the matched name.
    """

    matches = _match_names_combine_data(
        data_first, data_second, [name_column], [name_column]
    )
    unmatched_idx = data_first.index.difference(matches.index)
    if len(unmatched_idx) > 0:
        unmatched = data_first.loc[unmatched_idx]
        matcher.load_and_process_master_data(name_column, data_second, transform=True)
        new_matches = matcher.match_names(
            to_be_matched=unmatched,
            column_matching=name_column
        )
        return pd.concat([matches, new_matches])
    
    print("All data matched with basic string matching")
    return matches


def _match_names_match_group(
    matcher: NameMatcher,
    data_first: pd.DataFrame,
    data_second: pd.DataFrame,
    name_column: str,
    group_column_first: str,
    group_column_second: str,
) -> pd.DataFrame:
    """
    Perform the name matching based on the subgroups as indicated by the group_column strings. First by doing
    a perfect string match with a merge statement, followed by the fuzzy matching approach as done in NameMatcher.
    ----------
    matcher: NameMatcher
        The NameMatcher to be used for the name matching part
    data_first: pd.DataFrame
        The first dataframe or series used for the name matching
    data_second: pd.DataFrame
        The second dataframe or series used for the name matching
    name_column: str
        The column in which the name that should be matched can be found for both dataframes
    group_column_first: str
        The name of the column that should be used to generate groups within the data.
    group_column_second: str
        The name of the column that should be used to generate groups within the data.

    Returns
    -------
    pd.DataFrame
        A dataframe containing the original name, matched name, match score and match index. The index of the
        dataframe is equal to the original index of data_first, the match index is the index in data_second
        for the matched name.
    """

    matches = _match_names_combine_data(
        data_first,
        data_second,
        [name_column, group_column_first],
        [name_column, group_column_second],
    )
    unmatched_idx = data_first.index.difference(matches.index)
    unmatched = data_first.loc[unmatched_idx]
    
    if len(unmatched) > 0:
        matcher.load_and_process_master_data(name_column, data_second, transform=False)
        
        unique_groups = data_first[group_column_first].unique()
        group_matches = []
        
        for group in unique_groups:
            group_mask = data_second[group_column_second] == group
            data_second_group = data_second[group_mask]
            
            if len(data_second_group) > 0:
                matcher.load_and_process_master_data(
                    name_column, data_second_group, start_processing=False
                )
                matcher.transform_data()
                
                unmatched_group = unmatched[
                    unmatched[group_column_first] == group
                ]
                
                if len(unmatched_group) > 0:
                    group_matches.append(
                        matcher.match_names(
                            to_be_matched=unmatched_group,
                            column_matching=name_column,
                        )
                    )
        
        if group_matches:
            matches = pd.concat([matches] + group_matches)
    
    return matches


def match_names(
    data_first: Union[pd.DataFrame, pd.Series],
    data_second: Union[pd.DataFrame, pd.Series],
    column_first="",
    column_second="",
    group_column_first="",
    group_column_second="",
    case_sensitive=False,
    punctuation_sensitive=False,
    special_character_sensitive=False,
    threshold=95,
    **kwargs
) -> pd.DataFrame:
    """Function which performs name matching. First a simple merge on the data is performed
    to get the instances in which the name matches perfectly. Subsequently the matches are
    matched using the name matching algorithm as defined in name_matcher.

    Parameters
    ----------
    data_first: Union[pd.DataFrame, pd.Series]
        The first dataframe or series used for the name matching
    data_second: Union[pd.DataFrame, pd.Series]
        The second dataframe or series used for the name matching, for matching the data to
        itself data_second should be equal to data first
    column_first: str
        If data_first is a dataframe column_first should be the column in which the name
        that should be matched can be found for data_first
        default=''
    column_second: str
        If data_second is a dataframe column_second should be the column in which the name
        that should be matched can be found for data_second
        default=''
    group_column_first: str
        The name of the column that should be used to generate groups within the data_first
        dataframe. The matchig is then only performed for instances in which the groups are
        identical
        default=''
    group_column_second: str
        The name of the column that should be used to generate groups within the data_second
        dataframe. The matchig is then only performed for instances in which the groups are
        identical
        default=''
    case_sensitive: bool
        Boolean value indicating whether the names should be converted to lower case names
        before the name matching starts. If False all the characters are converted to lowercase
        default=False
    punctuation_sensitive: bool
        Boolean value indicating whether punctuations should be removed from the original names
        before the name matching starts. If False the punctuations are removed
        default=False
    special_character_sensitive: bool
        Boolean value indicating whether special characters should be converted to unicode
        before the name matching starts. If False the special characters are replaced
        default=False
    threshold: int
        the minimal score a match should have to be part of the output
        default=95
    **kwargs
        Additional inputs for the name_matcher

    Returns
    -------
    pd.DataFrame
        A dataframe containing the matched rows were the match score is above the threshold. The
        dataframe consists of 4 columns; original_name: the original name from data_first after
        preprocessing, match_name_0: the name it is matched to from data_second after preprocessing,
        score_0: the score of the match, match_index_0: the index of the match in data_second. The
        match_index_0 can be used to join the data from both dataframes.
    """
    if "number_of_matches" in kwargs:
        raise ValueError(
            "The number of matches can only be changed by using a custom matching approach"
        )

    data_first = _match_names_check_data(data_first, column_first, group_column_first)
    data_second = _match_names_check_data(
        data_second, column_second, group_column_second
    )

    name_column = "name_matching_data"

    if ((group_column_first == "") & (group_column_second != "")) | (
        (group_column_second == "") & (group_column_first != "")
    ):
        raise ValueError(
            "For the grouping to work both the grouping column in the first as well as the second dataframe have to be indicated"
        )

    if (threshold > 100) | (threshold < 0):
        raise ValueError("Please pick a threshold between 0 and 100")

    data_first, data_second = _match_names_preprocess_data(
        name_column,
        data_first,
        data_second,
        case_sensitive,
        punctuation_sensitive,
        special_character_sensitive,
    )

    matcher = NameMatcher(**kwargs)

    if group_column_first == "":
        matches = _match_names_match_single(
            matcher, data_first, data_second, name_column
        )
    else:
        matches = _match_names_match_group(
            matcher,
            data_first,
            data_second,
            name_column,
            group_column_first,
            group_column_second,
        )

    return matches[matches["score"] > threshold]
