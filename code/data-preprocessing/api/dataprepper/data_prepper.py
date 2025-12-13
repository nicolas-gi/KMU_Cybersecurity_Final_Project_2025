import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Binarizer, RobustScaler, LabelEncoder, OrdinalEncoder
from enum import Enum
import seaborn as sns
import matplotlib.pyplot as plt
from functools import wraps
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix

from typing import Optional, Tuple


#region COLORS
class Color(Enum):
    '''
    A simple enum to portrait text colorful in the console.
    '''
    RESET = "\033[0m"
    RED   = "\033[31m"
    GREEN = "\033[32m"
    YELLOW= "\033[33m"
    BLUE  = "\033[34m"
    CYAN  = "\033[36m"
    WHITE = "\033[37m"
#endregion

#region WRAPPERS
def use_self_df(df_adj):
    '''
    A wrapper which handles the case, when no dp is given: It assigns the classes initialization dp and automatically updates it.
    '''
    @wraps(df_adj)
    def _wrapped(self, df: pd.DataFrame | None = None, *args, **kwargs):
        use_self = df is None
        if use_self:
            df = self.df

        result = df_adj(self, df, *args, **kwargs)
        if isinstance(result, tuple):
            new_df, rest = result[0], result[1:]
        else:
            new_df, rest = result, None

        if use_self:
            self.df = new_df

        return (new_df, *rest) if rest else new_df
    return _wrapped

def adj_self_numerical(df_adj):
    '''
    A wrapper that causes the decorated function to update the number calls after execution.
    Needed when the numerical data is actually adjusted and altered within the decorated function.
    '''
    @wraps(df_adj)
    def wrapper(self, *args, **kwargs):
        result = df_adj(self, *args, **kwargs)

        self.update_num_cols()

        return result
    return wrapper

#endregion


class DataPrepper:
    '''
    A class containing some data inspection and analyzing functions.
    '''
#region INITIALIZE
    def __init__(self, name: str | None = None, data_file: str | None = None,
                target_col: str | None | list = None, df: pd.DataFrame | None = None):

        self.set_name = name

        # assign the initial dataframe (empty if none given)
        if df is not None:
            self.df = df
        elif data_file is not None:
            self.df = pd.read_csv(data_file, header=0, comment="#", low_memory=False)
        else:
            self.df = pd.DataFrame([])

        # clean column names
        col_names = {col: col.strip() for col in self.df.columns}
        self.df.rename(columns=col_names, inplace=True)

        # if a target_col is manually passed
        if(isinstance(target_col, list)):
            self.__target_list = True
        else:
            self.__target_list = False
        self.target_col = target_col

        # store and handle all numerical columns
        self.num_cols = None
        self._df_num = None
        self.update_num_cols()

        # if the target_col is numerical: remove it from the "original" data
        if self.__target_list:
            for col in self.target_col:
                if col in self.num_cols:
                    self.num_cols.remove(col)
        else:
            if self.target_col in self.num_cols:
                self.num_cols.remove(self.target_col)

#endregion

#region INSPECT
    #TODO: Integrate more colors
    def inspect(self, df: pd.DataFrame | None = None):
        '''
        This function prints some basic information about the dataframe. It is good in "being called first" and start the
        analyzing process. It gives the user a first idea of how the data is structured to plan further steps.
        '''
        df = self.df if df is None else df

        print(f"\n{Color.GREEN.value}=== {self.set_name}: SHAPE ==={Color.RESET.value}\n{df.shape}")

        print(f"\n{Color.BLUE.value}=== {self.set_name}: HEAD ==={Color.RESET.value}\n{df.head()}")

        print(f"\n{Color.YELLOW.value}=== {self.set_name}: INFO ==={Color.RESET.value}"); df.info()

        print(f"\n{Color.RED.value}=== {self.set_name}: MISSING ==={Color.RESET.value}")
        self.__print_missing(df)

        print(f"\n{Color.WHITE.value}=== {self.set_name}: DUPLICATES ==={Color.RESET.value}"); print(df.duplicated().sum())

        print(f"\n{Color.CYAN.value}=== {self.set_name}: INFINITY VALUES ==={Color.RESET.value}")
        inf_count = np.isinf(self.df[self.num_cols]).sum()
        print(inf_count[inf_count > 0])

        if self.num_cols:
            df_num = self._df_num

            valid_num_cols   = [c for c in self.num_cols if c in df_num.columns]
            missing_num_cols = [c for c in self.num_cols if c not in df_num.columns]

            if missing_num_cols:
                print(f"\n{Color.YELLOW.value}=== {self.set_name}: WARNING (numeric columns not in df) ==={Color.RESET.value}")
                print(missing_num_cols)

            if valid_num_cols:
                print(f"\n{Color.CYAN.value}=== {self.set_name}: DESCRIBE (numeric) ==={Color.RESET.value}\n"); return df_num[valid_num_cols].describe().transpose()
            else:
                print(f"\n{Color.CYAN.value}=== {self.set_name}: DESCRIBE === No Numeric Columns Detected.{Color.RESET.value}")
                print(f"Columns: {list(df.columns)}")
        else:
            print(f"\n{Color.CYAN.value}=== {self.set_name}: DESCRIBE === No Numeric Columns Detected.{Color.RESET.value}")
            print(f"Columns: {list(df.columns)}")

    def __print_missing(self, df: pd.DataFrame):
        '''
        This function is only class-internal. It colors the missing values depending on their amount to make the output graphically
        more appealing and easy to understand.
        '''
        total = len(df)
        missing = df.isna().sum()

        for col, n_miss in missing.items():
            if n_miss == 0:
                continue
            ratio = n_miss / total
            if ratio > 0.5:
                color = Color.RED.value
            else:
                color = Color.YELLOW.value
            print(f"{color}{col:<15}{n_miss:>5}/{total} ({ratio:.0%}){Color.RESET.value}")

    def inspect_histograms(self, df: pd.DataFrame | None = None, column: str | None = None, bins: int=20, kde: bool = True):
        '''
        This function plots a histogram for any numerical feature in the passed dataframe or for a specific column.
        '''
        df = self.df if df is None else df

        if column is None:
            num_cols = df.select_dtypes(include=[np.number]).columns
        else:
            if column not in df.columns:
                raise ValueError(f"Column '{column}' not found in DataFrame.")
            if not np.issubdtype(df[column].dtype, np.number):
                raise ValueError(f"Column '{column}' is not numerical.")
            num_cols = [column]

        for col in num_cols:
            plt.figure(figsize=(6, 4))
            sns.histplot(df[col].dropna(), bins=bins, kde=kde, color="steelblue", edgecolor="black")
            plt.title(f"Histogram of {col}")
            plt.xlabel(col)
            plt.ylabel("Frequency")
            plt.tight_layout()
            plt.show()

    def inspect_correlation(self, df: pd.DataFrame | None = None, annotate: bool = True, categories: list | None = None, big_dataset: bool = False):
        '''
        This function plots the correlation of all features (passed in the dataframe) or passed specific features to each other.
        Only numerical features are plotted.
        '''
        if df is None:
            if not hasattr(self, "df") or self.df is None:
                raise ValueError("No DataFrame provided and self.df is not set.")
            df = self.df

        if categories is not None:
            missing = [c for c in categories if c not in df.columns]
            if missing:
                raise ValueError(f"Columns not found in DataFrame: {missing}")
            df = df[categories]

        if(big_dataset):
            corr = df.corr(numeric_only = True).round(2)
            return corr.style.background_gradient(cmap = 'coolwarm', axis = None).format(precision = 2)


        plt.figure(figsize=(8, 6))
        sns.heatmap(df.corr(numeric_only=True).round(2), annot=annotate, cmap="coolwarm", fmt=".2f")
        plt.title("Correlation Matrix")
        plt.show()

    @staticmethod
    def inspect_boxplot(data: pd.DataFrame, x: str, y: str | None = None,
                    title: str = "Some cool title", hue: str | None = None):
        '''
        This function plots a boxplot for the passed dataframe.
        '''
        plt.figure(figsize=(8, 3))

        if y is None:
            sns.boxplot(x=data[x])
        else:
            if hue is None:
                hue = ""
            sns.boxplot(x=x, y=y, hue=hue, data=data)

        plt.title(title)
        plt.show()

    def inspect_outliers(
        self,
        df: Optional[pd.DataFrame] = None,
        iqr_mult: float = 1.5,
        print_result: bool = True,
        sort_by: str = "percentage",
        ascending: bool = False,
        top_n: Optional[int] = None,
        group_by: Optional[str] = None,
        return_group_stats: bool = False,
        plot: bool = False,
    ) -> pd.DataFrame | Tuple[pd.DataFrame, pd.DataFrame]:
        """
        This function identifies the outliers of the dataframe or a given subset.
        """
        df = self.df if df is None else df

        if df is None or df.empty:
            raise ValueError("DataFrame is empty or not set.")

        numeric_data = df.select_dtypes(include=["float", "int", np.number])
        if numeric_data.empty:
            raise ValueError("No numeric columns found for outlier inspection.")

        q1 = numeric_data.quantile(0.25)
        q3 = numeric_data.quantile(0.75)
        iqr = q3 - q1

        lower_bound = q1 - iqr_mult * iqr
        upper_bound = q3 + iqr_mult * iqr
        outlier_mask = (numeric_data < lower_bound) | (numeric_data > upper_bound)

        outlier_count = outlier_mask.sum()
        outlier_percentage = (outlier_mask.mean() * 100).round(2)

        overall_stats = pd.concat([outlier_count, outlier_percentage], axis=1)
        overall_stats.columns = ["Outlier Count", "Outlier Percentage"]

        if sort_by.lower().startswith("count"):
            overall_stats = overall_stats.sort_values(
                by="Outlier Count", ascending=ascending
            )
        else:
            overall_stats = overall_stats.sort_values(
                by="Outlier Percentage", ascending=ascending
            )

        if top_n is not None:
            overall_stats = overall_stats.head(top_n)

        group_stats = None
        if group_by is not None:
            if group_by not in df.columns:
                raise ValueError(f"Column '{group_by}' not found in DataFrame.")

            rows: list[dict] = []

            for feature in numeric_data.columns:
                col = df[feature]
                for grp_value, grp_idx in df.groupby(group_by).groups.items():
                    grp_values = col.loc[grp_idx].dropna()
                    if grp_values.empty:
                        continue

                    q1_g, q3_g = np.percentile(grp_values, [25, 75])
                    iqr_g = q3_g - q1_g

                    if iqr_g == 0:
                        continue

                    lower_g = q1_g - iqr_mult * iqr_g
                    upper_g = q3_g + iqr_mult * iqr_g

                    mask_g = (grp_values < lower_g) | (grp_values > upper_g)
                    num_outliers = int(mask_g.sum())
                    outlier_percent = float(num_outliers / len(grp_values) * 100)

                    rows.append(
                        {
                            "Feature": feature,
                            group_by: grp_value,
                            "Outlier Count": num_outliers,
                            "Outlier Percentage": round(outlier_percent, 2),
                        }
                    )

            if rows:
                group_stats = pd.DataFrame(rows).set_index(["Feature", group_by])

        if print_result:
            print(
                f"\n{Color.RED.value}=== Outlier statistics (IQR method, mult={iqr_mult}) ==={Color.RESET.value}"
            )
            print(overall_stats)

            if group_stats is not None:
                print(
                    f"\n{Color.RED.value}=== Grouped outlier statistics by '{group_by}' ==={Color.RESET.value}"
                )
                print(group_stats)

        if plot:
            if group_by is None:
                raise ValueError("plot=True requires 'group_by' to be set.")
            if group_stats is not None and not group_stats.empty:
                threshold = 20.0
                plot_df = group_stats[group_stats["Outlier Percentage"] > threshold].reset_index()

                if not plot_df.empty:
                    labels = (
                        plot_df["Feature"].astype(str)
                        + " - "
                        + plot_df[group_by].astype(str)
                    )

                    fig, ax = plt.subplots(figsize=(24, 10))
                    ax.bar(labels, plot_df["Outlier Percentage"])
                    ax.set_xlabel(f"Feature - {group_by}")
                    ax.set_ylabel("Percentage of Outliers")
                    ax.set_title(f"Outlier Analysis (>%{threshold})")
                    ax.set_yticks(np.arange(0, 41, 10))
                    plt.xticks(rotation=90)
                    plt.tight_layout()
                    plt.show()

        if return_group_stats:
            return overall_stats, group_stats

        return overall_stats

    def inspect_column(
    self,
    column: str,
    df: Optional[pd.DataFrame] = None,
    dropna: bool = False,
    top_n: Optional[int] = None,
    print_result: bool = True,
) -> pd.DataFrame:
        """
        Analyzes a specific column and provides the user with absolute and relative value amounts.
        """
        df = self.df if df is None else df

        if df is None or df.empty:
            raise ValueError("DataFrame is empty or not set.")

        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found in DataFrame.")

        value_counts = df[column].value_counts(dropna=dropna)

        total_count = value_counts.sum()

        percentages = (value_counts / total_count * 100).round(2)

        stats = pd.concat(
            [value_counts.rename("Count"), percentages.rename("Percentage")],
            axis=1
        )

        if top_n is not None:
            stats = stats.head(top_n)

        if print_result:
            print(f"\n=== Column inspection: '{column}' ===")
            print(stats)

        return stats

    def inspect_strong_corr_pairs(
        self,
        df: pd.DataFrame | None = None,
        corr_th: float = 0.85,
        highlight_th: float = 0.99,
        ncols: int = 4,
        exclude_last_n: int = 0,
        alpha: float = 0.1,
        base_width: int = 24,
        row_height_factor: float = 1.7
    ) -> list[tuple[float, str, str]]:
        """
        Find pairs of strongly correlated features.
        """
        df = self.df if df is None else df

        num_df = df.select_dtypes(include=[np.number])
        if num_df.empty:
            print("No numeric columns found for correlation analysis.")
            return []

        cols = list(num_df.columns)
        if exclude_last_n > 0:
            cols = cols[:-exclude_last_n]

        high_corr_pairs: list[tuple[float, str, str]] = []

        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                val = num_df[cols[i]].corr(num_df[cols[j]])
                if np.isnan(val) or val < corr_th:
                    continue
                high_corr_pairs.append((val, cols[i], cols[j]))

        if not high_corr_pairs:
            print(f"No feature pairs with correlation >= {corr_th} found.")
            return []

        size = len(high_corr_pairs)
        nrows, rem = divmod(size, ncols)
        if rem:
            nrows += 1

        fig_height = max(4, int(size * row_height_factor))
        fig, axs = plt.subplots(nrows, ncols, figsize=(base_width, fig_height))

        if nrows == 1:
            axs = np.array([axs])
        if ncols == 1:
            axs = axs.reshape(-1, 1)

        idx = 0
        for i in range(nrows):
            for j in range(ncols):
                if idx >= size:
                    fig.delaxes(axs[i, j])
                    continue

                val, x, y = high_corr_pairs[idx]
                color = "green" if val > highlight_th else "blue"

                axs[i, j].scatter(num_df[x], num_df[y], color=color, alpha=alpha)
                axs[i, j].set_xlabel(x)
                axs[i, j].set_ylabel(y)
                axs[i, j].set_title(f"{x} vs\n{y} ({val:.2f})")

                idx += 1

        fig.tight_layout()
        plt.show()

        return high_corr_pairs

    def inspect_filter_features(
        self,
        target_col: str,
        df: pd.DataFrame | None = None,
        min_corr: float = 0.0,
        max_corr: float = 1.0
    ) -> dict[str, list[str]]:
        """
        Identifies and visualizes:
        - Features with positive correlation towards the target_col (in a interval)
        - Features with exactly 0 variance (std == 0) --> these are completely irrelevant
        - Features with correlation exactly 0 --> no linear similarity
        """
        df = self.df if df is None else df

        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in DataFrame.")

        df_num = df.select_dtypes(include=[np.number])
        if target_col not in df_num.columns:
            raise ValueError(f"Target column '{target_col}' is not numeric or not in numeric subset.")

        std = df_num.std(numeric_only=True)
        zero_std_cols = std[std == 0].index.tolist()
        zero_std_cols = [c for c in zero_std_cols if c != target_col]

        corr = df_num.corr()

        if target_col not in corr.columns:
            raise ValueError(f"Target column '{target_col}' is not in correlation matrix.")

        target_series = corr[target_col]

        mask_pos = (target_series > min_corr) & (target_series < max_corr)
        pos_corr = target_series[mask_pos].sort_values(ascending=False)

        zero_corr_cols = target_series[target_series == 0].index.tolist()
        zero_corr_cols = [c for c in zero_corr_cols if c != target_col]

        print(f"\n{Color.GREEN.value}=== Positive correlation with '{target_col}' ==={Color.RESET.value}\n")
        if not pos_corr.empty:
            for i, (feature, value) in enumerate(pos_corr.items(), start=1):
                print('{:<3} {:<24} :{:.3f}'.format(f'{i}.', feature, value))
        else:
            print("No positively correlated features in the given range.")

        print(f"\n{Color.YELLOW.value}=== Zero-variance features (std == 0) ==={Color.RESET.value}\n")
        if zero_std_cols:
            for col in zero_std_cols:
                print(f"- {col}")
        else:
            print("No zero-variance features detected.")

        print(f"\n{Color.CYAN.value}=== Zero-correlation features with '{target_col}' (corr == 0) ==={Color.RESET.value}\n")
        if zero_corr_cols:
            for col in zero_corr_cols:
                print(f"- {col}")
        else:
            print("No features with exactly zero correlation detected.")

        return {
            "positive_corr": list(pos_corr.index),
            "zero_std": zero_std_cols,
            "zero_corr": zero_corr_cols,
        }

#endregion

#region MISSING
    @adj_self_numerical
    def drop_uniques(
    self,
    df: Optional[pd.DataFrame] = None,
    print_result: bool = True
    ) -> list[str]:
        """
        Removes all columns that carry only one unique value.
        """
        df = self.df if df is None else df

        if df is None or df.empty:
            raise ValueError("DataFrame is empty or not set.")

        nunique = df.nunique()
        drop_cols = nunique[nunique == 1].index.tolist()

        keep_cols = nunique[nunique > 1].index.tolist()
        self.df = df[keep_cols]

        if print_result:
            print("Dropped columns:", drop_cols)

        return drop_cols

    @use_self_df
    def missing_kill(self, df: pd.DataFrame | None = None, col: str | None = None) -> pd.DataFrame:
        '''
        This function simply kills all rows, that have a missing value in the passed column.
        '''

        before = len(df)

        if col:
            df = df.dropna(subset=[col]).copy()
        else:
            df = df.dropna().copy()

        after = len(df)

        print(f"{Color.GREEN.value}Removed {before - after} rows containing NaN values in \"{col}\" ({after} remain).{Color.RESET.value}")
        return df

    @use_self_df
    def missing_forceVal(self, df: pd.DataFrame | None = None, col: str | None = None, value=0) -> pd.DataFrame:
        '''
        This function simply fills missing values in the passed column, with an also manually passed value.
        '''
        if col:
            n_missing = df[col].isna().sum()
            df[col] = df[col].fillna(value)
        else:
            n_missing = df.isna().sum().sum()
            df = df.fillna(value)

        print(f"{Color.GREEN.value}Replaced {n_missing} NaN values of \"{col}\" with {value}.{Color.RESET.value}")
        return df

    @use_self_df
    def missing_mean(self, df: pd.DataFrame | None = None, col: str | None = None) -> pd.DataFrame:
        '''
        This column fills all missing values in the passed column with the mean-value of the underlying column.
        '''

        if col:
            if df[col].dtype.kind in "biufc":
                n_missing = df[col].isna().sum()
                df[col] = df[col].fillna(df[col].mean())
            else:
                print(f"{Color.RED.value}Column '{col}' is not numeric. Skipped.{Color.RESET.value}")
                return df
        else:
            num_cols = df.select_dtypes(include=["number"]).columns
            n_missing = df[num_cols].isna().sum().sum()
            df[num_cols] = df[num_cols].fillna(df[num_cols].mean())

        print(f"{Color.GREEN.value}Replaced {n_missing} numeric NaN values in \"{col}\" with mean {df[col].mean()}.{Color.RESET.value}")
        return df

    @use_self_df
    def missing_median(self, df: pd.DataFrame | None = None, col: str | None = None) -> pd.DataFrame:
        '''
        This column fills all missing values in the passed column with the median-value of the underlying column.
        '''
        if col:
            if df[col].dtype.kind in "biufc":
                n_missing = df[col].isna().sum()
                df[col] = df[col].fillna(df[col].median())
            else:
                print(f"{Color.RED.value}Column '{col}' is not numeric. Skipped.{Color.RESET.value}")
                return df
        else:
            num_cols = df.select_dtypes(include=["number"]).columns
            n_missing = df[num_cols].isna().sum().sum()
            df[num_cols] = df[num_cols].fillna(df[num_cols].median())

        print(f"{Color.GREEN.value}Replaced {n_missing} numeric NaN values in \"{col}\" with median {df[col].median()}.{Color.RESET.value}")
        return df

#endregion

#region SCALING_FUNCS
    def apply_minMax(self, df: pd.DataFrame | None = None, col: str | None = None, feature_range=(0,1)) -> pd.DataFrame:
        '''
        This function applies the MinMaxScaler() on the passed column.
        '''
        df = self.df if df is None else df

        if not col or col not in df.columns:
            print(f"{Color.RED.value}Invalid or missing column name: {col}{Color.RESET.value}")
            return df

        data = df[[col]].astype(float)

        scaler = MinMaxScaler(feature_range=feature_range)

        scaled = scaler.fit_transform(data)

        df[col] = scaled

        print(f"{Color.GREEN.value}Scaled '{col}' with MinMaxScaler (range={feature_range}){Color.RESET.value}")
        return df

    def apply_standard(self, df: pd.DataFrame | None = None, col: str | None = None) -> pd.DataFrame:
        '''
        This function applies the StandardScaler() on the passed column.
        '''
        df = self.df if df is None else df

        if not col or col not in df.columns:
            print(f"{Color.RED.value}Invalid or missing column name: {col}{Color.RESET.value}")
            return df

        data = df[[col]].astype(float)

        scaler = StandardScaler()

        scaled = scaler.fit_transform(data)

        df[col] = scaled

        print(f"{Color.GREEN.value}Scaled '{col}' with StandardScaler{Color.RESET.value}")
        return df

    def apply_robust(self, df: pd.DataFrame | None = None, col: str | None = None, quantile_range=(25.0, 75.0)) -> pd.DataFrame:
        '''
        This function applies the RobustScaler() to the passed column.
        '''
        df = self.df if df is None else df

        if not col or col not in df.columns:
            print(f"{Color.RED.value}Invalid or missing column name: {col}{Color.RESET.value}")
            return df

        data = df[[col]].astype(float)

        scaler = RobustScaler(quantile_range=quantile_range)

        scaled = scaler.fit_transform(data)
        df[col] = scaled

        print(f"{Color.GREEN.value}Scaled '{col}' with RobusScaler (quantile_range:{quantile_range}){Color.RESET.value}")
        return df

    def apply_binarize(self, df: pd.DataFrame | None = None, col: str | None = None, threshold=5.0) -> pd.DataFrame:
        '''
        This function applies the Binarizer() to the passed column.
        '''
        df = self.df if df is None else df

        if not col or col not in df.columns:
            print(f"{Color.RED.value}Invalid or missing column name: {col}{Color.RESET.value}")
            return df

        data = df[[col]].astype(float)

        scaler = Binarizer(threshold=threshold)

        scaled = scaler.fit_transform(data)

        df[col] = scaled

        print(f"{Color.GREEN.value}Scaled '{col}' with Binarizer (threshold={threshold}){Color.RESET.value}")
        return df
#endregion

#region ENCODERS
    @use_self_df
    def apply_onehot(self, df: pd.DataFrame | None = None, col: str | None = None) -> pd.DataFrame:
        '''
        This function applies one-hot encoding to the passed column.
        '''
        if not col or col not in df.columns:
            print(f"{Color.RED.value}Invalid or missing column name: {col}{Color.RESET.value}")
            return df

        encoded = pd.get_dummies(df[col].astype("category"), prefix=col)

        df = pd.concat([df.drop(columns=[col]), encoded], axis=1)

        print(f"{Color.GREEN.value}Applied One-Hot-Encoding on '{col}' && created {len(encoded.columns)} columns{Color.RESET.value}")

        return df

    @adj_self_numerical
    def apply_labelEncoding(self, df: pd.DataFrame | None = None, col: str | None = None) -> pd.DataFrame:
        '''
        This function applies label-encoding to the passed column.
        '''
        df = self.df if df is None else df

        if not col or col not in df.columns:
            print(f"{Color.RED.value}Invalid or missing column name: {col}{Color.RESET.value}")
            return df

        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

        print(f"{Color.GREEN.value}Applied Label-Encoding on '{col}' ({len(le.classes_)} unique classes){Color.RESET.value}")
        return df

    @use_self_df
    @adj_self_numerical
    def apply_ordinal(self, df: pd.DataFrame | None = None, col: str | None = None,
                    categories=None, new_col: str | None = None) -> pd.DataFrame:
        '''
        This function applies ordinal-encoding to the passed column.
        '''
        if not col or col not in df.columns:
            print(f"{Color.RED.value}Invalid or missing column name: {col}{Color.RESET.value}")
            return df

        if categories is None:
            cats = sorted(df[col].dropna().unique().tolist())
            categories = [cats]
            print(f"{Color.YELLOW.value}No category order specified for '{col}', using {cats}.{Color.RESET.value}")

        oe = OrdinalEncoder(categories=categories)
        encoded = oe.fit_transform(df[[col]]).ravel()

        new_col = new_col or f"{col}_ord"
        df[new_col] = encoded

        print(f"{Color.GREEN.value}Applied Ordinal-Encoding on '{col}' → '{new_col}' with categories: {categories[0]}{Color.RESET.value}")
        return df

#endregion

#region EVALUATE
    @staticmethod
    def evaluate_importance(pipe: Pipeline, y: pd.DataFrame | None = None, steps: list | None = None):
        '''
        This function evaluates and plots the most important features in a dataset using a passed pipeline
        '''
        if steps is None:
            steps = ["pre", "clf"]

        feature_names = pipe.named_steps[steps[0]].get_feature_names_out()

        importances = pipe.named_steps[steps[1]].feature_importances_

        feat_imp = pd.DataFrame({"Feature": feature_names, "Importance": importances})
        feat_imp = feat_imp.sort_values(by="Importance", ascending=False)

        plt.figure(figsize=(10, 6))
        sns.barplot(data=feat_imp.head(15), x="Importance", y="Feature", palette="viridis", hue=y, legend=True)
        plt.title("Top Feature Importances")
        plt.tight_layout()
        plt.show()


    @staticmethod
    def evaluate_confusion(y_test:pd.DataFrame, y_pred:pd.DataFrame):
        '''
        This function plots a confusion matrix of given test vs predicted values
        '''
        plt.figure(figsize=(6, 5))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()

    @staticmethod
    def evaluate_relationships(train_set: pd.DataFrame, features: list, hue: str):
        '''
        This function plots relationships between paired features in a dataset.
        '''
        if train_set is None or features is None or hue is None:
            print("No valid input values passed.")

        sns.pairplot(train_set, vars=features, hue=hue)
        plt.show()

#endregion

#region UTIL

    def update_num_cols(self):
        '''
        This function updates the list of numerical columns, depending on numerical columns in the dataset.
        '''
        num_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        if not num_cols:
            _df_num = self.df.apply(
                lambda s: pd.to_numeric(s, errors="coerce") if s.dtype == "object" else s
            )
            num_cols = _df_num.select_dtypes(include=[np.number]).columns.tolist()
        else:
            _df_num = self.df

        self.num_cols = num_cols
        self._df_num = _df_num

#endregion


if __name__=="__main__":
    pass