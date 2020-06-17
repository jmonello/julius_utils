import numpy as np
import pandas as pd
import seaborn as sns
from copy import deepcopy

def data_exploration_plot(
    df: pd.DataFrame, save_to_file: Union[str, None] = None, plot=True
):
    for i, col in enumerate(df.columns):
        if not str(df[col].dtype).startswith(("object", "datetime64")):
            plt.figure(i)
            sns.distplot(df[col].dropna(), kde=False, rug=True)
        else:
            if not str(df[col].dtype).startswith(("datetime64")):
                print(str(df[col].dtype))
                if df[col].nunique() < 20:
                    sns.countplot(x=df[col].values)

        if save_to_file is not None:
            plt.savefig(save_to_file)

        if plot:
            plt.show()


def data_correlation_plot(
    df_orig: pd.DataFrame, save_to_file: Union[str, None] = None, plot=True
):
    sns.set(rc={"figure.figsize": (16, 10)})

    df = deepcopy(df_orig)

    for i, col in enumerate(df.columns):
        df[col] = df[col].astype(float, errors="ignore")

    corr = df.corr()
    corr = corr.round(1)
    mask = np.zeros_like(corr)
    mask[np.triu_indices_from(mask)] = True

    with sns.axes_style("white"):
        sns.heatmap(
            corr,
            mask=mask,
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values,
            center=0,
            cbar=False,
            annot=True,
        )

    if save_to_file is not None:
        plt.savefig(save_to_file)

    if plot:
        plt.show()
