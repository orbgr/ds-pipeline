from dataclasses import dataclass


@dataclass
class FS:
    sfa_methods: list
    mfa_methods: list
    masks: dict


sfa_methods = ["gini", "coverage", "uniq", "correlation"]
mfa_methods = []
masks = dict(gini=lambda df: df.loc[df["gini"] > 0.04],
             coverage=lambda df: df.loc[df["coverage"] > 0.25])

cfg_fs = FS(sfa_methods=sfa_methods,
            mfa_methods=mfa_methods,
            masks=masks
            )