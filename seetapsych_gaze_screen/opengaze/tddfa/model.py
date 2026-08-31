# -*- coding: utf-8 -*-

import os.path

from seetapsych_lib import api

ROOT = os.path.dirname(os.path.abspath(__file__))


class TDDFAModel(api.Model):
    def __init__(self, resource_root: str):
        self.__resource_root = resource_root
        self.__path = os.path.join(ROOT, resource_root)

        self.__bfm_noneck_name = "bfm-noneck-v3.pkl"
        self.__param_norm_name = "param-mean-std.pkl"
        self.__tddfa_onnx_name = "tddfa-v2-mb1.onnx"

    def exists(self) -> bool:
        return (
            os.path.exists(os.path.join(self.__path, self.__bfm_noneck_name))
            and os.path.exists(os.path.join(self.__path, self.__param_norm_name))
            and os.path.exists(os.path.join(self.__path, self.__tddfa_onnx_name))
        )

    def cache(self) -> str:
        if self.exists():
            return self.__path

        files = ", ".join([self.__bfm_noneck_name, self.__param_norm_name, self.__tddfa_onnx_name])
        raise RuntimeError(
            f"Unable to download on my own. Need to contact the developer to obtain "
            f"{files} and place it in the directory {self.__path}"
        )


def load() -> api.Model:
    return TDDFAModel("3ddfa-v2")


def main():
    pass


if __name__ == "__main__":
    main()
